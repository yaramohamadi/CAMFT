"""Plain DiT wrapper using the original diffusion objective."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import imfDiT
from utils.dit_diffusion import create_diffusion


class PlainDiT(nn.Module):
    """JAX/Flax DiT backbone trained with the original DiT DDPM objective."""

    model_str: str
    dtype: jnp.dtype = jnp.float32
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"
    learn_sigma: bool = True
    predict_xstart: bool = False
    rescale_learned_sigmas: bool = False
    output_prediction_space: str = "epsilon"
    eval: bool = False

    def setup(self):
        if not self.model_str.startswith("flaxDiT"):
            raise ValueError(f"PlainDiT expects a flaxDiT_* backbone, got {self.model_str!r}.")

        net_fn = getattr(imfDiT, self.model_str)
        self.net = net_fn(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            learn_sigma=self.learn_sigma,
            return_learned_sigma=True,
            eval=self.eval,
        )
        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule=self.noise_schedule,
            learn_sigma=self.learn_sigma,
            predict_xstart=self.predict_xstart,
            rescale_learned_sigmas=self.rescale_learned_sigmas,
            diffusion_steps=self.diffusion_steps,
        )

    def _drop_labels(self, labels, rng):
        if (
            (not self.target_use_null_class)
            or self.class_dropout_prob <= 0.0
            or self.eval
        ):
            return labels

        drop_mask = jax.random.uniform(rng, labels.shape, dtype=jnp.float32)
        drop_mask = drop_mask < self.class_dropout_prob
        null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
        return jnp.where(drop_mask, null_labels, labels)

    def _validate_output_prediction_space(self):
        if self.output_prediction_space not in ("epsilon", "velocity"):
            raise ValueError(
                "PlainDiT output_prediction_space must be 'epsilon' or 'velocity', "
                f"got {self.output_prediction_space!r}."
            )

    def _network_output(self, x, t, labels):
        return self.net(x.astype(self.dtype), t.astype(jnp.float32), labels)

    def _split_prediction_and_variance(self, model_output):
        return jnp.split(model_output, 2, axis=-1)

    def _wrap_prediction(self, x_t, t, raw_prediction):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "epsilon":
            return raw_prediction
        return self.diffusion.predict_velocity_from_eps(x_t, t, raw_prediction)

    def _unwrap_prediction(self, x_t, t, wrapped_prediction):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "epsilon":
            return wrapped_prediction
        return self.diffusion.predict_eps_from_velocity(x_t, t, wrapped_prediction)

    def predict_wrapped_output(self, x, t, y):
        model_output = self._network_output(x, t, y)
        raw_prediction, model_var_values = self._split_prediction_and_variance(model_output)
        wrapped_prediction = self._wrap_prediction(x, t.astype(jnp.int32), raw_prediction)
        return jnp.concatenate([wrapped_prediction, model_var_values], axis=-1)

    def forward(self, images, labels):
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_t, rng_noise = jax.random.split(self.make_rng("gen"), 3)
        labels = self._drop_labels(labels, rng_drop)
        t = jax.random.randint(
            rng_t,
            (x.shape[0],),
            minval=0,
            maxval=self.diffusion.num_timesteps,
            dtype=jnp.int32,
        )
        noise = jax.random.normal(rng_noise, x.shape, dtype=jnp.float32)
        x_t = self.diffusion.q_sample(x, t, noise=noise)

        model_output = self._network_output(x_t, t, labels).astype(jnp.float32)
        raw_prediction, model_var_values = self._split_prediction_and_variance(model_output)
        wrapped_prediction = self._wrap_prediction(x_t, t, raw_prediction)

        frozen_out = jnp.concatenate(
            [jax.lax.stop_gradient(raw_prediction), model_var_values],
            axis=-1,
        )
        vb = self.diffusion._vb_terms_bpd(
            model_fn=lambda _x, _t: frozen_out,
            x_start=x.astype(jnp.float32),
            x_t=x_t.astype(jnp.float32),
            t=t,
            clip_denoised=False,
        )["output"]
        target = (
            noise
            if self.output_prediction_space == "epsilon"
            else self.diffusion.predict_velocity_from_eps(x_t, t, noise)
        )
        mse = jnp.mean((target - wrapped_prediction) ** 2, axis=(1, 2, 3))
        loss = jnp.mean(mse + vb)
        dict_losses = {
            "loss": loss,
            "loss_diffusion": loss,
            "mse": jnp.mean(mse),
            "vb": jnp.mean(vb),
            "t_mean": jnp.mean(t.astype(jnp.float32)),
        }
        return loss, dict_losses

    def __call__(self, x, t, y):
        return self.predict_wrapped_output(x, t, y)
