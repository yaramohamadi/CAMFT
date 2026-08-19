"""Plain JiT wrapper using the official pixel-space denoising objective."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import jit


class PlainJiT(nn.Module):
    """JAX/Flax JiT denoiser trained in pixel space.

    The wrapped network predicts clean images. The training loss follows the
    official JiT denoiser by converting both target and prediction into velocity
    with ``(x - z) / (1 - t)``.
    """

    model_str: str
    dtype: jnp.dtype = jnp.float32
    input_size: int = 256
    in_channels: int = 3
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    P_mean: float = -0.8
    P_std: float = 0.8
    t_eps: float = 1e-5
    noise_scale: float = 1.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    eval: bool = False

    def setup(self):
        if not self.model_str.startswith("flaxJiT"):
            raise ValueError(f"PlainJiT expects a flaxJiT_* backbone, got {self.model_str!r}.")

        net_fn = getattr(jit, self.model_str)
        self.net = net_fn(
            name="net",
            input_size=self.input_size,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            eval=self.eval,
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

    def _sample_t(self, rng, batch_size):
        z = jax.random.normal(rng, (batch_size,), dtype=jnp.float32)
        z = z * self.P_std + self.P_mean
        return jax.nn.sigmoid(z)

    def _time_to_image_shape(self, t, ndim):
        return t.reshape((t.shape[0],) + (1,) * (ndim - 1))

    def _predict_velocity(self, z, t, labels):
        t_img = self._time_to_image_shape(t, z.ndim)
        denom = jnp.maximum(1.0 - t_img, self.t_eps)
        x_pred = self.net(z.astype(self.dtype), t.astype(self.dtype), labels)
        return (x_pred - z) / denom

    def forward(self, images, labels):
        """Compute the official JiT training loss on BHWC pixel images."""
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_t, rng_noise = jax.random.split(self.make_rng("gen"), 3)
        labels = self._drop_labels(labels, rng_drop)

        t = self._sample_t(rng_t, x.shape[0])
        t_img = self._time_to_image_shape(t, x.ndim)
        noise = jax.random.normal(rng_noise, x.shape, dtype=x.dtype) * self.noise_scale
        z = t_img * x + (1.0 - t_img) * noise

        denom = jnp.maximum(1.0 - t_img, self.t_eps)
        target_velocity = (x - z) / denom
        pred_velocity = self._predict_velocity(z, t, labels)

        per_example_loss = jnp.mean(
            jnp.square(target_velocity - pred_velocity),
            axis=tuple(range(1, x.ndim)),
        )
        loss = jnp.mean(per_example_loss)
        dict_losses = {
            "loss": loss,
            "loss_jit": loss,
            "t_mean": jnp.mean(t),
            "t_min": jnp.min(t),
            "t_max": jnp.max(t),
        }
        return loss, dict_losses

    def forward_sample(self, z, t, labels, cfg_scale=1.0, interval_min=0.0, interval_max=1.0):
        """Predict classifier-free-guided velocity for sampling."""
        labels = labels.astype(jnp.int32)
        t = t.reshape((z.shape[0],)).astype(self.dtype)
        v_cond = self._predict_velocity(z, t, labels)

        if not self.target_use_null_class:
            return v_cond

        null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
        v_uncond = self._predict_velocity(z, t, null_labels)
        t_img = self._time_to_image_shape(t, z.ndim)
        interval_mask = (t_img < interval_max) & (
            (interval_min == 0.0) | (t_img > interval_min)
        )
        scale = jnp.where(interval_mask, cfg_scale, 1.0).astype(z.dtype)
        return v_uncond + scale * (v_cond - v_uncond)

    def euler_step(self, z, t, t_next, labels, cfg_scale=1.0, interval_min=0.0, interval_max=1.0):
        v_pred = self.forward_sample(z, t, labels, cfg_scale, interval_min, interval_max)
        step = self._time_to_image_shape(t_next - t, z.ndim)
        return z + step * v_pred

    def heun_step(self, z, t, t_next, labels, cfg_scale=1.0, interval_min=0.0, interval_max=1.0):
        v_pred_t = self.forward_sample(z, t, labels, cfg_scale, interval_min, interval_max)
        step = self._time_to_image_shape(t_next - t, z.ndim)
        z_next_euler = z + step * v_pred_t
        v_pred_t_next = self.forward_sample(
            z_next_euler,
            t_next,
            labels,
            cfg_scale,
            interval_min,
            interval_max,
        )
        return z + step * 0.5 * (v_pred_t + v_pred_t_next)

    def __call__(self, x, t, y):
        """Initialization/evaluation forward that mirrors the JiT backbone."""
        return self.net(x.astype(self.dtype), t.astype(self.dtype), y.astype(jnp.int32))
