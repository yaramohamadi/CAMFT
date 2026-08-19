"""Shared model and forward-time terms for adversarial SiT-MeFT training."""

import dataclasses

import jax
import jax.numpy as jnp
from jax import random

from afm_sit_meft import (
    generated_future_endpoint,
    sample_time_pairs,
    sit_linear_path,
)
from imf import iMeanFlow
from models import imfDiT
from models.sit_meft_discriminator import create_sit_meft_discriminator


class SiTMeFTAdversarialFlow(iMeanFlow):
    """iMeanFlow with forward-time adversarial helpers for SiT-DMF."""

    compute_adversarial_retention: bool = False

    def afm_u_fn(self, x, r, t, omega, t_min, t_max, y):
        if not self._uses_sit_dmf_time_convention():
            raise ValueError("This path requires a SiT-DMF MeFT backbone.")
        return self._predict_target_velocity(
            x, t, r, omega, t_min, t_max, y
        )

    def _sit_adversarial_samples(
        self, images, labels, current_step, interval_eps
    ):
        x = images.astype(self.dtype)
        batch_size = x.shape[0]
        t, r, _ = self.sample_split_tr(batch_size)
        eps = jnp.asarray(interval_eps, self.dtype)
        t = jnp.minimum(t, 1.0 - eps)
        r = jnp.maximum(r, t + eps)
        r = jnp.minimum(r, 1.0)

        noise = random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        x_t = (1.0 - t) * noise + t * x
        x_r = (1.0 - r) * noise + r * x
        no_diagonal = jnp.zeros_like(t, dtype=bool)
        t_min, t_max = self.sample_cfg_interval(batch_size, no_diagonal)
        omega = self._sample_guidance_scale(batch_size)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )
        labels, _ = self.cond_drop(x_t, x_t, labels)
        u = self._predict_target_velocity(
            x_t, t, r, model_omega, t_min, t_max, labels
        )
        xhat_r = x_t + (r - t) * u
        return {
            "u": u,
            "x_t": x_t,
            "x_r": x_r,
            "xhat_r": xhat_r,
            "t": t.reshape(batch_size),
            "r": r.reshape(batch_size),
            "labels": labels,
        }

    def forward_caimf_discriminator_samples(
        self, images, labels, current_step=None, interval_eps=1e-3
    ):
        return self._sit_adversarial_samples(
            images, labels, current_step, interval_eps
        )

    def forward_caimf_generator_terms(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
        interval_eps=1e-3,
    ):
        terms = self._sit_adversarial_samples(
            images, labels, current_step, interval_eps
        )
        zero = jnp.asarray(0.0, dtype=images.dtype)
        loss_imf, loss_u, loss_v = zero, zero, zero
        if self.compute_adversarial_retention:
            loss_imf, metrics = self.forward_imf_jvp(
                images,
                labels,
                source_params=source_params,
                teacher_params=teacher_params,
                current_step=current_step,
            )
            loss_u = metrics["loss_u"]
            loss_v = metrics["loss_v"]
        terms.update(
            {"loss_imf": loss_imf, "loss_u": loss_u, "loss_v": loss_v}
        )
        return terms


def create_models(config, *, retention_weight=0.0):
    """Create the exact MeFT generator wrapper and matching scalar D."""
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model_config = {
        key: value for key, value in model_config.items() if key in valid_keys
    }
    model = SiTMeFTAdversarialFlow(
        **model_config,
        compute_adversarial_retention=float(retention_weight) > 0.0,
    )
    model_str = str(config.model.model_str)
    # DiT-DMF shares the SiT-DMF backbone class (imfDiT_DMF_XL_2 = imfSiT_DMF_XL_2),
    # so the same discriminator + adversarial forward apply. Accept both prefixes.
    if not (model_str.startswith("imfSiT_DMF_") or model_str.startswith("imfDiT_DMF_")):
        raise ValueError(
            "Expected model.model_str=imf{SiT,DiT}_DMF_* from a MeFT checkpoint; "
            f"got {model_str!r}."
        )
    net_fn = getattr(imfDiT, model_str)
    generator_net = net_fn(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        use_context_guidance_conditioning=bool(
            config.model.get("use_context_guidance_conditioning", False)
        ),
        use_adaln_guidance_scale_conditioning=bool(
            config.model.get("use_adaln_guidance_scale_conditioning", False)
        ),
        adaln_guidance_scale_init=str(
            config.model.get("adaln_guidance_scale_init", "timestep")
        ),
        use_adaln_condition_mixing=bool(
            config.model.get("use_adaln_condition_mixing", False)
        ),
        decoder_only_guidance_conditioning=bool(
            config.model.get("decoder_only_guidance_conditioning", False)
        ),
        time_conditioning_mode=str(
            config.model.get("time_conditioning_mode", "split")
        ),
        eval=False,
    )
    return model, generator_net, create_sit_meft_discriminator(generator_net)


def make_afm_endpoint_terms(model, params, images, labels, rng, config):
    """Construct real/fake future endpoints using SiT's noise-to-data path."""
    rng_time, rng_noise = random.split(rng)
    p_r_one = float(
        config.afm.get("p_r_one", config.afm.get("p_r_zero", 0.25))
    )
    t, r, interval, boundary_mask = sample_time_pairs(
        rng_time,
        images.shape[0],
        min_interval=float(config.afm.min_interval),
        p_r_one=p_r_one,
    )
    noise = random.normal(rng_noise, images.shape, dtype=images.dtype)
    x_t = sit_linear_path(images, noise, t)
    x_r = sit_linear_path(images, noise, r)
    omega = jnp.full(
        (images.shape[0],), float(config.sampling.omega), images.dtype
    )
    t_min = jnp.full(
        (images.shape[0],), float(config.sampling.t_min), images.dtype
    )
    t_max = jnp.full(
        (images.shape[0],), float(config.sampling.t_max), images.dtype
    )
    u = model.apply(
        {"params": params},
        x_t,
        r,
        t,
        omega,
        t_min,
        t_max,
        labels,
        method=model.afm_u_fn,
    )
    return {
        "x_t": x_t,
        "x_r": x_r,
        "x_r_fake": generated_future_endpoint(x_t, u, t, r),
        "u": u,
        "r": r,
        "t": t,
        "interval": interval,
        # Kept under the harness's historical key for metric compatibility.
        "zero_mask": boundary_mask,
        "labels": labels,
    }


def mask_sit_discriminator_grads(
    grads, discriminator, trainable_blocks, freeze_backbone=False
):
    """Mask a SiT D by encoder/decoder block depth or to scalar head only."""
    from flax import serialization

    if trainable_blocks < 0 and not freeze_backbone:
        return grads
    state = serialization.to_state_dict(grads)
    total_depth = discriminator.encoder_depth + discriminator.decoder_depth
    first_trainable = max(total_depth - trainable_blocks, 0)

    def block_index(name):
        for prefix, offset in (
            ("encoder_blocks_", 0),
            ("decoder_blocks_", discriminator.encoder_depth),
        ):
            if name.startswith(prefix):
                try:
                    return offset + int(name[len(prefix) :])
                except ValueError:
                    return None
        return None

    def mask(tree, path=()):
        if isinstance(tree, dict):
            return {key: mask(value, path + (key,)) for key, value in tree.items()}
        top = path[0] if path else ""
        if freeze_backbone:
            train = top in {"dis_norm", "dis_head"}
        else:
            index = block_index(top)
            train = top in {"dis_norm", "dis_head", "t_embedder", "y_embedder"}
            train = train or (index is not None and index >= first_trainable)
        return tree if train else jnp.zeros_like(tree)

    return serialization.from_state_dict(grads, mask(state))
