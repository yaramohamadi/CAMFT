"""Generator-initialized scalar discriminator for SiT-DMF MeFT checkpoints."""

import jax
import jax.numpy as jnp

from models.imfDiT import imfSiT_DMF
from models.torch_models import RMSNorm, TorchLinear


class SiTMeFTDiscriminator(imfSiT_DMF):
    """Reuse the SiT-DMF backbone and replace image output by D(x_s, s, y)."""

    def setup(self):
        super().setup()
        self.dis_norm = RMSNorm(self.hidden_size)
        self.dis_head = TorchLinear(
            self.hidden_size,
            1,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def __call__(
        self,
        x,
        point_time,
        interval_start=None,
        interval_end=None,
        y=None,
    ):
        # AFM calls D(x, s, y); CA-iMF calls D(x, s, t, r, y).
        if y is None and interval_end is None and interval_start is not None:
            y = interval_start
        del interval_start, interval_end
        batch_size = x.shape[0]
        point_time = jnp.asarray(point_time, dtype=x.dtype).reshape(batch_size)
        if y is None:
            y = jnp.zeros((batch_size,), dtype=jnp.int32)
        y = jnp.asarray(y, dtype=jnp.int32).reshape(batch_size)

        sequence = self.x_embedder(x) + self.pos_embed
        unit_guidance = jnp.ones_like(point_time)
        zero = jnp.zeros_like(point_time)
        one = jnp.ones_like(point_time)

        guidance_tokens = None
        if self.use_context_guidance_conditioning:
            guidance_tokens = self._build_guidance_context_tokens(
                unit_guidance, zero, one
            )
            if not self.decoder_only_guidance_conditioning:
                sequence = jnp.concatenate([guidance_tokens, sequence], axis=1)

        y_embed = self.y_embedder(y)
        time_embed = self.t_embedder(point_time)
        if self.use_adaln_condition_mixing:
            omega_feature = jnp.zeros_like(point_time)
            omega_embed = self.omega_embedder(omega_feature)
            time_condition = self.time_condition_projector(time_embed, time_embed)
            class_condition = self.class_condition_projector(y_embed, omega_embed)
            encoder_condition = time_condition + class_condition
            decoder_condition = encoder_condition
        elif self.time_conditioning_mode == "both":
            shared_time = self.time_condition_projector(time_embed, time_embed)
            encoder_condition = shared_time + y_embed
            decoder_condition = encoder_condition
        else:
            encoder_condition = time_embed + y_embed
            decoder_condition = encoder_condition

        if (
            self.use_adaln_guidance_scale_conditioning
            and not self.use_adaln_condition_mixing
        ):
            omega_embed = self.omega_embedder(jnp.zeros_like(point_time))
            omega_delta = omega_embed * jax.lax.stop_gradient(y_embed)
            decoder_condition = decoder_condition + omega_delta
            if not self.decoder_only_guidance_conditioning:
                encoder_condition = encoder_condition + omega_delta

        for block in self.encoder_blocks:
            sequence = block(sequence, encoder_condition)

        if guidance_tokens is not None and self.decoder_only_guidance_conditioning:
            sequence = jnp.concatenate([guidance_tokens, sequence], axis=1)

        for block in self.decoder_blocks:
            sequence = block(sequence, decoder_condition)

        if self.prefix_tokens:
            sequence = sequence[:, self.prefix_tokens :]
        pooled = jnp.mean(sequence, axis=1)
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_sit_meft_discriminator(generator_net):
    """Build a scalar D with names/shapes compatible with a SiT-DMF generator."""
    if not isinstance(generator_net, imfSiT_DMF):
        raise ValueError(
            "SiT-MeFT adversarial training requires an imfSiT_DMF generator."
        )
    return SiTMeFTDiscriminator(
        input_size=generator_net.input_size,
        patch_size=generator_net.patch_size,
        in_channels=generator_net.in_channels,
        hidden_size=generator_net.hidden_size,
        encoder_depth=generator_net.encoder_depth,
        decoder_depth=generator_net.decoder_depth,
        num_heads=generator_net.num_heads,
        mlp_ratio=generator_net.mlp_ratio,
        num_classes=generator_net.num_classes,
        use_null_class=generator_net.use_null_class,
        use_context_guidance_conditioning=(
            generator_net.use_context_guidance_conditioning
        ),
        use_adaln_guidance_scale_conditioning=(
            generator_net.use_adaln_guidance_scale_conditioning
        ),
        adaln_guidance_scale_init=generator_net.adaln_guidance_scale_init,
        use_adaln_condition_mixing=generator_net.use_adaln_condition_mixing,
        decoder_only_guidance_conditioning=(
            generator_net.decoder_only_guidance_conditioning
        ),
        time_conditioning_mode=generator_net.time_conditioning_mode,
        num_cfg_tokens=generator_net.num_cfg_tokens,
        num_interval_tokens=generator_net.num_interval_tokens,
        eval=False,
        weight_init=generator_net.weight_init,
        weight_init_constant=generator_net.weight_init_constant,
    )
