"""Endpoint-only discriminator for AFM post-training of improved MeanFlow."""

import jax.numpy as jnp

from models.imfDiT import imfDiT
from models.torch_models import RMSNorm, TorchLinear


class AFMDiscriminator(imfDiT):
    """iMF-DiT feature backbone with D(endpoint, endpoint_time, label) output."""

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

    def __call__(self, endpoint, endpoint_time, target_label):
        batch_size = endpoint.shape[0]
        endpoint_time = jnp.asarray(endpoint_time, dtype=endpoint.dtype).reshape(
            batch_size
        )
        target_label = jnp.asarray(target_label, dtype=jnp.int32).reshape(batch_size)

        # Only endpoint_time and target_label vary. The generator's otherwise
        # reusable guidance/interval token slots receive fixed constants, so D
        # has no access to x_t, noise, r/t pairs, or the source endpoint.
        unit_guidance = jnp.ones((batch_size,), dtype=endpoint.dtype)
        zero = jnp.zeros((batch_size,), dtype=endpoint.dtype)
        one = jnp.ones((batch_size,), dtype=endpoint.dtype)
        sequence = self._build_sequence(
            endpoint,
            endpoint_time,
            unit_guidance,
            zero,
            one,
            target_label,
        )
        sequence = self._run_shared_blocks(sequence)
        for block in self.u_heads:
            sequence = block(sequence, self.rope_freqs)
        pooled = sequence[:, 0]
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_afm_discriminator(
    generator_net,
    *,
    width_multiplier=1.0,
    depth=None,
):
    if not isinstance(generator_net, imfDiT):
        raise ValueError("AFM currently requires a plain imfDiT generator backbone.")
    if width_multiplier <= 0.0:
        raise ValueError("discriminator_width_multiplier must be positive.")

    hidden_size = int(round(generator_net.hidden_size * width_multiplier))
    hidden_size = max(
        generator_net.num_heads,
        round(hidden_size / generator_net.num_heads) * generator_net.num_heads,
    )
    discriminator_depth = generator_net.depth if depth is None else int(depth)
    if discriminator_depth < generator_net.aux_head_depth:
        raise ValueError(
            "discriminator_depth must be at least generator aux_head_depth "
            f"({generator_net.aux_head_depth})."
        )

    return AFMDiscriminator(
        input_size=generator_net.input_size,
        patch_size=generator_net.patch_size,
        in_channels=generator_net.in_channels,
        hidden_size=hidden_size,
        depth=discriminator_depth,
        num_heads=generator_net.num_heads,
        mlp_ratio=generator_net.mlp_ratio,
        num_classes=generator_net.num_classes,
        use_null_class=generator_net.use_null_class,
        use_auxiliary_v_head=False,
        aux_head_depth=generator_net.aux_head_depth,
        num_class_tokens=generator_net.num_class_tokens,
        num_time_tokens=generator_net.num_time_tokens,
        num_cfg_tokens=generator_net.num_cfg_tokens,
        num_interval_tokens=generator_net.num_interval_tokens,
        token_init_constant=generator_net.token_init_constant,
        embedding_init_constant=generator_net.embedding_init_constant,
        weight_init_constant=generator_net.weight_init_constant,
        eval=False,
    )

