"""Generator-initialized discriminator for continuous adversarial iMF."""

import jax.numpy as jnp

from models.imfDiT import imfDiT
from models.torch_models import RMSNorm, TorchLinear


class CAIMFDiscriminator(imfDiT):
    """An iMF-DiT backbone whose image projection is replaced by a scalar head.

    The module deliberately keeps the generator's module names for the patch,
    condition, shared, and u-branch transformer layers.  Consequently all
    shape-compatible tensors can be copied directly from an iMF generator
    checkpoint.  Only ``dis_norm`` and ``dis_head`` are new.

    ``point_time`` is the time attached to the potential D(x_s, s).  The full
    finite interval [r, t] is also embedded through the generator's two CFG
    interval embedders, giving the discriminator both point-time and interval
    information without introducing another randomly initialized backbone.
    """

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

    def __call__(self, x, point_time, interval_r, interval_t, y):
        batch_size = x.shape[0]
        point_time = jnp.asarray(point_time, dtype=x.dtype).reshape(batch_size)
        interval_r = jnp.asarray(interval_r, dtype=x.dtype).reshape(batch_size)
        interval_t = jnp.asarray(interval_t, dtype=x.dtype).reshape(batch_size)
        unit_guidance = jnp.ones((batch_size,), dtype=x.dtype)

        sequence = self._build_sequence(
            x,
            point_time,
            unit_guidance,
            interval_r,
            interval_t,
            y,
        )
        sequence = self._run_shared_blocks(sequence)
        for block in self.u_heads:
            sequence = block(sequence, self.rope_freqs)

        # The first class-conditioning token summarizes the complete sequence.
        pooled = sequence[:, 0]
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_caimf_discriminator(generator_net):
    """Create a discriminator with the exact architecture of an iMF-DiT net."""
    if not isinstance(generator_net, imfDiT):
        raise ValueError(
            "Continuous adversarial iMF currently supports the plain imfDiT "
            "backbone (for example imfDiT_XL_2)."
        )

    return CAIMFDiscriminator(
        input_size=generator_net.input_size,
        patch_size=generator_net.patch_size,
        in_channels=generator_net.in_channels,
        hidden_size=generator_net.hidden_size,
        depth=generator_net.depth,
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
