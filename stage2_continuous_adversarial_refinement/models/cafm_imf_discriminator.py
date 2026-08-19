"""Generator-initialized potential discriminator for original CAFM on iMF."""

import jax.numpy as jnp

from models.imfDiT import imfDiT
from models.torch_models import RMSNorm, TorchLinear


class CAFMIMFDiscriminator(imfDiT):
    """Potential D(x_t, t, y) used through its JVP.

    The iMF generator has extra conditioning slots.  For the discriminator,
    its interval and guidance slots are fixed constants and the iMF interval
    slot is repurposed as the absolute point time.  Consequently D is a
    function only of x_t, t, and y, as required by original CAFM.
    """

    def setup(self):
        super().setup()
        self.dis_norm = RMSNorm(self.hidden_size)
        self.dis_head = TorchLinear(
            self.hidden_size,
            1,
            bias=False,
            weight_init="scaled_variance",
            init_constant=1.0,
        )

    def __call__(self, x, point_time, y):
        batch_size = x.shape[0]
        point_time = jnp.asarray(point_time, dtype=x.dtype).reshape(batch_size)
        unit_guidance = jnp.ones((batch_size,), dtype=x.dtype)
        interval_start = jnp.zeros((batch_size,), dtype=x.dtype)
        interval_end = jnp.ones((batch_size,), dtype=x.dtype)

        sequence = self._build_sequence(
            x,
            point_time,
            unit_guidance,
            interval_start,
            interval_end,
            y,
        )
        sequence = self._run_shared_blocks(sequence)
        for block in self.u_heads:
            sequence = block(sequence, self.rope_freqs)

        # The class-conditioning prefix plays the role of CAFM's discriminator
        # token while retaining exact generator-compatible backbone shapes.
        pooled = sequence[:, 0]
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_cafm_imf_discriminator(generator_net):
    if not isinstance(generator_net, imfDiT):
        raise ValueError("CAFM-on-iMF currently supports plain imfDiT backbones.")
    return CAFMIMFDiscriminator(
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
