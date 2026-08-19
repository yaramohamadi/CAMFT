"""Generator-initialized scalar discriminator for JiT-DMF MeFT checkpoints.

Mirrors :class:`models.jit.imfJiT_DMF` but replaces the pixel output head with a
scalar potential head ``D(x_s, s, y)`` used by the forward-time CA-iMF loss.
Analogous to :class:`models.sit_meft_discriminator.SiTMeFTDiscriminator`, but the
JiT backbone differs from the SiT/DiT DMF backbone (RoPE attention, in-context
label tokens, a 3-argument block call), so it cannot subclass the SiT D.
"""

import jax.numpy as jnp

from models.jit import imfJiT_DMF
from models.torch_models import RMSNorm, TorchLinear


class JiTMeFTDiscriminator(imfJiT_DMF):
    """Reuse the JiT-DMF backbone; emit a scalar D(x_s, s, y) via mean-pool."""

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
        # CA-iMF calls D(x, s, t, r, y); AFM calls D(x, s, y). The scalar
        # potential is evaluated at the single time ``point_time``; the interval
        # endpoints (t, r) are only used by the caller to form quotients.
        if y is None and interval_end is None and interval_start is not None:
            y = interval_start
        del interval_start, interval_end
        batch_size = x.shape[0]
        point_time = jnp.asarray(point_time, dtype=x.dtype).reshape(batch_size)
        if y is None:
            y = jnp.zeros((batch_size,), dtype=jnp.int32)
        y = jnp.asarray(y, dtype=jnp.int32).reshape(batch_size)

        y_emb = self.y_embedder(y)
        # Single evaluation time -> both encoder and decoder conditions use it
        # (matches the SiT-MeFT discriminator, which conditions on point_time
        # for the whole stack rather than a (t, r) split).
        s_emb = self.t_embedder(point_time)
        cond = s_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        blocks = list(self.encoder_blocks) + list(self.decoder_blocks)
        for i, block in enumerate(blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = jnp.repeat(
                    y_emb[:, None, :],
                    self.in_context_len,
                    axis=1,
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = jnp.concatenate([in_context_tokens, x], axis=1)

            rope = (
                self.feat_rope
                if i < self.in_context_start
                else self.feat_rope_incontext
            )
            x = block(x, cond, rope)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        pooled = jnp.mean(x, axis=1)
        return self.dis_head(self.dis_norm(pooled)).reshape(batch_size)


def create_jit_meft_discriminator(generator_net):
    """Build a scalar D with names/shapes compatible with a JiT-DMF generator."""
    if not isinstance(generator_net, imfJiT_DMF):
        raise ValueError(
            "JiT-MeFT adversarial training requires an imfJiT_DMF generator."
        )
    return JiTMeFTDiscriminator(
        input_size=generator_net.input_size,
        patch_size=generator_net.patch_size,
        in_channels=generator_net.in_channels,
        hidden_size=generator_net.hidden_size,
        num_heads=generator_net.num_heads,
        mlp_ratio=generator_net.mlp_ratio,
        encoder_depth=generator_net.encoder_depth,
        decoder_depth=generator_net.decoder_depth,
        num_classes=generator_net.num_classes,
        use_null_class=generator_net.use_null_class,
        bottleneck_dim=generator_net.bottleneck_dim,
        in_context_len=generator_net.in_context_len,
        in_context_start=generator_net.in_context_start,
        time_conditioning_mode=generator_net.time_conditioning_mode,
        eval=False,
    )
