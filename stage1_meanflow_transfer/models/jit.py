"""Flax/JAX Just image Transformer architecture.

This mirrors the PyTorch JiT model from https://github.com/LTH14/JiT while using
the BHWC image convention used by this JAX codebase.
"""

import math
from functools import partial

import numpy as np
import jax.numpy as jnp
from flax import linen as nn

from models.torch_models import RMSNorm, TorchEmbedding, TorchLinear


def modulate(x, shift, scale):
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x.reshape(*x.shape[:-2], -1)


def _repeat_interleave_last(x, repeats):
    return jnp.repeat(x, repeats, axis=-1)


def _broadcat(a, b, axis=-1):
    a = jnp.broadcast_to(a, (max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1]), a.shape[2]))
    b = jnp.broadcast_to(b, (a.shape[0], a.shape[1], b.shape[2]))
    return jnp.concatenate([a, b], axis=axis)


def vision_rope_frequencies(dim, pt_seq_len=16, ft_seq_len=None, num_cls_token=0):
    if ft_seq_len is None:
        ft_seq_len = pt_seq_len
    freqs = 1.0 / (
        10000 ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: dim // 2] / dim)
    )
    t = jnp.arange(ft_seq_len, dtype=jnp.float32) / ft_seq_len * pt_seq_len
    freqs = jnp.einsum("...,f->...f", t, freqs)
    freqs = _repeat_interleave_last(freqs, 2)
    freqs = _broadcat(freqs[:, None, :], freqs[None, :, :], axis=-1)
    freqs = freqs.reshape((-1, freqs.shape[-1]))
    cos = jnp.cos(freqs)
    sin = jnp.sin(freqs)
    if num_cls_token > 0:
        cos_pad = jnp.ones((num_cls_token, cos.shape[-1]), dtype=cos.dtype)
        sin_pad = jnp.zeros((num_cls_token, sin.shape[-1]), dtype=sin.dtype)
        cos = jnp.concatenate([cos_pad, cos], axis=0)
        sin = jnp.concatenate([sin_pad, sin], axis=0)
    return cos, sin


def apply_rotary_pos_emb(x, rope):
    cos, sin = rope
    cos = cos[None, :, None, :].astype(x.dtype)
    sin = sin[None, :, None, :].astype(x.dtype)
    return x * cos + rotate_half(x) * sin


def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    return np.concatenate([emb_sin, emb_cos], axis=1)


class BottleneckPatchEmbed(nn.Module):
    """Image to patch embedding used by JiT."""

    img_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    pca_dim: int = 768
    embed_dim: int = 768
    bias: bool = True

    def setup(self):
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.proj1 = nn.Conv(
            self.pca_dim,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            use_bias=False,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
        )
        self.proj2 = nn.Conv(
            self.embed_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            use_bias=self.bias,
            kernel_init=nn.initializers.xavier_uniform(in_axis=(0, 1, 2), out_axis=-1),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        batch, height, width, _ = x.shape
        if height != self.img_size or width != self.img_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({self.img_size}*{self.img_size})."
            )
        x = self.proj2(self.proj1(x))
        return x.reshape(batch, -1, x.shape[-1])


class JiTTimestepEmbedder(nn.Module):
    hidden_size: int
    frequency_embedding_size: int = 256

    def setup(self):
        self.mlp = nn.Sequential(
            [
                TorchLinear(self.frequency_embedding_size, self.hidden_size, bias=True),
                nn.silu,
                TorchLinear(self.hidden_size, self.hidden_size, bias=True),
            ]
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    def __call__(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


class JiTLabelEmbedder(nn.Module):
    num_classes: int
    hidden_size: int

    def setup(self):
        self.embedding_table = TorchEmbedding(
            self.num_classes + 1,
            self.hidden_size,
            weight_init=None,
        )

    def __call__(self, labels):
        return self.embedding_table(labels)


class JiTAttention(nn.Module):
    dim: int
    num_heads: int = 8
    qkv_bias: bool = True
    qk_norm: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    eval: bool = False

    def setup(self):
        self.head_dim = self.dim // self.num_heads
        self.q_norm = RMSNorm(self.head_dim) if self.qk_norm else (lambda x: x)
        self.k_norm = RMSNorm(self.head_dim) if self.qk_norm else (lambda x: x)
        self.qkv = TorchLinear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.proj = TorchLinear(self.dim, self.dim, bias=True)
        self.attn_dropout = nn.Dropout(rate=self.attn_drop)
        self.proj_dropout = nn.Dropout(rate=self.proj_drop)

    def __call__(self, x, rope):
        batch, seq_len, channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q = jnp.squeeze(q, axis=2)
        k = jnp.squeeze(k, axis=2)
        v = jnp.squeeze(v, axis=2)

        q = apply_rotary_pos_emb(self.q_norm(q), rope)
        k = apply_rotary_pos_emb(self.k_norm(k), rope)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum("bqhd,bkhd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32))
        attn = nn.softmax(attn * scale, axis=-1).astype(x.dtype)
        attn = self.attn_dropout(attn, deterministic=self.eval or self.attn_drop == 0.0)
        x = jnp.einsum("bhqk,bkhd->bqhd", attn, v)
        x = x.reshape(batch, seq_len, channels)
        x = self.proj(x)
        return self.proj_dropout(x, deterministic=self.eval or self.proj_drop == 0.0)


class JiTSwiGLUFFN(nn.Module):
    dim: int
    hidden_dim: int
    drop: float = 0.0
    bias: bool = True
    eval: bool = False

    def setup(self):
        hidden_dim = int(self.hidden_dim * 2 / 3)
        self.inner_dim = hidden_dim
        self.w12 = TorchLinear(self.dim, 2 * hidden_dim, bias=self.bias)
        self.w3 = TorchLinear(hidden_dim, self.dim, bias=self.bias)
        self.ffn_dropout = nn.Dropout(rate=self.drop)

    def __call__(self, x):
        x12 = self.w12(x)
        x1, x2 = jnp.split(x12, 2, axis=-1)
        hidden = nn.silu(x1) * x2
        hidden = self.ffn_dropout(hidden, deterministic=self.eval or self.drop == 0.0)
        return self.w3(hidden)


class JiTFinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.norm_final = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
        )
        self.adaLN_modulation = TorchLinear(
            self.hidden_size,
            2 * self.hidden_size,
            bias=True,
            weight_init="zeros",
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(nn.silu(c)), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class JiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    eval: bool = False

    def setup(self):
        self.norm1 = RMSNorm(self.hidden_size, eps=1e-6)
        self.attn = JiTAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            qkv_bias=True,
            qk_norm=True,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
            eval=self.eval,
        )
        self.norm2 = RMSNorm(self.hidden_size, eps=1e-6)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = JiTSwiGLUFFN(
            self.hidden_size,
            mlp_hidden_dim,
            drop=self.proj_drop,
            eval=self.eval,
        )
        self.adaLN_modulation = TorchLinear(
            self.hidden_size,
            6 * self.hidden_size,
            bias=True,
            weight_init="zeros",
        )

    def __call__(self, x, c, feat_rope):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            self.adaLN_modulation(nn.silu(c)),
            6,
            axis=-1,
        )
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            feat_rope,
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class JiT(nn.Module):
    """Just image Transformer.

    Input and output tensors are BHWC images in [-1, 1] pixel space.
    """

    input_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1024
    depth: int = 24
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    num_classes: int = 1000
    bottleneck_dim: int = 128
    in_context_len: int = 32
    in_context_start: int = 8
    eval: bool = False

    def setup(self):
        self.out_channels = self.in_channels
        self.t_embedder = JiTTimestepEmbedder(self.hidden_size)
        self.y_embedder = JiTLabelEmbedder(self.num_classes, self.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            pca_dim=self.bottleneck_dim,
            embed_dim=self.hidden_size,
            bias=True,
        )

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = self.param(
            "pos_embed",
            lambda key, shape: jnp.asarray(
                get_2d_sincos_pos_embed(
                    self.hidden_size,
                    int(num_patches**0.5),
                ).reshape(shape),
                dtype=jnp.float32,
            ),
            (1, num_patches, self.hidden_size),
        )

        if self.in_context_len > 0:
            self.in_context_posemb = self.param(
                "in_context_posemb",
                nn.initializers.normal(stddev=0.02),
                (1, self.in_context_len, self.hidden_size),
            )

        half_head_dim = self.hidden_size // self.num_heads // 2
        hw_seq_len = self.input_size // self.patch_size
        self.feat_rope = vision_rope_frequencies(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )
        self.feat_rope_incontext = vision_rope_frequencies(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        self.blocks = [
            JiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_drop=(
                    self.attn_drop
                    if (self.depth // 4 * 3 > i >= self.depth // 4)
                    else 0.0
                ),
                proj_drop=(
                    self.proj_drop
                    if (self.depth // 4 * 3 > i >= self.depth // 4)
                    else 0.0
                ),
                eval=self.eval,
            )
            for i in range(self.depth)
        ]
        self.final_layer = JiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def unpatchify(self, x):
        channels = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Cannot unpatchify sequence length {x.shape[1]}.")
        x = x.reshape((x.shape[0], h, w, p, p, channels))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], h * p, w * p, channels))

    def __call__(self, x, t, y):
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y)
        c = t_emb + y_emb

        x = self.x_embedder(x)
        x = x + self.pos_embed

        for i, block in enumerate(self.blocks):
            if self.in_context_len > 0 and i == self.in_context_start:
                in_context_tokens = jnp.repeat(
                    y_emb[:, None, :],
                    self.in_context_len,
                    axis=1,
                )
                in_context_tokens = in_context_tokens + self.in_context_posemb
                x = jnp.concatenate([in_context_tokens, x], axis=1)

            rope = self.feat_rope if i < self.in_context_start else self.feat_rope_incontext
            x = block(x, c, rope)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        x = self.final_layer(x, c)
        return self.unpatchify(x)


class imfJiT_DMF(nn.Module):
    """Decoupled single-head pixel-space JiT for MeanFlow transfer (MeFT).

    This mirrors :class:`JiT` but splits the transformer stack into an encoder
    (conditioned on ``t``) and a decoder (conditioned on ``r``), matching the
    Decoupled MeanFlow (DMF) single-head interface used by the DiT/SiT MeFT
    backbones in ``models/imfDiT.py``. The forward signature is intentionally
    identical to ``imfSiT_DMF.__call__``:

        ``(x, t, r, y, omega=None, t_min=None, t_max=None)``

    Input and output tensors are BHWC images in pixel space. The pretrained
    JiT network predicts clean images (data space); the :class:`iMeanFlow`
    wrapper converts this raw data prediction into transport velocity, so this
    backbone stays a drop-in data-space predictor.
    """

    input_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    hidden_size: int = 1280
    encoder_depth: int = 24
    decoder_depth: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    num_classes: int = 1000
    use_null_class: bool = True
    bottleneck_dim: int = 256
    in_context_len: int = 32
    in_context_start: int = 10
    time_conditioning_mode: str = "split"
    # Guidance-conditioning knobs are accepted for interface parity with
    # ``imfSiT_DMF`` (imf.py forwards them to every DMF backbone). Only the
    # disabled/plain configuration is supported for the JiT MeFT path.
    use_context_guidance_conditioning: bool = False
    use_adaln_guidance_scale_conditioning: bool = False
    adaln_guidance_scale_init: str = "timestep"
    use_adaln_condition_mixing: bool = False
    decoder_only_guidance_conditioning: bool = False
    eval: bool = False

    def setup(self):
        if self.time_conditioning_mode not in {"split", "both"}:
            raise ValueError(
                "time_conditioning_mode must be one of ['split', 'both']."
            )
        if (
            self.use_context_guidance_conditioning
            or self.use_adaln_guidance_scale_conditioning
            or self.use_adaln_condition_mixing
            or self.decoder_only_guidance_conditioning
        ):
            raise NotImplementedError(
                "imfJiT_DMF currently supports only the plain (no guidance "
                "conditioning) configuration."
            )

        self.out_channels = self.in_channels
        self.depth = self.encoder_depth + self.decoder_depth

        self.t_embedder = JiTTimestepEmbedder(self.hidden_size)
        # Class embedder: +1 slot for the null class used by CFG dropout.
        num_label_slots = self.num_classes + (1 if self.use_null_class else 0)
        self.y_embedder = JiTLabelEmbedder(num_label_slots - 1, self.hidden_size)
        self.x_embedder = BottleneckPatchEmbed(
            img_size=self.input_size,
            patch_size=self.patch_size,
            in_chans=self.in_channels,
            pca_dim=self.bottleneck_dim,
            embed_dim=self.hidden_size,
            bias=True,
        )

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = self.param(
            "pos_embed",
            lambda key, shape: jnp.asarray(
                get_2d_sincos_pos_embed(
                    self.hidden_size,
                    int(num_patches**0.5),
                ).reshape(shape),
                dtype=jnp.float32,
            ),
            (1, num_patches, self.hidden_size),
        )

        if self.in_context_len > 0:
            self.in_context_posemb = self.param(
                "in_context_posemb",
                nn.initializers.normal(stddev=0.02),
                (1, self.in_context_len, self.hidden_size),
            )

        half_head_dim = self.hidden_size // self.num_heads // 2
        hw_seq_len = self.input_size // self.patch_size
        self.feat_rope = vision_rope_frequencies(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=0,
        )
        self.feat_rope_incontext = vision_rope_frequencies(
            dim=half_head_dim,
            pt_seq_len=hw_seq_len,
            num_cls_token=self.in_context_len,
        )

        def _make_block(i):
            drop_active = self.depth // 4 * 3 > i >= self.depth // 4
            return JiTBlock(
                self.hidden_size,
                self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_drop=self.attn_drop if drop_active else 0.0,
                proj_drop=self.proj_drop if drop_active else 0.0,
                eval=self.eval,
            )

        self.encoder_blocks = [_make_block(i) for i in range(self.encoder_depth)]
        self.decoder_blocks = [
            _make_block(self.encoder_depth + i) for i in range(self.decoder_depth)
        ]
        self.final_layer = JiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def unpatchify(self, x):
        channels = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        if h * w != x.shape[1]:
            raise ValueError(f"Cannot unpatchify sequence length {x.shape[1]}.")
        x = x.reshape((x.shape[0], h, w, p, p, channels))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], h * p, w * p, channels))

    def __call__(self, x, t, r, y, omega=None, t_min=None, t_max=None):
        del omega, t_min, t_max  # unused in the plain JiT MeFT configuration

        if y is None:
            y = jnp.zeros((x.shape[0],), dtype=jnp.int32)
        y = y.astype(jnp.int32)
        y_emb = self.y_embedder(y)

        t_emb = self.t_embedder(t)
        r_emb = self.t_embedder(r)
        if self.time_conditioning_mode == "both":
            shared = t_emb + r_emb
            encoder_c = shared + y_emb
            decoder_c = shared + y_emb
        else:
            encoder_c = t_emb + y_emb
            decoder_c = r_emb + y_emb

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
            c = encoder_c if i < self.encoder_depth else decoder_c
            x = block(x, c, rope)

        if self.in_context_len > 0:
            x = x[:, self.in_context_len :]
        x = self.final_layer(x, decoder_c)
        return self.unpatchify(x)


flaxJiT_H_16 = partial(
    JiT,
    depth=32,
    hidden_size=1280,
    num_heads=16,
    bottleneck_dim=256,
    in_context_len=32,
    in_context_start=10,
    patch_size=16,
)


# Decoupled MeanFlow (DMF) single-head pixel-space JiT backbone. Splits the 32
# pretrained JiT-H/16 blocks into a 24-block encoder + 8-block decoder, mirroring
# the imfSiT_DMF encoder/decoder ratio while preserving JiT-H hyperparameters.
imfJiT_DMF_H_16 = partial(
    imfJiT_DMF,
    input_size=256,
    patch_size=16,
    in_channels=3,
    hidden_size=1280,
    num_heads=16,
    bottleneck_dim=256,
    in_context_len=32,
    in_context_start=10,
    encoder_depth=24,
    decoder_depth=8,
)


JiT_models = {
    "flaxJiT_H_16": flaxJiT_H_16,
    "imfJiT_DMF_H_16": imfJiT_DMF_H_16,
}
