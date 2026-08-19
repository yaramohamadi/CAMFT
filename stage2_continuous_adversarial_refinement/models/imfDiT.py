import math
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn

from models.embedder import PatchEmbedder, TimestepEmbedder, LabelEmbedder
from models.torch_models import TorchLinear, RMSNorm, SwiGLUMlp


def unsqueeze(t, dim):
    """Adds a new axis to a tensor at the given position."""
    return jnp.expand_dims(t, axis=dim)


#################################################################################
#                   Modern Transformer Components with Vec Gates               #
#################################################################################


class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE and QK RMS norm."""

    hidden_size: int
    num_heads: int

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=False,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)

        self.head_dim = self.hidden_size // self.num_heads

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def __call__(self, x, rope_freqs):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rotary_pos_emb(q, rope_freqs)
        k = apply_rotary_pos_emb(k, rope_freqs)

        attn = nn.dot_product_attention(q, k, v, dtype=jnp.float32)
        attn = attn.reshape(batch, seq_len, self.hidden_size)

        return self.out_proj(attn)


class TransformerBlock(nn.Module):
    """Transformer block with zero-initialized vector gates on residuals."""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0

    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        self.norm1 = RMSNorm(self.hidden_size)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = RMSNorm(self.hidden_size)
        mlp_hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.mlp = SwiGLUMlp(
            self.hidden_size,
            mlp_hidden_dim,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.attn_scale = self.param(
            "attn_scale", nn.initializers.zeros, (self.hidden_size,)
        )
        self.mlp_scale = self.param(
            "mlp_scale", nn.initializers.zeros, (self.hidden_size,)
        )

    def __call__(self, x, rope_freqs):
        x = x + self.attn(self.norm1(x), rope_freqs) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """Final projection layer with RMSNorm and zero init weights."""

    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.norm = RMSNorm(self.hidden_size)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            bias_init="zeros",
        )

    def __call__(self, x):
        return self.linear(self.norm(x))


#################################################################################
#                improved MeanFlow DiT with In-context Conditioning             #
#################################################################################


class imfDiT(nn.Module):
    """
    A shared backbone processes the first (depth - aux_head_depth) layers.
    Two heads of equal depth (aux_head_depth) branch off afterwards.
    """

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 8 / 3
    num_classes: int = 1000
    use_null_class: bool = True
    use_auxiliary_v_head: bool = True

    aux_head_depth: int = 8

    num_class_tokens: int = 8
    num_time_tokens: int = 4
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2

    token_init_constant: float = 1.0
    embedding_init_constant: float = 1.0
    weight_init_constant: float = 0.32

    eval: bool = False

    def setup(self):
        """
        Set up the imfDiT model components.
         - Patch embedder for input images.
         - Embedders for time, omega, cfg intervals, and class labels.
         - Learnable tokens for conditioning.
         - Transformer blocks with shared backbone and dual heads.
         - Final projection layers for u and v outputs.
        """

        self.out_channels = self.in_channels

        self.x_embedder = PatchEmbedder(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )

        embed_kwargs = dict(
            hidden_size=self.hidden_size,
            weight_init="scaled_variance",
            init_constant=self.embedding_init_constant,
        )

        self.h_embedder = TimestepEmbedder(**embed_kwargs)
        self.omega_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_start_embedder = TimestepEmbedder(**embed_kwargs)
        self.cfg_t_end_embedder = TimestepEmbedder(**embed_kwargs)
        self.y_embedder = LabelEmbedder(
            self.num_classes,
            use_null_class=self.use_null_class,
            **embed_kwargs,
        )

        token_initializer = nn.initializers.normal(
            stddev=self.token_init_constant / math.sqrt(self.hidden_size)
        )

        self.time_tokens = self.param(
            "time_tokens",
            token_initializer,
            (self.num_time_tokens, self.hidden_size),
        )
        self.class_tokens = self.param(
            "class_tokens",
            token_initializer,
            (self.num_class_tokens, self.hidden_size),
        )
        self.omega_tokens = self.param(
            "omega_tokens",
            token_initializer,
            (self.num_cfg_tokens, self.hidden_size),
        )
        self.t_min_tokens = self.param(
            "t_min_tokens",
            token_initializer,
            (self.num_interval_tokens, self.hidden_size),
        )
        self.t_max_tokens = self.param(
            "t_max_tokens",
            token_initializer,
            (self.num_interval_tokens, self.hidden_size),
        )

        total_tokens = (
            self.x_embedder.num_patches
            + self.num_class_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.prefix_tokens = (
            self.num_class_tokens
            + self.num_cfg_tokens
            + 2 * self.num_interval_tokens
            + self.num_time_tokens
        )
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_freqs = precompute_rope_freqs(self.head_dim, total_tokens)

        head_depth = self.aux_head_depth
        shared_depth = self.depth - head_depth

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init="scaled_variance",
            weight_init_constant=self.weight_init_constant,
        )

        self.shared_blocks = [
            TransformerBlock(**block_kwargs) for _ in range(shared_depth)
        ]
        self.u_heads = [TransformerBlock(**block_kwargs) for _ in range(head_depth)]

        self.v_heads = []
        if self.use_auxiliary_v_head:
            # We don't need the deeper v refinement blocks during evaluation.
            self.v_heads = [
                TransformerBlock(**block_kwargs)
                for _ in range(head_depth if not self.eval else 0)
            ]

        self.u_final_layer = FinalLayer(
            self.hidden_size, self.patch_size, self.out_channels
        )
        self.v_final_layer = None
        if self.use_auxiliary_v_head:
            self.v_final_layer = FinalLayer(
                self.hidden_size, self.patch_size, self.out_channels
            )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        images = x.reshape((x.shape[0], h * p, w * p, c))
        return images

    def _build_sequence(self, x, h, w, t_min, t_max, y):
        """
        Build the input token sequence for the transformer.
        1. Embed the input image patches.
        2. Embed the conditioning information (time, omega, cfg, class labels).
        3. Prepend the conditioning tokens to the patch embeddings.

        Args:
            x: Input images
            h: timestep
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels

        Returns:
            seq: Token sequence for the transformer
        """

        x_embed = self.x_embedder(x)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(1 - 1 / w)
        t_min_embed = self.cfg_t_start_embedder(t_min)
        t_max_embed = self.cfg_t_end_embedder(t_max)
        y_embed = self.y_embedder(y)

        time_tokens = self.time_tokens + unsqueeze(h_embed, 1)
        omega_tokens = self.omega_tokens + unsqueeze(omega_embed, 1)
        t_min_tokens = self.t_min_tokens + unsqueeze(t_min_embed, 1)
        t_max_tokens = self.t_max_tokens + unsqueeze(t_max_embed, 1)
        class_tokens = self.class_tokens + unsqueeze(y_embed, 1)

        seq = jnp.concatenate(
            [
                class_tokens,
                omega_tokens,
                t_min_tokens,
                t_max_tokens,
                time_tokens,
                x_embed,
            ],
            axis=1,
        )

        return seq

    def _run_shared_blocks(self, seq):
        for block in self.shared_blocks:
            seq = block(seq, self.rope_freqs)
        return seq

    def _decode_u(self, seq):
        u_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs)
        u_tokens = u_seq[:, self.prefix_tokens :]
        return self.unpatchify(self.u_final_layer(u_tokens))

    def predict_u_only(self, x, t, h, w, t_min, t_max, y):
        """Run only the shared trunk and u branch, skipping the auxiliary v head."""
        del t  # iMF-DiT conditions on interval h=t-r rather than absolute t.
        seq = self._build_sequence(x, h, w, t_min, t_max, y)
        seq = self._run_shared_blocks(seq)
        return self._decode_u(seq)

    def _decode_v(self, seq):
        if not self.use_auxiliary_v_head:
            raise ValueError("_decode_v requires use_auxiliary_v_head=True.")

        v_seq = seq
        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs)
        v_tokens = v_seq[:, self.prefix_tokens :]
        return self.unpatchify(self.v_final_layer(v_tokens))

    def predict_v_only(self, x, t, h, w, t_min, t_max, y):
        """Run only the shared trunk plus the v branch."""
        del t
        seq = self._build_sequence(x, h, w, t_min, t_max, y)
        seq = self._run_shared_blocks(seq)
        return self._decode_v(seq)

    def __call__(self, x, t, h, w, t_min, t_max, y):
        """
        Forward pass of the imfDiT model.
        Returns the predicted u and v components.

        Args:
            x: Input images
            t, h: time steps
            w: CFG scale
            t_min, t_max: CFG interval
            y: Class labels
        
        Returns:
            u: Average velocity field
            v: Instantaneous velocity field. In single-head mode this aliases
               `u`, so callers can recover boundary-time v by making a second
               forward pass with `h = 0`.
        """

        # We don't explicitly condition on time t, only on h = t - r
        # following https://arxiv.org/abs/2502.13129
        seq = self._build_sequence(x, h, w, t_min, t_max, y)
        seq = self._run_shared_blocks(seq)
        u = self._decode_u(seq)

        if not self.use_auxiliary_v_head:
            return u, u

        v = self._decode_v(seq)

        return u, v


#################################################################################
#                           Rotary Position Helpers                             #
#################################################################################


def precompute_rope_freqs(dim: int, seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs_cis = jnp.outer(positions, freqs)
    real = jnp.cos(freqs_cis)
    imag = jnp.sin(freqs_cis)
    return jax.lax.complex(real, imag)


def apply_rotary_pos_emb(x, freqs_cis):
    x_complex = x.astype(jnp.float32).view(jnp.complex64)
    x_complex = x_complex.reshape(x.shape[:-1] + (-1,))
    freqs_cis = unsqueeze(unsqueeze(freqs_cis, 0), 2)
    x_rotated = x_complex * freqs_cis
    x_out = x_rotated.astype(x_complex.dtype).view(x.dtype)
    return x_out.reshape(x.shape)


def modulate(x, shift, scale):
    """Apply adaptive layer-norm modulation."""
    shift = shift.astype(x.dtype)
    scale = scale.astype(x.dtype)
    one = jnp.asarray(1.0, dtype=x.dtype)
    return x * (one + scale[:, None, :]) + shift[:, None, :]


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """Generate 1D sine-cosine positional embeddings from grid positions."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Generate 2D sine-cosine positional embeddings from a grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    Generate 2D sine-cosine positional embeddings.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim], dtype=np.float32), pos_embed], axis=0)
    return pos_embed


class SiTTimeEmbedder(nn.Module):
    """Sinusoidal timestep embedder matching the torch SiT implementation."""

    hidden_size: int
    frequency_embedding_size: int = 256
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0

    def setup(self):
        self.fc1 = TorchLinear(
            self.frequency_embedding_size,
            self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )
        self.fc2 = TorchLinear(
            self.hidden_size,
            self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = jnp.exp(
            -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
        )
        args = t[:, None].astype(jnp.float32) * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate([embedding, jnp.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t):
        t = jnp.asarray(t, dtype=jnp.float32)
        t_freq = self.positional_embedding(t, self.frequency_embedding_size)
        x = self.fc1(t_freq)
        x = nn.silu(x)
        return self.fc2(x)


class ResidualConcatConditionProjector(nn.Module):
    """Project concatenated condition pairs back to hidden size with [I, 0] init."""

    hidden_size: int

    def setup(self):
        self.delta_kernel = self.param(
            "delta_kernel",
            nn.initializers.zeros,
            (2 * self.hidden_size, self.hidden_size),
        )
        self.bias = self.param("bias", nn.initializers.zeros, (self.hidden_size,))

    def _base_kernel(self):
        eye = jnp.eye(self.hidden_size, dtype=jnp.float32)
        zeros = jnp.zeros_like(eye)
        return jnp.concatenate([eye, zeros], axis=0)

    def __call__(self, first, second):
        x = jnp.concatenate([first, second], axis=-1)
        kernel = self._base_kernel().astype(x.dtype) + self.delta_kernel.astype(x.dtype)
        return jnp.einsum("...d,df->...f", x, kernel) + self.bias.astype(x.dtype)


class SiTMlp(nn.Module):
    hidden_size: int
    mlp_ratio: float = 4.0
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0

    def setup(self):
        hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.fc1 = TorchLinear(
            self.hidden_size,
            hidden_dim,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )
        self.fc2 = TorchLinear(
            hidden_dim,
            self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.silu(x)
        return self.fc2(x)


class SiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        self.attn = RoPEAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        self.mlp = SiTMlp(
            self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.adaLN_modulation = TorchLinear(
            self.hidden_size,
            6 * self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

    def __call__(self, x, rope_freqs, c):
        modulation = self.adaLN_modulation(nn.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), rope_freqs
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class SiTFinalLayer(nn.Module):
    hidden_size: int
    patch_size: int
    out_channels: int

    def setup(self):
        self.norm_final = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        self.linear = TorchLinear(
            self.hidden_size,
            self.patch_size * self.patch_size * self.out_channels,
            bias=True,
            weight_init="zeros",
            init_constant=1.0,
        )
        self.adaLN_modulation = TorchLinear(
            self.hidden_size,
            2 * self.hidden_size,
            bias=True,
            weight_init="zeros",
            init_constant=1.0,
        )

    def __call__(self, x, c):
        shift, scale = jnp.split(self.adaLN_modulation(nn.silu(c)), 2, axis=-1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class ExactSiTAttention(nn.Module):
    """Attention module matching the original torch SiT block more closely."""

    hidden_size: int
    num_heads: int
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        init_kwargs = dict(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.q_proj = TorchLinear(**init_kwargs)
        self.k_proj = TorchLinear(**init_kwargs)
        self.v_proj = TorchLinear(**init_kwargs)
        self.out_proj = TorchLinear(**init_kwargs)
        self.head_dim = self.hidden_size // self.num_heads

    def __call__(self, x):
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim)

        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = nn.softmax(attn.astype(jnp.float32), axis=-1).astype(x.dtype)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch, seq_len, self.hidden_size)
        return self.out_proj(out)


class ExactSiTMlp(nn.Module):
    hidden_size: int
    mlp_ratio: float = 4.0
    weight_init: str = "scaled_variance"
    init_constant: float = 1.0

    def setup(self):
        hidden_dim = int(self.hidden_size * self.mlp_ratio)
        self.fc1 = TorchLinear(
            self.hidden_size,
            hidden_dim,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )
        self.fc2 = TorchLinear(
            hidden_dim,
            self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.init_constant,
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = nn.gelu(x, approximate=True)
        return self.fc2(x)


class ExactSiTBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        self.norm1 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        self.attn = ExactSiTAttention(
            self.hidden_size,
            num_heads=self.num_heads,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.norm2 = nn.LayerNorm(epsilon=1e-6, use_scale=False, use_bias=False)
        self.mlp = ExactSiTMlp(
            self.hidden_size,
            mlp_ratio=self.mlp_ratio,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.adaLN_modulation = TorchLinear(
            self.hidden_size,
            6 * self.hidden_size,
            bias=True,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

    def __call__(self, x, c):
        modulation = self.adaLN_modulation(nn.silu(c))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            modulation, 6, axis=-1
        )
        x = x + gate_msa[:, None, :] * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp[:, None, :] * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FlaxSiT(nn.Module):
    """Exact-flavor Flax SiT baseline for checkpoint parity debugging."""

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    use_null_class: bool = True
    learn_sigma: bool = True
    return_learned_sigma: bool = False
    use_r_conditioning: bool = False
    eval: bool = False
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        self.out_channels = self.in_channels * 2 if self.learn_sigma else self.in_channels

        self.x_embedder = PatchEmbedder(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )
        self.t_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        if self.use_r_conditioning:
            self.r_embedder = SiTTimeEmbedder(
                self.hidden_size,
                weight_init=self.weight_init,
                init_constant=self.weight_init_constant,
            )
        self.y_embedder = LabelEmbedder(
            self.num_classes,
            self.hidden_size,
            use_null_class=self.use_null_class,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = self.param(
            "pos_embed",
            lambda key, shape: jnp.array(
                get_2d_sincos_pos_embed(
                    self.hidden_size,
                    int(self.input_size / self.patch_size),
                ).reshape((num_patches, self.hidden_size))
            ),
            (num_patches, self.hidden_size),
        )

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.blocks = [ExactSiTBlock(**block_kwargs) for _ in range(self.depth)]
        self.final_layer = SiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], h * p, w * p, c))

    def __call__(self, x, t, y, r=None):
        x = self.x_embedder(x) + self.pos_embed

        if y is None:
            y = jnp.zeros((x.shape[0],), dtype=jnp.int32)

        c = self.t_embedder(t).astype(x.dtype)
        if self.use_r_conditioning:
            if r is None:
                r = t
            c = c + self.r_embedder(r).astype(x.dtype)
        c = c + self.y_embedder(y).astype(x.dtype)
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        if self.learn_sigma and not self.return_learned_sigma:
            x, _ = jnp.split(x, 2, axis=-1)
        return x


class imfSiT_MF(nn.Module):
    """SiT-style improved MeanFlow with shared trunk and dual u/v heads."""

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    aux_head_depth: int = 8
    num_classes: int = 1000
    use_null_class: bool = True
    eval: bool = False
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        self.out_channels = self.in_channels

        self.x_embedder = PatchEmbedder(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )
        self.t_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.h_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.omega_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.t_min_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.t_max_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        self.y_embedder = LabelEmbedder(
            self.num_classes,
            self.hidden_size,
            use_null_class=self.use_null_class,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = self.param(
            "pos_embed",
            lambda key, shape: jnp.array(
                get_2d_sincos_pos_embed(self.hidden_size, int(self.input_size / self.patch_size)).reshape(
                    (num_patches, self.hidden_size)
                )
            ),
            (num_patches, self.hidden_size),
        )

        self.rope_freqs = precompute_rope_freqs(
            self.hidden_size // self.num_heads,
            num_patches,
        )

        shared_depth = self.depth - self.aux_head_depth
        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )

        self.shared_blocks = [SiTBlock(**block_kwargs) for _ in range(shared_depth)]
        self.u_heads = [SiTBlock(**block_kwargs) for _ in range(self.aux_head_depth)]
        self.v_heads = [SiTBlock(**block_kwargs) for _ in range(self.aux_head_depth)]

        self.u_final_layer = SiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )
        self.v_final_layer = SiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], h * p, w * p, c))

    def __call__(self, x, t, h, omega, t_min, t_max, y):
        x = self.x_embedder(x) + self.pos_embed

        t_embed = self.t_embedder(t)
        h_embed = self.h_embedder(h)
        omega_embed = self.omega_embedder(omega)
        t_min_embed = self.t_min_embedder(t_min)
        t_max_embed = self.t_max_embedder(t_max)

        if y is None:
            y = jnp.zeros((x.shape[0],), dtype=jnp.int32)

        y_embed = self.y_embedder(y)
        c = (
            t_embed
            + h_embed
            + omega_embed
            + t_min_embed
            + t_max_embed
            + y_embed
        )

        for block in self.shared_blocks:
            x = block(x, self.rope_freqs, c)

        u_seq = x
        v_seq = x
        for block in self.u_heads:
            u_seq = block(u_seq, self.rope_freqs, c)
        for block in self.v_heads:
            v_seq = block(v_seq, self.rope_freqs, c)

        u = self.u_final_layer(u_seq, c)
        v = self.v_final_layer(v_seq, c)

        return self.unpatchify(u), self.unpatchify(v)


class imfSiT_DMF(nn.Module):
    """Decoupled single-head SiT variant with encoder/decoder block split."""

    input_size: int = 32
    patch_size: int = 2
    in_channels: int = 4
    hidden_size: int = 1152
    encoder_depth: int = 20
    decoder_depth: int = 8
    num_heads: int = 16
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    use_null_class: bool = True
    use_context_guidance_conditioning: bool = False
    use_adaln_guidance_scale_conditioning: bool = False
    adaln_guidance_scale_init: str = "timestep"
    use_adaln_condition_mixing: bool = False
    decoder_only_guidance_conditioning: bool = False
    time_conditioning_mode: str = "split"
    num_cfg_tokens: int = 4
    num_interval_tokens: int = 2
    eval: bool = False
    weight_init: str = "scaled_variance"
    weight_init_constant: float = 1.0

    def setup(self):
        if (
            self.use_context_guidance_conditioning
            and self.use_adaln_guidance_scale_conditioning
        ):
            raise ValueError(
                "use_context_guidance_conditioning and "
                "use_adaln_guidance_scale_conditioning are mutually exclusive."
            )
        if self.use_adaln_guidance_scale_conditioning and (
            self.adaln_guidance_scale_init not in {"timestep", "random", "zero"}
        ):
            raise ValueError(
                "adaln_guidance_scale_init must be one of "
                "['timestep', 'random', 'zero']."
            )
        if self.use_adaln_condition_mixing and (
            not self.use_adaln_guidance_scale_conditioning
        ):
            raise ValueError(
                "use_adaln_condition_mixing requires "
                "use_adaln_guidance_scale_conditioning=True."
            )
        if self.use_adaln_condition_mixing and self.decoder_only_guidance_conditioning:
            raise ValueError(
                "use_adaln_condition_mixing and "
                "decoder_only_guidance_conditioning cannot both be enabled."
            )
        if self.time_conditioning_mode not in {"split", "both"}:
            raise ValueError(
                "time_conditioning_mode must be one of ['split', 'both']."
            )

        self.out_channels = self.in_channels

        self.x_embedder = PatchEmbedder(
            self.input_size,
            self.patch_size,
            self.in_channels,
            self.hidden_size,
            bias=True,
        )
        self.t_embedder = SiTTimeEmbedder(
            self.hidden_size,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )
        if (
            self.use_context_guidance_conditioning
            or self.use_adaln_guidance_scale_conditioning
        ):
            omega_weight_init = (
                "zeros"
                if self.use_adaln_guidance_scale_conditioning
                and self.adaln_guidance_scale_init == "zero"
                else self.weight_init
            )
            self.omega_embedder = SiTTimeEmbedder(
                self.hidden_size,
                weight_init=omega_weight_init,
                init_constant=self.weight_init_constant,
            )
        if self.use_context_guidance_conditioning:
            self.t_min_embedder = SiTTimeEmbedder(
                self.hidden_size,
                weight_init=self.weight_init,
                init_constant=self.weight_init_constant,
            )
            self.t_max_embedder = SiTTimeEmbedder(
                self.hidden_size,
                weight_init=self.weight_init,
                init_constant=self.weight_init_constant,
            )
        if self.use_adaln_condition_mixing or self.time_conditioning_mode == "both":
            self.time_condition_projector = ResidualConcatConditionProjector(
                self.hidden_size
            )
        if self.use_adaln_condition_mixing:
            self.class_condition_projector = ResidualConcatConditionProjector(
                self.hidden_size
            )
        self.y_embedder = LabelEmbedder(
            self.num_classes,
            self.hidden_size,
            use_null_class=self.use_null_class,
            weight_init=self.weight_init,
            init_constant=self.weight_init_constant,
        )

        num_patches = (self.input_size // self.patch_size) ** 2
        self.pos_embed = self.param(
            "pos_embed",
            lambda key, shape: jnp.array(
                get_2d_sincos_pos_embed(
                    self.hidden_size,
                    int(self.input_size / self.patch_size),
                ).reshape((num_patches, self.hidden_size))
            ),
            (num_patches, self.hidden_size),
        )
        if self.use_context_guidance_conditioning:
            token_initializer = nn.initializers.normal(
                stddev=1.0 / math.sqrt(self.hidden_size)
            )
            self.omega_tokens = self.param(
                "omega_tokens",
                token_initializer,
                (self.num_cfg_tokens, self.hidden_size),
            )
            self.t_min_tokens = self.param(
                "t_min_tokens",
                token_initializer,
                (self.num_interval_tokens, self.hidden_size),
            )
            self.t_max_tokens = self.param(
                "t_max_tokens",
                token_initializer,
                (self.num_interval_tokens, self.hidden_size),
            )
            self.prefix_tokens = self.num_cfg_tokens + 2 * self.num_interval_tokens
        else:
            self.prefix_tokens = 0

        block_kwargs = dict(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            weight_init=self.weight_init,
            weight_init_constant=self.weight_init_constant,
        )
        self.encoder_blocks = [
            ExactSiTBlock(**block_kwargs) for _ in range(self.encoder_depth)
        ]
        self.decoder_blocks = [
            ExactSiTBlock(**block_kwargs) for _ in range(self.decoder_depth)
        ]
        self.final_layer = SiTFinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
        )

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = jnp.einsum("nhwpqc->nhpwqc", x)
        return x.reshape((x.shape[0], h * p, w * p, c))

    def _build_guidance_context_tokens(self, omega, t_min, t_max):
        omega = jnp.asarray(omega, dtype=jnp.float32)
        t_min = jnp.asarray(t_min, dtype=jnp.float32)
        t_max = jnp.asarray(t_max, dtype=jnp.float32)

        omega_feature = 1.0 - 1.0 / jnp.maximum(omega, 1e-6)
        omega_embed = self.omega_embedder(omega_feature)
        t_min_embed = self.t_min_embedder(t_min)
        t_max_embed = self.t_max_embedder(t_max)

        omega_tokens = self.omega_tokens + unsqueeze(omega_embed, 1)
        t_min_tokens = self.t_min_tokens + unsqueeze(t_min_embed, 1)
        t_max_tokens = self.t_max_tokens + unsqueeze(t_max_embed, 1)

        return jnp.concatenate(
            [omega_tokens, t_min_tokens, t_max_tokens],
            axis=1,
        )

    def __call__(self, x, t, r, y, omega=None, t_min=None, t_max=None):
        x = self.x_embedder(x) + self.pos_embed

        if y is None:
            y = jnp.zeros((x.shape[0],), dtype=jnp.int32)

        guidance_tokens = None
        if self.use_context_guidance_conditioning:
            if omega is None:
                omega = jnp.ones_like(t)
            if t_min is None:
                t_min = jnp.zeros_like(t)
            if t_max is None:
                t_max = jnp.ones_like(t)
            guidance_tokens = self._build_guidance_context_tokens(omega, t_min, t_max)
            if not self.decoder_only_guidance_conditioning:
                x = jnp.concatenate([guidance_tokens, x], axis=1)
        elif self.use_adaln_guidance_scale_conditioning and omega is None:
            omega = jnp.ones_like(t)

        y_embed = self.y_embedder(y)
        if self.use_adaln_condition_mixing:
            t_embed = self.t_embedder(t)
            r_embed = self.t_embedder(r)
            omega_feature = 1.0 - 1.0 / jnp.maximum(
                jnp.asarray(omega, dtype=jnp.float32), 1e-6
            )
            omega_embed = self.omega_embedder(omega_feature)
            time_c = self.time_condition_projector(t_embed, r_embed)
            class_c = self.class_condition_projector(y_embed, omega_embed)
            shared_c = time_c + class_c
            encoder_c = shared_c
            decoder_c = shared_c
        else:
            t_embed = self.t_embedder(t)
            r_embed = self.t_embedder(r)
            if self.time_conditioning_mode == "both":
                shared_time = self.time_condition_projector(t_embed, r_embed)
                encoder_c = shared_time + y_embed
                decoder_c = shared_time + y_embed
            else:
                encoder_c = t_embed + y_embed
                decoder_c = r_embed + y_embed
        if self.use_adaln_guidance_scale_conditioning and not self.use_adaln_condition_mixing:
            omega_feature = 1.0 - 1.0 / jnp.maximum(
                jnp.asarray(omega, dtype=jnp.float32), 1e-6
            )
            omega_embed = self.omega_embedder(omega_feature)
            omega_delta = omega_embed * jax.lax.stop_gradient(y_embed)
            decoder_c = decoder_c + omega_delta
            if not self.decoder_only_guidance_conditioning:
                encoder_c = encoder_c + omega_delta

        for block in self.encoder_blocks:
            x = block(x, encoder_c)

        if guidance_tokens is not None and self.decoder_only_guidance_conditioning:
            x = jnp.concatenate([guidance_tokens, x], axis=1)

        for block in self.decoder_blocks:
            x = block(x, decoder_c)

        if self.prefix_tokens:
            x = x[:, self.prefix_tokens :]

        u = self.final_layer(x, decoder_c)
        return self.unpatchify(u)


#################################################################################
#                                iMF DiT Configs                                #
#################################################################################


imfDiT_B_2 = partial(
    imfDiT,
    depth=12,
    hidden_size=768,
    patch_size=2,
    num_heads=12,
    aux_head_depth=8,
)

imfDiT_L_2 = partial(
    imfDiT,
    depth=32,
    hidden_size=1024,
    patch_size=2,
    num_heads=16,
    aux_head_depth=8,
)

imfDiT_XL_2 = partial(
    imfDiT,
    depth=48,
    hidden_size=1024,
    patch_size=2,
    num_heads=16,
    aux_head_depth=8,
)

imfSiT_XL_2 = partial(
    imfSiT_MF,
    depth=28,
    hidden_size=1152,
    patch_size=2,
    num_heads=16,
    aux_head_depth=8,
)

imfSiT_DMF_XL_2 = partial(
    imfSiT_DMF,
    hidden_size=1152,
    patch_size=2,
    num_heads=16,
    encoder_depth=20,
    decoder_depth=8,
)

imfSiT_DMF_XL_2_ZS = partial(
    imfSiT_DMF,
    hidden_size=1152,
    patch_size=2,
    num_heads=16,
    encoder_depth=22,
    decoder_depth=6,
)

# DiT_DMF reuses the exact single-head encoder/decoder layout from SiT_DMF in
# this repo; the distinction is the initialization/source family and how the
# raw DiT predictions may be wrapped into velocity.
imfDiT_DMF_XL_2 = imfSiT_DMF_XL_2
imfDiT_DMF_XL_2_ZS = imfSiT_DMF_XL_2_ZS

flaxSiT_XL_2 = partial(
    FlaxSiT,
    depth=28,
    hidden_size=1152,
    patch_size=2,
    num_heads=16,
    learn_sigma=True,
)

# DiT-XL/2 and SiT-XL/2 share the same transformer state-dict key layout and
# tensor shapes in the local PyTorch ports. This alias lets the plain transport
# path initialize from DiT checkpoints while keeping the source family explicit
# in config and logs.
flaxDiT_XL_2 = flaxSiT_XL_2
