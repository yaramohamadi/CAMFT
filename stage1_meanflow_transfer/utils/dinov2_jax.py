"""Pure-JAX DINOv2-base (ViT-B/14) forward, loading official HF weights.

Why this exists: `transformers.FlaxDinov2Model` was never shipped in any stock
`transformers` release (HF only ported DINOv2 to PyTorch). The FD-DINO metric in
utils/dino_util.py depended on that nonexistent symbol. Rather than reintroduce a
patched-fork dependency or a torch-CUDA wheel (which bundles its own cuDNN and
clashes with JAX's cuDNN in-process), we reimplement the exact forward pass in JAX
and load the official `facebook/dinov2-base` weights. Reference FD-DINO stats
(768-dim, dinov2-base pooler_output) stay valid because the math is identical.

Fixed input size 224x224 -> 16x16 patch grid. The one numerically delicate op
(bicubic interpolation of the 37x37 pretrained position grid down to 16x16) is
precomputed in torch and baked to a constant (dinov2_base_posembed_224.npy), so
this module has zero interpolation-fidelity risk.
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

HIDDEN = 768
LAYERS = 12
HEADS = 12
HEAD_DIM = HIDDEN // HEADS
PATCH = 14
LN_EPS = 1e-6


def _load_params(safetensors_path, posembed_path):
    from safetensors import safe_open

    raw = {}
    with safe_open(safetensors_path, framework="numpy") as f:
        for k in f.keys():
            raw[k] = f.get_tensor(k)

    def g(name):
        return jnp.asarray(raw[name], dtype=jnp.float32)

    p = {}
    # patch embedding conv weight [768,3,14,14], bias [768]
    p["patch_w"] = g("embeddings.patch_embeddings.projection.weight")
    p["patch_b"] = g("embeddings.patch_embeddings.projection.bias")
    p["cls_token"] = g("embeddings.cls_token")  # [1,1,768]
    p["pos_embed"] = jnp.asarray(
        np.load(posembed_path), dtype=jnp.float32
    )  # [1,257,768] baked
    layers = []
    for i in range(LAYERS):
        pre = f"encoder.layer.{i}."
        layers.append(
            {
                "n1_w": g(pre + "norm1.weight"),
                "n1_b": g(pre + "norm1.bias"),
                "q_w": g(pre + "attention.attention.query.weight"),
                "q_b": g(pre + "attention.attention.query.bias"),
                "k_w": g(pre + "attention.attention.key.weight"),
                "k_b": g(pre + "attention.attention.key.bias"),
                "v_w": g(pre + "attention.attention.value.weight"),
                "v_b": g(pre + "attention.attention.value.bias"),
                "o_w": g(pre + "attention.output.dense.weight"),
                "o_b": g(pre + "attention.output.dense.bias"),
                "ls1": g(pre + "layer_scale1.lambda1"),
                "n2_w": g(pre + "norm2.weight"),
                "n2_b": g(pre + "norm2.bias"),
                "fc1_w": g(pre + "mlp.fc1.weight"),
                "fc1_b": g(pre + "mlp.fc1.bias"),
                "fc2_w": g(pre + "mlp.fc2.weight"),
                "fc2_b": g(pre + "mlp.fc2.bias"),
                "ls2": g(pre + "layer_scale2.lambda1"),
            }
        )
    p["layers"] = layers
    p["ln_w"] = g("layernorm.weight")
    p["ln_b"] = g("layernorm.bias")
    return p


def _layernorm(x, w, b):
    mu = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mu) ** 2, axis=-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + LN_EPS) * w + b


def _linear(x, w, b):
    # torch Linear stores weight as [out,in] -> y = x @ w.T + b
    return jnp.matmul(x, w.T, precision=jax.lax.Precision.HIGHEST) + b


# Force fp32-exact matmuls so features match torch to ~1e-5 (default GPU TF32
# gives ~1e-2 error, which perturbs the FD-DINO distance). Applied only inside
# this module's ops, so the training loop's matmul precision is unaffected.
_PREC = jax.lax.Precision.HIGHEST


def _attention(x, lp):
    B, N, _ = x.shape
    q = _linear(x, lp["q_w"], lp["q_b"]).reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    k = _linear(x, lp["k_w"], lp["k_b"]).reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    v = _linear(x, lp["v_w"], lp["v_b"]).reshape(B, N, HEADS, HEAD_DIM).transpose(0, 2, 1, 3)
    scores = jnp.einsum("bhnd,bhmd->bhnm", q, k, precision=_PREC) / jnp.sqrt(HEAD_DIM)
    probs = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.einsum("bhnm,bhmd->bhnd", probs, v, precision=_PREC)
    ctx = ctx.transpose(0, 2, 1, 3).reshape(B, N, HIDDEN)
    return _linear(ctx, lp["o_w"], lp["o_b"])


def _block(x, lp):
    h = _layernorm(x, lp["n1_w"], lp["n1_b"])
    h = _attention(h, lp)
    x = x + h * lp["ls1"]
    h = _layernorm(x, lp["n2_w"], lp["n2_b"])
    h = _linear(h, lp["fc1_w"], lp["fc1_b"])
    h = jax.nn.gelu(h, approximate=False)  # HF hidden_act="gelu" == exact erf gelu
    h = _linear(h, lp["fc2_w"], lp["fc2_b"])
    x = x + h * lp["ls2"]
    return x


def _forward(params, pixel_values):
    # pixel_values: [B,3,224,224] float32 (already ImageNet-normalized, CHW)
    x = jax.lax.conv_general_dilated(
        pixel_values,
        params["patch_w"],
        window_strides=(PATCH, PATCH),
        padding="VALID",
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
        precision=jax.lax.Precision.HIGHEST,
    )  # [B,768,16,16]
    B = x.shape[0]
    x = x + params["patch_b"][None, :, None, None]
    x = x.reshape(B, HIDDEN, -1).transpose(0, 2, 1)  # [B,256,768]
    cls = jnp.broadcast_to(params["cls_token"], (B, 1, HIDDEN))
    x = jnp.concatenate([cls, x], axis=1)  # [B,257,768]
    x = x + params["pos_embed"]
    for lp in params["layers"]:
        x = _block(x, lp)
    x = _layernorm(x, params["ln_w"], params["ln_b"])
    return x[:, 0]  # pooler_output == final-LN'd CLS token, [B,768]


def build_forward(safetensors_path, posembed_path):
    params = _load_params(safetensors_path, posembed_path)
    fn = jax.jit(_forward)
    return params, fn
