# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted from the official pMF repository for ConvNeXt perceptual loss.

import re
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import torch
from flax import linen as nn

from utils.logging_util import log_for_0


class ConvNextLayerNorm(nn.Module):
    normalized_shape: int
    eps: float = 1e-6

    def setup(self):
        self.weight = self.param("weight", nn.initializers.ones, (self.normalized_shape,))
        self.bias = self.param("bias", nn.initializers.zeros, (self.normalized_shape,))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
        x = (x - mean) / jnp.sqrt(var + self.eps)
        return self.weight * x + self.bias


class ConvNextGRN(nn.Module):
    dim: int
    eps: float = 1e-6

    def setup(self):
        self.gamma = self.param("gamma", nn.initializers.zeros, (1, 1, 1, self.dim))
        self.beta = self.param("beta", nn.initializers.zeros, (1, 1, 1, self.dim))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        norm = jnp.sum(x**2, axis=(1, 2), keepdims=True)
        gx = jnp.sqrt(norm + self.eps)
        nx = gx / (jnp.mean(gx, axis=-1, keepdims=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNextBlock(nn.Module):
    dim: int

    def setup(self):
        self.dwconv = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding="SAME",
            feature_group_count=self.dim,
            name="dwconv",
        )
        self.norm = ConvNextLayerNorm(self.dim, eps=1e-6)
        self.pwconv1 = nn.Dense(features=4 * self.dim, name="pwconv1")
        self.grn = ConvNextGRN(4 * self.dim)
        self.pwconv2 = nn.Dense(features=self.dim, name="pwconv2")

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = jax.nn.gelu(x, approximate=False)
        x = self.grn(x)
        x = self.pwconv2(x)
        return residual + x


class ConvNextV2(nn.Module):
    in_chans: int = 3
    num_classes: int = 1000
    drop_path_rate: float = 0.0
    head_init_scale: float = 1.0
    depths: Sequence[int] = (3, 3, 9, 3)
    dims: Sequence[int] = (96, 192, 384, 768)

    def setup(self):
        stem = nn.Sequential(
            layers=[
                nn.Conv(features=self.dims[0], kernel_size=(4, 4), strides=(4, 4)),
                ConvNextLayerNorm(self.dims[0], eps=1e-6),
            ],
            name="downsample_layers_0",
        )
        layers = [stem]
        for i in range(3):
            layers.append(
                nn.Sequential(
                    layers=[
                        ConvNextLayerNorm(self.dims[i], eps=1e-6),
                        nn.Conv(
                            features=self.dims[i + 1],
                            kernel_size=(2, 2),
                            strides=(2, 2),
                        ),
                    ],
                    name=f"downsample_layers_{i + 1}",
                )
            )
        self.downsample_layers = layers

        self.stages = [
            nn.Sequential(
                layers=[ConvNextBlock(dim=self.dims[i]) for _ in range(self.depths[i])],
                name=f"stages_{i}",
            )
            for i in range(4)
        ]
        self.norm = nn.LayerNorm(epsilon=1e-6)
        self.head = nn.Dense(features=self.num_classes)

    def forward_features(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean(axis=(1, 2)))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.forward_features(x)


ConvNextBase = partial(ConvNextV2, depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024])


def convert_weights_to_jax(jax_params: dict, module_pt, hf: bool = False):
    log_for_0("Converting ConvNeXt weights to JAX...")
    jax_params_flat, jax_param_pytree = jax.tree_util.tree_flatten_with_path(jax_params)
    pt_params = {path: param for path, param in module_pt.items()}

    if hf:
        new_pt_params = {}
        for path, param in pt_params.items():
            path = re.sub(r"classifier\.", "head.", path)
            path = re.sub(r"convnextv2\.encoder\.", "", path)
            path = re.sub(
                r"convnextv2\.embeddings\.patch_embeddings\.",
                "downsample_layers_0.layers_0.",
                path,
            )
            path = re.sub(
                r"convnextv2\.embeddings\.layernorm\.",
                "downsample_layers_0.layers_1.",
                path,
            )
            path = re.sub(
                r"stages\.([0-3])\.downsampling_layer\.(\d+)",
                lambda m: f"downsample_layers_{m.group(1)}.layers_{m.group(2)}",
                path,
            )
            path = re.sub(
                r"stages\.([0-3])\.layers\.(\d+)",
                lambda m: f"stages_{m.group(1)}.layers_{m.group(2)}",
                path,
            )
            path = re.sub(r"layernorm", "norm", path)
            path = re.sub(r"convnextv2\.", "", path)
            path = re.sub(r"grn\.weight", "grn.gamma", path)
            path = re.sub(r"grn\.bias", "grn.beta", path)
            new_pt_params[path] = param
        pt_params = new_pt_params
    else:
        new_pt_params = {}
        for path, param in pt_params.items():
            for i in range(4):
                path = re.sub(
                    rf"stages\.{i}\.(\d+)",
                    lambda m: f"stages_{i}.layers_{m.group(1)}",
                    path,
                )
                path = re.sub(
                    rf"downsample_layers\.{i}\.(\d+)",
                    lambda m: f"downsample_layers_{i}.layers_{m.group(1)}",
                    path,
                )
            new_pt_params[path] = param
        pt_params = new_pt_params

    pt_params = {f"params.{path}": param for path, param in pt_params.items()}
    direct_copy = ("grn",)
    pt_params_flat = []

    for path, param in jax_params_flat:
        shape = param.shape
        path = ".".join([p.key for p in path])
        path = re.sub(r"\.scale|.kernel", ".weight", path)
        if path not in pt_params:
            log_for_0(
                "[WARNING] missing ConvNeXt param %r with shape %s from PyTorch model",
                path,
                shape,
            )
            pt_params_flat.append(None)
            continue

        pt_param = pt_params.pop(path)
        if not any(key in path for key in direct_copy):
            if len(shape) == 4:
                pt_param = torch.permute(pt_param, (2, 3, 1, 0))
            else:
                pt_param = torch.permute(pt_param, tuple(reversed(range(len(shape)))))
        if shape != pt_param.shape:
            log_for_0(
                "[WARNING] ConvNeXt shape mismatch for %r: expected %s, got %s",
                path,
                shape,
                pt_param.shape,
            )
        pt_params_flat.append(jnp.asarray(pt_param.detach().numpy()))

    for path, param in pt_params.items():
        log_for_0(
            "[WARNING] unused ConvNeXt source param %r with shape %s",
            path,
            param.shape,
        )

    log_for_0("ConvNeXt conversion done.")
    return jax.tree_util.tree_unflatten(jax_param_pytree, pt_params_flat)


def load_convnext_jax_model():
    model_jax = ConvNextBase()
    dummy_input = jnp.ones((1, 224, 224, 3))
    jax_params = model_jax.init(jax.random.PRNGKey(0), dummy_input)

    from transformers import ConvNextV2ForImageClassification

    model = ConvNextV2ForImageClassification.from_pretrained(
        "facebook/convnextv2-base-22k-224"
    )
    model_pt = model.state_dict()
    jax_params = convert_weights_to_jax(jax_params, model_pt, hf=True)
    return model_jax, jax_params
