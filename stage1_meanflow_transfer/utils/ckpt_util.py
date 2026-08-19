import os
import shutil

import numpy as np
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from flax import serialization

from utils.logging_util import log_for_0
from utils.trainstate_util import EvalState

# The PyTorch reference model files in models/torch_SiT.py and
# models/torch_DiT.py are not directly imported by the Flax training or
# evaluation code here. This loader only consumes a serialized PyTorch
# checkpoint (.pt/.pth) and converts its SiT state_dict tensors into the
# Flax parameter tree expected by the target SiT-based model family.


def restore_checkpoint(state, workdir):
    """
    Restores the model state from a checkpoint located in the specified working directory.
    """
    workdir = os.path.abspath(workdir)
    state = checkpoints.restore_checkpoint(workdir, state)
    log_for_0("Restored from checkpoint at {}".format(workdir))
    return state


def restore_eval_checkpoint(workdir, use_ema=False):
    """
    Restore a lightweight evaluation state without optimizer or grad buffers.

    This avoids allocating a full TrainState during eval-only runs, which can
    otherwise exhaust GPU memory before sampling begins.
    """
    workdir = os.path.abspath(workdir)

    if os.path.isfile(workdir) and workdir.endswith((".pt", ".pth", ".pth.tar")):
        params = load_checkpoint_params(workdir, prefer_ema=use_ema)
        ema_params = params if use_ema else None
        state = EvalState(
            step=jnp.array(0, dtype=jnp.int32),
            params=params,
            ema_params=ema_params,
        )
        log_for_0("Restored lightweight eval state from PyTorch checkpoint at %s", workdir)
        return state

    restored = checkpoints.restore_checkpoint(workdir, target=None)
    if restored is None:
        raise ValueError(
            f"Could not restore a Flax checkpoint from {workdir}. "
            "Pass a real checkpoint directory/checkpoint_* directory, or a .pt/.pth file."
        )
    step = restored.get("step", 0)
    params = restored.get("params")
    ema_params = restored.get("ema_params")

    if params is None and ema_params is None:
        raise ValueError(f"No params/ema_params found in checkpoint: {workdir}")

    if params is None:
        params = ema_params
    if use_ema and ema_params is None:
        ema_params = params
    if not use_ema:
        ema_params = None

    state = EvalState(
        step=jnp.asarray(step, dtype=jnp.int32),
        params=params,
        ema_params=ema_params,
    )
    log_for_0("Restored lightweight eval state from checkpoint at %s", workdir)
    return state


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load .pt checkpoints. Install torch or use a Flax checkpoint instead."
        ) from exc
    return torch


def _to_numpy(value):
    if isinstance(value, np.ndarray):
        return value
    try:
        torch = _import_torch()
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _set_param(tree, path, value):
    node = tree
    parts = path.split("/")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def _transpose_linear(weight):
    return _to_numpy(weight).T


def _slice_patchwise_output_channels(weight, bias, *, patch_size, target_out_channels):
    """Keep the first target_out_channels within each patch-position channel group."""
    weight_np = _to_numpy(weight)
    bias_np = _to_numpy(bias)
    num_patch_positions = patch_size * patch_size
    source_out_channels = weight_np.shape[0] // num_patch_positions
    reshaped_w = weight_np.reshape(num_patch_positions, source_out_channels, weight_np.shape[1])
    reshaped_b = bias_np.reshape(num_patch_positions, source_out_channels)
    sliced_w = reshaped_w[:, :target_out_channels, :].reshape(
        num_patch_positions * target_out_channels,
        weight_np.shape[1],
    )
    sliced_b = reshaped_b[:, :target_out_channels].reshape(
        num_patch_positions * target_out_channels,
    )
    return sliced_w, sliced_b


def _convert_qkv(weight, bias):
    w = _to_numpy(weight)
    b = _to_numpy(bias)
    q, k, v = np.split(w, 3, axis=0)
    qb, kb, vb = np.split(b, 3, axis=0)
    return (
        q.T,
        qb,
        k.T,
        kb,
        v.T,
        vb,
    )


def _load_torch_checkpoint_state_dict(workdir, prefer_ema=True):
    torch = _import_torch()
    try:
        raw = torch.load(workdir, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        raw = torch.load(workdir, map_location="cpu")
    if (
        isinstance(raw, dict)
        and prefer_ema
        and "model_ema1" in raw
        and isinstance(raw["model_ema1"], dict)
    ):
        raw = raw["model_ema1"]
    elif (
        isinstance(raw, dict)
        and prefer_ema
        and "model_ema2" in raw
        and isinstance(raw["model_ema2"], dict)
    ):
        raw = raw["model_ema2"]
    elif isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    elif isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
        raw = raw["model"]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported torch checkpoint format: {workdir}")
    return raw


def _source_has_key(source_dict, key):
    return key in source_dict or f"net.{key}" in source_dict


def _source_get(source_dict, key):
    if key in source_dict:
        return source_dict[key]
    net_key = f"net.{key}"
    if net_key in source_dict:
        return source_dict[net_key]
    raise KeyError(key)


def _source_keys_without_net_prefix(source_dict):
    for key in source_dict.keys():
        yield key[4:] if key.startswith("net.") else key


def _convert_torch_sit_common_state(source_dict):
    """Convert SiT checkpoint tensors shared across Flax SiT variants."""
    state = {}
    # Patch embedding
    if "x_embedder.proj.weight" in source_dict:
        _set_param(
            state,
            "x_embedder/proj/kernel",
            np.transpose(_to_numpy(source_dict["x_embedder.proj.weight"]), (2, 3, 1, 0)),
        )
    if "x_embedder.proj.bias" in source_dict:
        _set_param(state, "x_embedder/proj/bias", _to_numpy(source_dict["x_embedder.proj.bias"]))

    # Positional embedding
    if "pos_embed" in source_dict:
        _set_param(state, "pos_embed", _to_numpy(source_dict["pos_embed"]))

    # Label embedding
    if "y_embedder.embedding_table.weight" in source_dict:
        _set_param(
            state,
            "y_embedder/embedding_table/_flax_embedding/embedding",
            _to_numpy(source_dict["y_embedder.embedding_table.weight"]),
        )

    return state


def _extract_sit_time_embedder_params(source_dict):
    if "t_embedder.mlp.0.weight" not in source_dict:
        return None
    return (
        _transpose_linear(source_dict["t_embedder.mlp.0.weight"]),
        _to_numpy(source_dict["t_embedder.mlp.0.bias"]),
        _transpose_linear(source_dict["t_embedder.mlp.2.weight"]),
        _to_numpy(source_dict["t_embedder.mlp.2.bias"]),
    )


def _set_sit_time_embedder(state, name, params):
    fc1_w, fc1_b, fc2_w, fc2_b = params
    _set_param(state, f"{name}/fc1/_flax_linear/kernel", fc1_w)
    _set_param(state, f"{name}/fc1/_flax_linear/bias", fc1_b)
    _set_param(state, f"{name}/fc2/_flax_linear/kernel", fc2_w)
    _set_param(state, f"{name}/fc2/_flax_linear/bias", fc2_b)


def _extract_sit_block_params(source_dict, path_prefix):
    qkv_weight = source_dict[f"{path_prefix}.attn.qkv.weight"]
    qkv_bias = source_dict[f"{path_prefix}.attn.qkv.bias"]
    q_w, q_b, k_w, k_b, v_w, v_b = _convert_qkv(qkv_weight, qkv_bias)
    out_w = _transpose_linear(source_dict[f"{path_prefix}.attn.proj.weight"])
    out_b = _to_numpy(source_dict[f"{path_prefix}.attn.proj.bias"])
    mlp_fc1_w = _transpose_linear(source_dict[f"{path_prefix}.mlp.fc1.weight"])
    mlp_fc1_b = _to_numpy(source_dict[f"{path_prefix}.mlp.fc1.bias"])
    mlp_fc2_w = _transpose_linear(source_dict[f"{path_prefix}.mlp.fc2.weight"])
    mlp_fc2_b = _to_numpy(source_dict[f"{path_prefix}.mlp.fc2.bias"])
    ada_w = _transpose_linear(source_dict[f"{path_prefix}.adaLN_modulation.1.weight"])
    ada_b = _to_numpy(source_dict[f"{path_prefix}.adaLN_modulation.1.bias"])
    return q_w, q_b, k_w, k_b, v_w, v_b, out_w, out_b, mlp_fc1_w, mlp_fc1_b, mlp_fc2_w, mlp_fc2_b, ada_w, ada_b


def _assign_sit_block(state, flax_prefix, block_params):
    (
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        out_w,
        out_b,
        mlp_fc1_w,
        mlp_fc1_b,
        mlp_fc2_w,
        mlp_fc2_b,
        ada_w,
        ada_b,
    ) = block_params
    _set_param(state, f"{flax_prefix}/attn/q_proj/_flax_linear/kernel", q_w)
    _set_param(state, f"{flax_prefix}/attn/q_proj/_flax_linear/bias", q_b)
    _set_param(state, f"{flax_prefix}/attn/k_proj/_flax_linear/kernel", k_w)
    _set_param(state, f"{flax_prefix}/attn/k_proj/_flax_linear/bias", k_b)
    _set_param(state, f"{flax_prefix}/attn/v_proj/_flax_linear/kernel", v_w)
    _set_param(state, f"{flax_prefix}/attn/v_proj/_flax_linear/bias", v_b)
    _set_param(state, f"{flax_prefix}/attn/out_proj/_flax_linear/kernel", out_w)
    _set_param(state, f"{flax_prefix}/attn/out_proj/_flax_linear/bias", out_b)
    _set_param(state, f"{flax_prefix}/mlp/fc1/_flax_linear/kernel", mlp_fc1_w)
    _set_param(state, f"{flax_prefix}/mlp/fc1/_flax_linear/bias", mlp_fc1_b)
    _set_param(state, f"{flax_prefix}/mlp/fc2/_flax_linear/kernel", mlp_fc2_w)
    _set_param(state, f"{flax_prefix}/mlp/fc2/_flax_linear/bias", mlp_fc2_b)
    _set_param(state, f"{flax_prefix}/adaLN_modulation/_flax_linear/kernel", ada_w)
    _set_param(state, f"{flax_prefix}/adaLN_modulation/_flax_linear/bias", ada_b)


def _convert_torch_sit_state_dict_to_flax_mf(source_dict):
    """Convert a SiT PyTorch state_dict into the dual-head Flax SiT tree."""
    state = _convert_torch_sit_common_state(source_dict)

    # Time / interval / omega embedder reuse
    # The pretrained SiT timestep embedder is reused for all scalar embedders,
    # including both t_embedder and h_embedder (the r-related signal).
    # That means both t and r conditioning are initialized from the same SiT weights.
    time_params = _extract_sit_time_embedder_params(source_dict)
    if time_params is not None:
        for name in [
            "t_embedder",
            "h_embedder",
            "omega_embedder",
            "t_min_embedder",
            "t_max_embedder",
        ]:
            _set_sit_time_embedder(state, name, time_params)

    # Transformer blocks
    blocks = [key for key in source_dict.keys() if key.startswith("blocks.")]
    block_indices = sorted({int(k.split(".")[1]) for k in blocks})
    if block_indices:
        total_blocks = max(block_indices) + 1
        shared_depth = total_blocks - 8
        for i in range(total_blocks):
            path_prefix = f"blocks.{i}"
            block_params = _extract_sit_block_params(source_dict, path_prefix)

            if i < shared_depth:
                _assign_sit_block(state, f"shared_blocks_{i}", block_params)
            else:
                head_idx = i - shared_depth
                _assign_sit_block(state, f"u_heads_{head_idx}", block_params)
                _assign_sit_block(state, f"v_heads_{head_idx}", block_params)

    # Final output layers
    if "final_layer.linear.weight" in source_dict:
        final_w = _transpose_linear(source_dict["final_layer.linear.weight"])
        final_b = _to_numpy(source_dict["final_layer.linear.bias"])
        final_ada_w = _transpose_linear(source_dict["final_layer.adaLN_modulation.1.weight"])
        final_ada_b = _to_numpy(source_dict["final_layer.adaLN_modulation.1.bias"])
        for head in ["u_final_layer", "v_final_layer"]:
            _set_param(state, f"{head}/linear/_flax_linear/kernel", final_w)
            _set_param(state, f"{head}/linear/_flax_linear/bias", final_b)
            _set_param(state, f"{head}/adaLN_modulation/_flax_linear/kernel", final_ada_w)
            _set_param(state, f"{head}/adaLN_modulation/_flax_linear/bias", final_ada_b)

    return {"net": state}


def _convert_torch_sit_state_dict_to_flax_exact(source_dict, target_state=None):
    """Convert a SiT PyTorch state_dict into the exact Flax SiT tree."""
    state = _convert_torch_sit_common_state(source_dict)

    time_params = _extract_sit_time_embedder_params(source_dict)
    if time_params is not None:
        _set_sit_time_embedder(state, "t_embedder", time_params)
        target_net_state = (
            target_state.get("net", {}) if isinstance(target_state, dict) else {}
        )
        if isinstance(target_net_state, dict) and "r_embedder" in target_net_state:
            _set_sit_time_embedder(state, "r_embedder", time_params)

    blocks = [key for key in source_dict.keys() if key.startswith("blocks.")]
    block_indices = sorted({int(k.split(".")[1]) for k in blocks})
    for i in block_indices:
        path_prefix = f"blocks.{i}"
        block_params = _extract_sit_block_params(source_dict, path_prefix)
        _assign_sit_block(state, f"blocks_{i}", block_params)

    if "final_layer.linear.weight" in source_dict:
        final_w = _transpose_linear(source_dict["final_layer.linear.weight"])
        final_b = _to_numpy(source_dict["final_layer.linear.bias"])
        final_ada_w = _transpose_linear(source_dict["final_layer.adaLN_modulation.1.weight"])
        final_ada_b = _to_numpy(source_dict["final_layer.adaLN_modulation.1.bias"])
        _set_param(state, "final_layer/linear/_flax_linear/kernel", final_w)
        _set_param(state, "final_layer/linear/_flax_linear/bias", final_b)
        _set_param(state, "final_layer/adaLN_modulation/_flax_linear/kernel", final_ada_w)
        _set_param(state, "final_layer/adaLN_modulation/_flax_linear/bias", final_ada_b)

    return {"net": state}


def _convert_torch_sit_state_dict_to_flax_dmf(
    source_dict,
    encoder_depth=20,
    target_state=None,
    target_model_config=None,
):
    """Convert a SiT PyTorch state_dict into the decoupled single-head Flax SiT tree."""
    state = _convert_torch_sit_common_state(source_dict)

    time_params = _extract_sit_time_embedder_params(source_dict)
    if time_params is not None:
        _set_sit_time_embedder(state, "t_embedder", time_params)
        target_net_state = target_state.get("net", {}) if isinstance(target_state, dict) else {}
        model_cfg = target_model_config or {}
        use_context = bool(model_cfg.get("use_context_guidance_conditioning", False))
        use_adaln = bool(
            model_cfg.get("use_adaln_guidance_scale_conditioning", False)
        )
        adaln_init = model_cfg.get("adaln_guidance_scale_init", "timestep")

        if isinstance(target_net_state, dict):
            if use_context:
                for name in ["omega_embedder", "t_min_embedder", "t_max_embedder"]:
                    if name in target_net_state:
                        _set_sit_time_embedder(state, name, time_params)
            elif use_adaln and adaln_init == "timestep" and "omega_embedder" in target_net_state:
                _set_sit_time_embedder(state, "omega_embedder", time_params)

    blocks = [key for key in source_dict.keys() if key.startswith("blocks.")]
    block_indices = sorted({int(k.split(".")[1]) for k in blocks})
    if block_indices:
        total_blocks = max(block_indices) + 1
        for i in range(total_blocks):
            path_prefix = f"blocks.{i}"
            block_params = _extract_sit_block_params(source_dict, path_prefix)
            if i < encoder_depth:
                _assign_sit_block(state, f"encoder_blocks_{i}", block_params)
            else:
                _assign_sit_block(state, f"decoder_blocks_{i - encoder_depth}", block_params)

    if "final_layer.linear.weight" in source_dict:
        final_w_raw, final_b_raw = _slice_patchwise_output_channels(
            source_dict["final_layer.linear.weight"],
            source_dict["final_layer.linear.bias"],
            patch_size=2,
            target_out_channels=4,
        )
        final_w = final_w_raw.T
        final_b = final_b_raw
        final_ada_w = _transpose_linear(source_dict["final_layer.adaLN_modulation.1.weight"])
        final_ada_b = _to_numpy(source_dict["final_layer.adaLN_modulation.1.bias"])
        _set_param(state, "final_layer/linear/_flax_linear/kernel", final_w)
        _set_param(state, "final_layer/linear/_flax_linear/bias", final_b)
        _set_param(state, "final_layer/adaLN_modulation/_flax_linear/kernel", final_ada_w)
        _set_param(state, "final_layer/adaLN_modulation/_flax_linear/bias", final_ada_b)

    return {"net": state}


def _assign_jit_block(state, source_dict, source_prefix, flax_prefix):
    """Copy one JiT transformer block from a torch state_dict into ``state``."""
    _set_param(
        state,
        f"{flax_prefix}/norm1/kernel",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.norm1.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/q_norm/kernel",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.attn.q_norm.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/k_norm/kernel",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.attn.k_norm.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/qkv/_flax_linear/kernel",
        _transpose_linear(_source_get(source_dict, f"{source_prefix}.attn.qkv.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/qkv/_flax_linear/bias",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.attn.qkv.bias")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/proj/_flax_linear/kernel",
        _transpose_linear(_source_get(source_dict, f"{source_prefix}.attn.proj.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/attn/proj/_flax_linear/bias",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.attn.proj.bias")),
    )
    _set_param(
        state,
        f"{flax_prefix}/norm2/kernel",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.norm2.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/mlp/w12/_flax_linear/kernel",
        _transpose_linear(_source_get(source_dict, f"{source_prefix}.mlp.w12.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/mlp/w12/_flax_linear/bias",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.mlp.w12.bias")),
    )
    _set_param(
        state,
        f"{flax_prefix}/mlp/w3/_flax_linear/kernel",
        _transpose_linear(_source_get(source_dict, f"{source_prefix}.mlp.w3.weight")),
    )
    _set_param(
        state,
        f"{flax_prefix}/mlp/w3/_flax_linear/bias",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.mlp.w3.bias")),
    )
    _set_param(
        state,
        f"{flax_prefix}/adaLN_modulation/_flax_linear/kernel",
        _transpose_linear(
            _source_get(source_dict, f"{source_prefix}.adaLN_modulation.1.weight")
        ),
    )
    _set_param(
        state,
        f"{flax_prefix}/adaLN_modulation/_flax_linear/bias",
        _to_numpy(_source_get(source_dict, f"{source_prefix}.adaLN_modulation.1.bias")),
    )


def _assign_jit_common_state(source_dict, state):
    """Copy JiT embedders / patch-embed / final-layer shared by JiT variants."""
    if _source_has_key(source_dict, "pos_embed"):
        _set_param(state, "pos_embed", _to_numpy(_source_get(source_dict, "pos_embed")))
    if _source_has_key(source_dict, "in_context_posemb"):
        _set_param(
            state,
            "in_context_posemb",
            _to_numpy(_source_get(source_dict, "in_context_posemb")),
        )

    if _source_has_key(source_dict, "t_embedder.mlp.0.weight"):
        _set_param(
            state,
            "t_embedder/mlp/layers_0/_flax_linear/kernel",
            _transpose_linear(_source_get(source_dict, "t_embedder.mlp.0.weight")),
        )
        _set_param(
            state,
            "t_embedder/mlp/layers_0/_flax_linear/bias",
            _to_numpy(_source_get(source_dict, "t_embedder.mlp.0.bias")),
        )
        _set_param(
            state,
            "t_embedder/mlp/layers_2/_flax_linear/kernel",
            _transpose_linear(_source_get(source_dict, "t_embedder.mlp.2.weight")),
        )
        _set_param(
            state,
            "t_embedder/mlp/layers_2/_flax_linear/bias",
            _to_numpy(_source_get(source_dict, "t_embedder.mlp.2.bias")),
        )

    if _source_has_key(source_dict, "y_embedder.embedding_table.weight"):
        _set_param(
            state,
            "y_embedder/embedding_table/_flax_embedding/embedding",
            _to_numpy(_source_get(source_dict, "y_embedder.embedding_table.weight")),
        )

    if _source_has_key(source_dict, "x_embedder.proj1.weight"):
        _set_param(
            state,
            "x_embedder/proj1/kernel",
            np.transpose(
                _to_numpy(_source_get(source_dict, "x_embedder.proj1.weight")),
                (2, 3, 1, 0),
            ),
        )
    if _source_has_key(source_dict, "x_embedder.proj2.weight"):
        _set_param(
            state,
            "x_embedder/proj2/kernel",
            np.transpose(
                _to_numpy(_source_get(source_dict, "x_embedder.proj2.weight")),
                (2, 3, 1, 0),
            ),
        )
        _set_param(
            state,
            "x_embedder/proj2/bias",
            _to_numpy(_source_get(source_dict, "x_embedder.proj2.bias")),
        )

    if _source_has_key(source_dict, "final_layer.norm_final.weight"):
        _set_param(
            state,
            "final_layer/norm_final/kernel",
            _to_numpy(_source_get(source_dict, "final_layer.norm_final.weight")),
        )
        _set_param(
            state,
            "final_layer/linear/_flax_linear/kernel",
            _transpose_linear(_source_get(source_dict, "final_layer.linear.weight")),
        )
        _set_param(
            state,
            "final_layer/linear/_flax_linear/bias",
            _to_numpy(_source_get(source_dict, "final_layer.linear.bias")),
        )
        _set_param(
            state,
            "final_layer/adaLN_modulation/_flax_linear/kernel",
            _transpose_linear(_source_get(source_dict, "final_layer.adaLN_modulation.1.weight")),
        )
        _set_param(
            state,
            "final_layer/adaLN_modulation/_flax_linear/bias",
            _to_numpy(_source_get(source_dict, "final_layer.adaLN_modulation.1.bias")),
        )


def _jit_source_block_indices(source_dict):
    return sorted(
        {
            int(key.split(".")[1])
            for key in _source_keys_without_net_prefix(source_dict)
            if key.startswith("blocks.")
        }
    )


def _convert_torch_jit_state_dict_to_flax_dmf(source_dict, encoder_depth=24):
    """Convert a JiT PyTorch state_dict into the decoupled single-head Flax JiT tree.

    The pretrained JiT block stack is split into a ``encoder_depth``-block encoder
    (``encoder_blocks_i``) and the remaining decoder blocks (``decoder_blocks_j``),
    matching :class:`models.jit.imfJiT_DMF`.
    """
    state = {}
    _assign_jit_common_state(source_dict, state)

    for i in _jit_source_block_indices(source_dict):
        source_prefix = f"blocks.{i}"
        if i < encoder_depth:
            flax_prefix = f"encoder_blocks_{i}"
        else:
            flax_prefix = f"decoder_blocks_{i - encoder_depth}"
        _assign_jit_block(state, source_dict, source_prefix, flax_prefix)

    return {"net": state}


def _convert_torch_jit_state_dict_to_flax(source_dict):
    """Convert a JiT PyTorch state_dict into the Flax JiT parameter tree."""
    state = {}
    _assign_jit_common_state(source_dict, state)
    for i in _jit_source_block_indices(source_dict):
        _assign_jit_block(state, source_dict, f"blocks.{i}", f"blocks_{i}")
    return {"net": state}


def _target_uses_dmf_sit_layout(target_state):
    if not isinstance(target_state, dict):
        return False
    net_state = target_state.get("net")
    return isinstance(net_state, dict) and "encoder_blocks_0" in net_state


def _target_uses_jit_layout(target_state):
    if not isinstance(target_state, dict):
        return False
    net_state = target_state.get("net")
    if not isinstance(net_state, dict):
        return False
    x_embedder = net_state.get("x_embedder")
    final_layer = net_state.get("final_layer")
    return (
        isinstance(x_embedder, dict)
        and "proj1" in x_embedder
        and isinstance(final_layer, dict)
        and "norm_final" in final_layer
    )


def _target_uses_dmf_jit_layout(target_state):
    """Detect the decoupled single-head pixel-space JiT (imfJiT_DMF) layout.

    Distinguished from the plain JiT layout by the encoder/decoder block split,
    and from the DMF SiT layout by the JiT-style bottleneck patch-embed (proj1).
    """
    if not isinstance(target_state, dict):
        return False
    net_state = target_state.get("net")
    if not isinstance(net_state, dict):
        return False
    x_embedder = net_state.get("x_embedder")
    return (
        isinstance(x_embedder, dict)
        and "proj1" in x_embedder
        and "encoder_blocks_0" in net_state
    )


def _infer_dmf_jit_encoder_depth(target_state):
    if not _target_uses_dmf_jit_layout(target_state):
        return None
    net_state = target_state["net"]
    return sum(1 for key in net_state if key.startswith("encoder_blocks_"))


def _target_uses_exact_sit_layout(target_state):
    if not isinstance(target_state, dict):
        return False
    net_state = target_state.get("net")
    return (
        isinstance(net_state, dict)
        and "blocks_0" in net_state
        and "shared_blocks_0" not in net_state
        and "encoder_blocks_0" not in net_state
    )


def _infer_dmf_encoder_depth(target_state):
    if not _target_uses_dmf_sit_layout(target_state):
        return None
    net_state = target_state["net"]
    return sum(1 for key in net_state if key.startswith("encoder_blocks_"))


def _shape_matches(target_value, source_value):
    if not hasattr(target_value, "shape") or not hasattr(source_value, "shape"):
        return False
    return target_value.shape == source_value.shape


def load_checkpoint_params(workdir, prefer_ema=True, target_state=None, target_model_config=None):
    """
    Load params or ema_params from a checkpoint without shape adaptation.
    Supports Flax checkpoints and PyTorch .pt checkpoints.
    """
    workdir = os.path.abspath(workdir)
    if os.path.isfile(workdir) and workdir.endswith((".pt", ".pth", ".pth.tar")):
        source_tree = _load_torch_checkpoint_state_dict(workdir, prefer_ema=prefer_ema)
        log_for_0("Loaded PyTorch checkpoint from {}".format(workdir))
        if _target_uses_dmf_jit_layout(target_state):
            encoder_depth = _infer_dmf_jit_encoder_depth(target_state)
            return _convert_torch_jit_state_dict_to_flax_dmf(
                source_tree,
                encoder_depth=24 if encoder_depth is None else encoder_depth,
            )
        if _target_uses_jit_layout(target_state):
            return _convert_torch_jit_state_dict_to_flax(source_tree)
        if _target_uses_exact_sit_layout(target_state):
            return _convert_torch_sit_state_dict_to_flax_exact(
                source_tree,
                target_state=target_state,
            )
        if _target_uses_dmf_sit_layout(target_state):
            encoder_depth = _infer_dmf_encoder_depth(target_state)
            return _convert_torch_sit_state_dict_to_flax_dmf(
                source_tree,
                encoder_depth=20 if encoder_depth is None else encoder_depth,
                target_state=target_state,
                target_model_config=target_model_config,
            )
        return _convert_torch_sit_state_dict_to_flax_mf(source_tree)

    restored = checkpoints.restore_checkpoint(workdir, target=None)
    log_for_0("Loaded raw checkpoint from {}".format(workdir))

    source_tree = restored.get("ema_params") if prefer_ema else None
    if source_tree is None:
        source_tree = restored.get("params")
    if source_tree is None:
        raise ValueError(f"No params/ema_params found in checkpoint: {workdir}")
    return source_tree


def restore_partial_checkpoint(state, workdir, prefer_ema=True, target_model_config=None):
    """
    Restore shape-compatible model parameters from a checkpoint.

    This is useful for fine-tuning on a new dataset where some parameter shapes
    differ, such as the class embedding table after changing num_classes.
    Optimizer state and training step are intentionally left fresh.
    """
    workdir = os.path.abspath(workdir)
    target_state = serialization.to_state_dict(state.params)
    source_tree = load_checkpoint_params(
        workdir,
        prefer_ema=prefer_ema,
        target_state=target_state,
        target_model_config=target_model_config,
    )
    source_state = serialization.to_state_dict(source_tree)

    loaded_count = 0
    skipped_count = 0
    skipped_examples = []

    def merge_state(target_subtree, source_subtree, key_path=()):
        nonlocal loaded_count, skipped_count, skipped_examples

        if isinstance(target_subtree, dict):
            merged = {}
            source_subtree = source_subtree if isinstance(source_subtree, dict) else {}
            for key, target_value in target_subtree.items():
                merged[key] = merge_state(
                    target_value,
                    source_subtree.get(key),
                    key_path + (key,),
                )
            return merged

        if source_subtree is not None and _shape_matches(target_subtree, source_subtree):
            loaded_count += 1
            return source_subtree

        skipped_count += 1
        if len(skipped_examples) < 10:
            skipped_examples.append("/".join(key_path))
        return target_subtree

    merged_state = merge_state(target_state, source_state)
    merged_params = serialization.from_state_dict(state.params, merged_state)
    ema_params = None
    if getattr(state, "ema_params", None) is not None:
        ema_params = jax.tree_util.tree_map(
            lambda x: jnp.array(x, copy=True),
            merged_params,
        )
    new_state = state.replace(params=merged_params, ema_params=ema_params)

    log_for_0(
        "Partially restored checkpoint: loaded %d tensors, skipped %d tensors.",
        loaded_count,
        skipped_count,
    )
    if skipped_examples:
        log_for_0("Skipped tensor examples: %s", ", ".join(skipped_examples))

    return new_state


def save_checkpoint(state, workdir):
    """
    Saves the model state to a checkpoint in the specified working directory.
    """
    workdir = os.path.abspath(workdir)
    # Save only one copy from device 0.
    state = jax.device_get(
        jax.tree_util.tree_map(
            lambda x: x if x is None else x[0],
            state,
            is_leaf=lambda x: x is None,
        )
    )
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=3)
    log_for_0("Checkpoint step %d saved.", step)


def save_best_checkpoint(state, workdir, step=None, keep=1, eval_state_only=False):
    """
    Saves the model state to a best-checkpoint directory.

    When keep=1, this also removes any previously saved best checkpoint
    directories so only the latest best checkpoint remains.
    """
    workdir = os.path.abspath(workdir)
    if eval_state_only:
        state_to_save = EvalState(
            step=state.step,
            params=state.params,
            ema_params=getattr(state, "ema_params", None),
        )
    else:
        state_to_save = state
    state_to_save = jax.device_get(
        jax.tree_util.tree_map(
            lambda x: x if x is None else x[0],
            state_to_save,
            is_leaf=lambda x: x is None,
        )
    )
    ckpt_step = int(state_to_save.step) if step is None else int(step)
    log_for_0("Saving best checkpoint step %d to %s.", ckpt_step, workdir)
    checkpoints.save_checkpoint_multiprocess(
        workdir,
        state_to_save,
        ckpt_step,
        keep=keep,
        overwrite=True,
    )
    log_for_0("Best checkpoint step %d saved.", ckpt_step)

    if keep == 1:
        for entry in os.listdir(workdir):
            if not entry.startswith("checkpoint_"):
                continue
            if entry == f"checkpoint_{ckpt_step}":
                continue
            path = os.path.join(workdir, entry)
            if os.path.isdir(path):
                try:
                    shutil.rmtree(path)
                    log_for_0("Removed old best checkpoint directory %s.", path)
                except Exception as exc:
                    log_for_0(
                        "Failed to remove old best checkpoint directory %s: %s",
                        path,
                        exc,
                    )
