"""Parameter-tree helpers for full iMF variants."""


def use_v_only_teacher_source_copies(model_config):
    return bool(model_config.get("use_v_only_teacher_source_copies", False))


def v_only_teacher_source_compatible(model_config):
    model_str = str(model_config.get("model_str", ""))
    return bool(model_config.get("use_auxiliary_v_head", False)) and (
        "DiT" in model_str or ("SiT" in model_str and "DMF" not in model_str)
    )


def extract_v_only_params(param_tree):
    """Drop the u-only branch from an iMF auxiliary-head parameter tree."""
    if not isinstance(param_tree, dict):
        return param_tree

    filtered = {}
    for key, value in param_tree.items():
        if key == "u_final_layer" or key.startswith("u_heads_"):
            continue
        filtered[key] = extract_v_only_params(value)
    return filtered


def is_v_only_param_tree(param_tree):
    if not isinstance(param_tree, dict):
        return False
    keys = set(param_tree.keys())
    has_v = "v_final_layer" in keys or any(key.startswith("v_heads_") for key in keys)
    has_u = "u_final_layer" in keys or any(key.startswith("u_heads_") for key in keys)
    return has_v and not has_u
