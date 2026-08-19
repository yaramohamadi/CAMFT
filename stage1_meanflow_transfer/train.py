"""
Training and evaluation for improved MeanFlow.
"""

import dataclasses
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import os
import torch
from flax import jax_utils
from jax import lax, random
from copy import deepcopy
from functools import partial
from PIL import Image, ImageDraw

from imf import iMeanFlow, generate

import utils.input_pipeline as input_pipeline
from utils.ckpt_util import (
    load_checkpoint_params,
    save_best_checkpoint,
    save_checkpoint,
    restore_checkpoint,
    restore_eval_checkpoint,
    restore_partial_checkpoint,
)
from utils.ema_util import ema_schedules, update_ema
from utils.imf_param_util import (
    extract_v_only_params,
    use_v_only_teacher_source_copies,
)
from utils.logging_util import MetricsTracker, Timer, log_for_0, Writer
from utils.vae_util import LatentManager
from utils.lr_utils import lr_schedules
from utils.eval_csv_util import append_eval_metrics_row
from utils.preview_util import (
    format_preview_guidance_label,
    generate_preview_samples_eager,
    generate_preview_samples_first_device,
    make_side_by_side_preview_panel,
    make_stacked_grid_panel,
)
from utils.sample_util import (
    get_image_metric_evaluator,
    get_sample_device_batch_size,
    get_sample_devices,
    get_sample_local_device_count,
    get_sampling_param_dtype,
    has_controllable_sampling_guidance,
)
from utils.trainstate_util import create_train_state, TrainState

#######################################################
#                    Train Step                       #
#######################################################


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    metrics = lax.pmean(metrics, axis_name="batch")
    return metrics


def compute_metrics_single_device(dict_losses):
    return {k: jnp.mean(v) for k, v in dict_losses.items()}


def train_step_with_vae(
    state,
    batch,
    rng_init,
    ema_fn,
    lr_fn,
    latent_manager,
    use_ema,
    grad_accum_steps,
    v_only_teacher_source_copies,
):
    """
    Perform a single training step.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]  # [B, H, W, C]
    labels = batch["label"]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)  # [B, H, W, C] sample latent

    def loss_fn(params):
        """loss function used for training."""
        outputs = state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            source_params=state.source_params,
            teacher_params=state.ema_params,
            current_step=state.step,
            rngs=dict(
                gen=rng_base,
            ),
        )
        return outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    lr_value = lr_fn(state.step)
    dict_losses = aux[1]
    metrics = compute_metrics(dict_losses)
    metrics["lr"] = lr_value
    if state.grad_accum is None:
        new_grad_accum = grads
    else:
        new_grad_accum = jax.tree_util.tree_map(
            lambda acc, g: acc + g, state.grad_accum, grads
        )
    new_accum_step = state.grad_accum_step + 1
    should_apply = new_accum_step >= grad_accum_steps

    def apply_update(args):
        current_state, accum_grads = args
        mean_grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum_steps, accum_grads
        )
        updated_state = current_state.apply_gradients(grads=mean_grads)
        if use_ema:
            ema_value = ema_fn(current_state.step)
            ema_target_params = (
                extract_v_only_params(updated_state.params)
                if v_only_teacher_source_copies
                else updated_state.params
            )
            new_ema = update_ema(
                updated_state.ema_params, ema_target_params, ema_value
            )
            updated_state = updated_state.replace(ema_params=new_ema)
        # Keep the pytree structure stable across lax.cond branches.
        reset_grad_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        updated_state = updated_state.replace(
            grad_accum=reset_grad_accum,
            grad_accum_step=jnp.array(0, dtype=jnp.int32),
        )
        return updated_state

    def keep_accumulating(args):
        current_state, accum_grads = args
        return current_state.replace(
            grad_accum=accum_grads,
            grad_accum_step=new_accum_step,
        )

    new_state = jax.lax.cond(
        should_apply,
        apply_update,
        keep_accumulating,
        (state, new_grad_accum),
    )
    metrics["did_update"] = should_apply.astype(jnp.float32)
    return new_state, metrics


def train_step_with_vae_single_device(
    state,
    batch,
    rng_init,
    ema_fn,
    lr_fn,
    latent_manager,
    use_ema,
    grad_accum_steps,
    v_only_teacher_source_copies,
):
    """Single-device variant that avoids pmap collectives/compiler bugs."""
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = rng_step

    images = batch["image"][0]
    labels = batch["label"][0]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)

    def loss_fn(params):
        outputs = state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            source_params=state.source_params,
            teacher_params=state.ema_params,
            current_step=state.step,
            rngs=dict(gen=rng_base),
        )
        return outputs

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    lr_value = lr_fn(state.step)
    dict_losses = aux[1]
    metrics = compute_metrics_single_device(dict_losses)
    metrics["lr"] = lr_value

    if state.grad_accum is None:
        new_grad_accum = grads
    else:
        new_grad_accum = jax.tree_util.tree_map(
            lambda acc, g: acc + g, state.grad_accum, grads
        )
    new_accum_step = state.grad_accum_step + 1
    should_apply = new_accum_step >= grad_accum_steps

    def apply_update(args):
        current_state, accum_grads = args
        mean_grads = jax.tree_util.tree_map(
            lambda g: g / grad_accum_steps, accum_grads
        )
        updated_state = current_state.apply_gradients(grads=mean_grads)
        if use_ema:
            ema_value = ema_fn(current_state.step)
            ema_target_params = (
                extract_v_only_params(updated_state.params)
                if v_only_teacher_source_copies
                else updated_state.params
            )
            new_ema = update_ema(
                updated_state.ema_params, ema_target_params, ema_value
            )
            updated_state = updated_state.replace(ema_params=new_ema)
        # Keep the pytree structure stable across lax.cond branches.
        reset_grad_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        updated_state = updated_state.replace(
            grad_accum=reset_grad_accum,
            grad_accum_step=jnp.array(0, dtype=jnp.int32),
        )
        return updated_state

    def keep_accumulating(args):
        current_state, accum_grads = args
        return current_state.replace(
            grad_accum=accum_grads,
            grad_accum_step=new_accum_step,
        )

    new_state = jax.lax.cond(
        should_apply,
        apply_update,
        keep_accumulating,
        (state, new_grad_accum),
    )
    metrics["did_update"] = should_apply.astype(jnp.float32)
    return new_state, metrics


def _cosine_similarity(a, b, eps=1e-8):
    a = a.reshape((a.shape[0], -1))
    b = b.reshape((b.shape[0], -1))
    num = jnp.sum(a * b, axis=1)
    den = jnp.linalg.norm(a, axis=1) * jnp.linalg.norm(b, axis=1) + eps
    return jnp.mean(num / den)


def debug_step_with_vae(state, batch, rng_init, latent_manager, model):
    """
    Run a single debug forward pass and expose intermediate tensors.
    """
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]
    labels = batch["label"]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)

    outputs = model.apply(
        {"params": state.params},
        images=images,
        labels=labels,
        source_params=state.source_params,
        teacher_params=state.ema_params,
        current_step=state.step,
        rngs=dict(gen=rng_base),
        method=model.debug_forward,
    )

    v_u = outputs["v_u"]
    v_c = outputs["v_c"]
    v_g = outputs["v_g"]
    v_pred = outputs["v_pred"]
    V = outputs["V"]

    metrics = {
        "debug/omega_mean": jnp.mean(outputs["omega"]),
        "debug/w_eff_mean": jnp.mean(outputs["w_eff_mean"]),
        "debug/guidance_blend_mean": jnp.mean(outputs["guidance_blend_mean"]),
        "debug/t_mean": jnp.mean(outputs["t"]),
        "debug/r_mean": jnp.mean(outputs["r"]),
        "debug/fm_fraction": jnp.mean(outputs["fm_mask"]),
        "debug/v_u_mean": jnp.mean(v_u),
        "debug/v_u_abs_mean": jnp.mean(jnp.abs(v_u)),
        "debug/v_c_mean": jnp.mean(v_c),
        "debug/v_c_abs_mean": jnp.mean(jnp.abs(v_c)),
        "debug/v_c_minus_v_u_abs_mean": jnp.mean(jnp.abs(v_c - v_u)),
        "debug/v_pred_mean": jnp.mean(v_pred),
        "debug/v_pred_abs_mean": jnp.mean(jnp.abs(v_pred)),
        "debug/V_mean": jnp.mean(V),
        "debug/V_abs_mean": jnp.mean(jnp.abs(V)),
        "debug/v_g_mean": jnp.mean(v_g),
        "debug/v_g_abs_mean": jnp.mean(jnp.abs(v_g)),
        "debug/v_u_to_v_c_cosine": _cosine_similarity(v_u, v_c),
        "debug/v_pred_to_v_g_cosine": _cosine_similarity(v_pred, v_g),
        "debug/V_to_v_g_cosine": _cosine_similarity(V, v_g),
        "debug/v_pred_to_v_g_mse": jnp.mean((v_pred - v_g) ** 2),
        "debug/V_to_v_g_mse": jnp.mean((V - v_g) ** 2),
    }
    metrics = lax.pmean(metrics, axis_name="batch")
    return outputs, metrics


def debug_step_with_vae_single_device(state, batch, rng_init, latent_manager, model):
    """Single-device debug forward without pmap collectives."""
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = rng_step

    images = batch["image"][0]
    labels = batch["label"][0]

    rng_base, rng_vae = random.split(rng_base)
    images = latent_manager.cached_encode(images, rng_vae)

    outputs = model.apply(
        {"params": state.params},
        images=images,
        labels=labels,
        source_params=state.source_params,
        teacher_params=state.ema_params,
        current_step=state.step,
        rngs=dict(gen=rng_base),
        method=model.debug_forward,
    )

    v_u = outputs["v_u"]
    v_c = outputs["v_c"]
    v_g = outputs["v_g"]
    v_pred = outputs["v_pred"]
    V = outputs["V"]

    metrics = {
        "debug/omega_mean": jnp.mean(outputs["omega"]),
        "debug/w_eff_mean": jnp.mean(outputs["w_eff_mean"]),
        "debug/guidance_blend_mean": jnp.mean(outputs["guidance_blend_mean"]),
        "debug/t_mean": jnp.mean(outputs["t"]),
        "debug/r_mean": jnp.mean(outputs["r"]),
        "debug/fm_fraction": jnp.mean(outputs["fm_mask"]),
        "debug/v_u_mean": jnp.mean(v_u),
        "debug/v_u_abs_mean": jnp.mean(jnp.abs(v_u)),
        "debug/v_c_mean": jnp.mean(v_c),
        "debug/v_c_abs_mean": jnp.mean(jnp.abs(v_c)),
        "debug/v_c_minus_v_u_abs_mean": jnp.mean(jnp.abs(v_c - v_u)),
        "debug/v_pred_mean": jnp.mean(v_pred),
        "debug/v_pred_abs_mean": jnp.mean(jnp.abs(v_pred)),
        "debug/V_mean": jnp.mean(V),
        "debug/V_abs_mean": jnp.mean(jnp.abs(V)),
        "debug/v_g_mean": jnp.mean(v_g),
        "debug/v_g_abs_mean": jnp.mean(jnp.abs(v_g)),
        "debug/v_u_to_v_c_cosine": _cosine_similarity(v_u, v_c),
        "debug/v_pred_to_v_g_cosine": _cosine_similarity(v_pred, v_g),
        "debug/V_to_v_g_cosine": _cosine_similarity(V, v_g),
        "debug/v_pred_to_v_g_mse": jnp.mean((v_pred - v_g) ** 2),
        "debug/V_to_v_g_mse": jnp.mean((V - v_g) ** 2),
    }
    return outputs, metrics


def _latents_to_uint8_images(latent_manager, latents_bhwc):
    num_images = latents_bhwc.shape[0]
    decode_total = latent_manager.batch_size * latent_manager.decode_num_local_devices
    decoded_chunks = []
    for start in range(0, num_images, decode_total):
        chunk = latents_bhwc[start : start + decode_total]
        chunk_size = chunk.shape[0]
        if chunk_size < decode_total:
            pad_shape = (decode_total - chunk_size,) + chunk.shape[1:]
            chunk = jnp.concatenate(
                [chunk, jnp.zeros(pad_shape, dtype=chunk.dtype)],
                axis=0,
            )

        latents_bchw = chunk.transpose(0, 3, 1, 2)
        samples = latent_manager.decode(latents_bchw)
        samples = samples[:chunk_size]
        decoded_chunks.append(samples)

    samples = jnp.concatenate(decoded_chunks, axis=0)
    samples = samples.transpose(0, 2, 3, 1)
    samples = 127.5 * samples + 128.0
    return jnp.clip(samples, 0, 255).astype(jnp.uint8)


def _velocity_step_to_uint8_images(latent_manager, z_t_bhwc, velocity_bhwc, step_scale):
    stepped_latents = z_t_bhwc - step_scale * velocity_bhwc
    return _latents_to_uint8_images(latent_manager, stepped_latents)


def _make_uint8_image_grid(images, grid_size):
    if len(images) != grid_size ** 2:
        raise ValueError(
            f"Number of images must match grid size squared, got {len(images)} and {grid_size}."
        )

    images = np.asarray(images, dtype=np.uint8)
    image_h, image_w, image_c = images.shape[1:]
    rows = []
    for row_idx in range(grid_size):
        row = images[row_idx * grid_size : (row_idx + 1) * grid_size]
        rows.append(np.concatenate(list(row), axis=1))
    return np.concatenate(rows, axis=0).reshape(
        grid_size * image_h, grid_size * image_w, image_c
    )


def _make_side_by_side_preview_panel(step_to_images, grid_size, separator_width=8):
    ordered_steps = sorted(step_to_images.keys())
    grids = [_make_uint8_image_grid(step_to_images[num_steps], grid_size) for num_steps in ordered_steps]

    separator = np.full(
        (grids[0].shape[0], separator_width, grids[0].shape[2]),
        255,
        dtype=np.uint8,
    )
    panel_parts = []
    for idx, grid in enumerate(grids):
        if idx > 0:
            panel_parts.append(separator)
        panel_parts.append(grid)
    return np.concatenate(panel_parts, axis=1)


def _make_stacked_grid_panel(
    image_groups,
    grid_size,
    separator_width=8,
    separator_height=8,
    header_height=24,
):
    ordered_names = list(image_groups.keys())
    grids = [_make_uint8_image_grid(image_groups[name], grid_size) for name in ordered_names]

    panel_parts = []
    for idx, (name, grid) in enumerate(zip(ordered_names, grids)):
        if idx > 0:
            separator = np.full(
                (separator_height, grid.shape[1], grid.shape[2]),
                255,
                dtype=np.uint8,
            )
            panel_parts.append(separator)

        header = Image.new("RGB", (grid.shape[1], header_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(header)
        draw.text((8, 4), name, fill=(0, 0, 0))
        panel_parts.append(np.asarray(header, dtype=np.uint8))
        panel_parts.append(grid)
    return np.concatenate(panel_parts, axis=0)


def _format_preview_guidance_label(omega, t_min, t_max):
    return f"omega={float(omega):g}, t=[{float(t_min):g}, {float(t_max):g}]"


def _generate_preview_samples_first_device(
    state,
    p_sample_step,
    latent_manager,
    ema=True,
    num_samples=None,
    **kwargs,
):
    target_num_samples = num_samples
    if target_num_samples is None:
        raise ValueError("num_samples must be provided for first-device preview generation.")

    num_iters = int(np.ceil(target_num_samples / latent_manager.batch_size))
    samples_all = []

    params = state.ema_params if ema else state.params
    variable = {"params": params}

    log_for_0("Note: the first preview sample may be significantly slower")
    for step in range(num_iters):
        sample_idx = jnp.arange(jax.local_device_count(), dtype=jnp.int32)
        sample_idx = sample_idx + step * jax.local_device_count()
        log_for_0(f"Preview sampling step {step} / {num_iters} on first local device...")

        latent = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
        latent = latent[0]  # first local device only
        decode_total = latent_manager.batch_size * jax.local_device_count()
        if latent.shape[0] < decode_total:
            pad_shape = (decode_total - latent.shape[0],) + latent.shape[1:]
            latent = jnp.concatenate(
                [latent, jnp.zeros(pad_shape, dtype=latent.dtype)],
                axis=0,
            )

        samples = latent_manager.decode(latent)
        samples = samples[: latent_manager.batch_size]
        samples = samples.transpose(0, 2, 3, 1)
        samples = 127.5 * samples + 128.0
        samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)
        samples_all.append(np.asarray(jax.device_get(samples)))

    samples_all = np.concatenate(samples_all, axis=0)
    return samples_all[:target_num_samples]


#######################################################
#               Sampling and Metrics                  #
#######################################################


def sample_step(variable, sample_idx, model, rng_init, device_batch_size, 
                config, num_steps, omega, t_min, t_max):
    """
    sample_idx: each random sampled image corrresponds to a seed
    """
    rng_sample = random.fold_in(rng_init, sample_idx)  # fold

    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max, sample_idx=sample_idx)

    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images


def infer_num_classes_from_latents(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        raise ValueError(f"Latent train directory not found: {train_root}")

    max_label = -1
    num_files = 0
    for filename in os.listdir(train_root):
        if not filename.endswith(".pt"):
            continue
        sample = torch.load(
            os.path.join(train_root, filename),
            map_location="cpu",
            weights_only=False,
        )
        max_label = max(max_label, int(sample["label"]))
        num_files += 1

    if num_files == 0:
        raise ValueError(f"No latent .pt files found under: {train_root}")

    return max_label + 1


def _get_eval_sampling_configs(config):
    """
    Normalize evaluation CFG settings from either eval-style list configs or
    train-style scalar sampling fields.
    """
    sampling = config.sampling
    guidance_controllable = has_controllable_sampling_guidance(config.model)

    intervals = sampling.get("interval", None)
    omegas = sampling.get("omegas", None)
    if (not guidance_controllable) and intervals is not None and omegas is not None:
        t_min, t_max = intervals[0]
        omega = omegas[0]
        return [(omega, t_min, t_max)]
    if intervals is not None and omegas is not None:
        configs = []
        for interval in intervals:
            t_min, t_max = interval
            for omega in omegas:
                configs.append((omega, t_min, t_max))
        return configs

    omega = sampling.get("omega", None)
    t_min = sampling.get("t_min", None)
    t_max = sampling.get("t_max", None)
    if omega is not None and t_min is not None and t_max is not None:
        return [(omega, t_min, t_max)]

    raise ValueError(
        "Evaluation requires either sampling.interval + sampling.omegas "
        "or sampling.omega + sampling.t_min + sampling.t_max in the config."
    )


def _should_run_fid(current_step, training_config):
    forced_fid_steps = str(training_config.get("force_fid_steps", "") or "").strip()
    if forced_fid_steps:
        return current_step in {
            int(step) for step in forced_fid_steps.replace(",", " ").split()
        }

    forced_fid_per_step = int(training_config.get("force_fid_per_step", 0) or 0)
    if forced_fid_per_step > 0:
        return current_step % forced_fid_per_step == 0

    fid_schedule = training_config.get("fid_schedule", [])
    if fid_schedule:
        for schedule_item in fid_schedule:
            from_step = int(schedule_item.get("from_step", 0))
            until_step = schedule_item.get("until_step", None)
            every_steps = int(schedule_item["every_steps"])
            if current_step < from_step:
                continue
            if until_step is not None and current_step >= int(until_step):
                continue
            if (current_step - from_step) % every_steps == 0:
                return True
        return False

    fid_per_step = int(training_config.get("fid_per_step", 0))
    return fid_per_step > 0 and current_step % fid_per_step == 0


def _get_metric_num_steps(config):
    forced_steps = str(config.training.get("force_metric_num_steps", "") or "").strip()
    if forced_steps:
        return tuple(int(step) for step in forced_steps.replace(",", " ").split())

    configured_steps = config.training.get("metric_num_steps", ())
    if configured_steps:
        num_steps = [int(step) for step in configured_steps]
    else:
        num_steps = [int(config.sampling.num_steps)]

    primary_steps = int(config.sampling.num_steps)
    ordered_steps = []
    for step in [primary_steps] + num_steps:
        if step < 1:
            raise ValueError("Metric sampling steps must be >= 1.")
        if step not in ordered_steps:
            ordered_steps.append(step)
    return tuple(ordered_steps)


def _primary_metric_mode(use_ema):
    return "ema" if use_ema else "online"


def _write_eval_metrics_csv(workdir, **row):
    append_eval_metrics_row(workdir, row)


def _set_num_classes_from_data(config):
    if config.dataset.get("num_classes_from_data", False):
        inferred_num_classes = infer_num_classes_from_latents(config.dataset.root)
        config.dataset.num_classes = inferred_num_classes
        config.model.num_classes = inferred_num_classes
        config.sampling.num_classes = inferred_num_classes
        log_for_0(
            "Inferred dataset.num_classes from latent data: %s",
            inferred_num_classes,
        )


#######################################################
#                       Main                          #
#######################################################


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ########### Initialize ###########
    writer = Writer(config, workdir)

    _set_num_classes_from_data(config)

    rng = random.key(config.training.seed)
    image_size = config.dataset.image_size
    metric_device_bsz = int(config.fid.device_batch_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    use_ema = config.training.get("use_ema", True)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = config.training.get("grad_accum_steps", 1)
    v_only_teacher_source_copies = use_v_only_teacher_source_copies(config.model)

    log_for_0("config.training.batch_size: {}".format(config.training.batch_size))
    log_for_0("config.training.use_ema: {}".format(use_ema))
    log_for_0("config.training.max_train_steps: {}".format(max_train_steps))
    log_for_0("config.training.grad_accum_steps: {}".format(grad_accum_steps))
    log_for_0("config.fid.device_batch_size: %s", metric_device_bsz)
    log_for_0("config.fid.sample_device_batch_size: %s", sample_device_bsz)
    log_for_0("sampling local device count: %s", sample_local_device_count)
    local_batch_size = config.training.batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    ########### Create DataLoaders ###########
    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: {}".format(steps_per_epoch))

    ########### Create Model ###########
    model_config = config.model.to_dict()
    valid_model_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    ignored_model_keys = sorted(set(model_config.keys()) - valid_model_keys)
    if ignored_model_keys:
        log_for_0(
            "Ignoring unsupported iMeanFlow model config keys: %s",
            ignored_model_keys,
        )
    model_config = {
        key: value for key, value in model_config.items() if key in valid_model_keys
    }
    model = iMeanFlow(**model_config)

    ########### Create Train State ###########
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(rng, config, model, image_size, lr_fn)

    if config.load_from != "":
        if config.get("partial_load", False):
            state = restore_partial_checkpoint(
                state,
                config.load_from,
                target_model_config=config.model,
            )
        else:
            state = restore_checkpoint(state, config.load_from)
        if use_ema and v_only_teacher_source_copies:
            state = state.replace(ema_params=extract_v_only_params(state.params))
        if config.model.get("training_mode", "imf_jvp") == "imf_jvp_free_src_reg":
            state = state.replace(source_params=deepcopy(state.params))
            log_for_0(
                "Captured frozen source_params from restored initial model for "
                "imf_jvp_free_src_reg."
            )
        elif (
            config.model.get("training_mode", "imf_jvp") == "imf_split_consistency"
            and (
                config.model.get("split_consistency_source_first_prob", 0.0) > 0.0
                or config.model.get("split_consistency_source_second_prob", 0.0) > 0.0
            )
        ):
            state = state.replace(source_params=deepcopy(state.params))
            log_for_0(
                "Captured frozen source_params from restored initial model for "
                "imf_split_consistency source ablations."
            )
        elif config.training.get("capture_source_from_load", False):
            source_target_state = {"net": deepcopy(state.source_params)}
            source_model_config = deepcopy(config.model)
            source_model_config["num_classes"] = int(config.model.source_num_classes)
            source_model_config["target_use_null_class"] = True
            source_model_config["use_dogfit"] = False
            loaded_source_params = load_checkpoint_params(
                config.load_from,
                prefer_ema=True,
                target_state=source_target_state,
                target_model_config=source_model_config,
            )
            source_params = deepcopy(loaded_source_params["net"])
            if v_only_teacher_source_copies:
                source_params = extract_v_only_params(source_params)
            state = state.replace(source_params=source_params)
            log_for_0(
                "Loaded frozen DogFit source_params from pretrained source checkpoint.",
            )

    step = int(state.step)
    epoch_offset = step // steps_per_epoch
    single_device_training = jax.local_device_count() == 1
    if single_device_training:
        log_for_0("Using single-device JIT training fallback on one local GPU.")
    else:
        state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=sample_local_device_count,
    )
    preview_latent_manager = LatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=1,
    )

    ########### Create train and sample pmap ###########

    if single_device_training:
        p_train_step = jax.jit(
            partial(
                train_step_with_vae_single_device,
                rng_init=rng,
                ema_fn=ema_fn,
                lr_fn=lr_fn,
                latent_manager=latent_manager,
                use_ema=use_ema,
                grad_accum_steps=grad_accum_steps,
                v_only_teacher_source_copies=v_only_teacher_source_copies,
            ),
            donate_argnums=(0,),
        )
        p_debug_step = partial(
            debug_step_with_vae_single_device,
            rng_init=rng,
            latent_manager=latent_manager,
            model=model,
        )
    else:
        p_train_step = jax.pmap(
            partial(
                train_step_with_vae,
                rng_init=rng,
                ema_fn=ema_fn,
                lr_fn=lr_fn,
                latent_manager=latent_manager,
                use_ema=use_ema,
                grad_accum_steps=grad_accum_steps,
                v_only_teacher_source_copies=v_only_teacher_source_copies,
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )

        p_debug_step = jax.pmap(
            partial(
                debug_step_with_vae,
                rng_init=rng,
                latent_manager=latent_manager,
                model=model,
            ),
            axis_name="batch",
        )

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(99),
                config=config,
                device_batch_size=sample_device_bsz,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in metric_num_steps
    }
    log_for_0("Metric evaluation sampling steps: %s", metric_num_steps)

    preview_num_steps = config.training.get("preview_num_steps", (1, 2, 4))
    preview_num_steps = (
        tuple(int(num) for num in preview_num_steps)
        if preview_num_steps
        else (1, 2, 4)
    )
    guidance_controllable = has_controllable_sampling_guidance(config.model)
    preview_guidance_scales = config.training.get("preview_guidance_scales", [])
    if preview_guidance_scales:
        preview_guidance_scales = [float(omega) for omega in preview_guidance_scales]
    else:
        preview_guidance_scales = [float(config.sampling.omega)]
    if not guidance_controllable:
        preview_guidance_scales = [1.0]
    preview_t_min = float(config.sampling.t_min)
    preview_t_max = float(config.sampling.t_max)
    eval_sample_kwargs = {
        "omega": 1.0 if not guidance_controllable else float(config.sampling.omega),
        "t_min": preview_t_min,
        "t_max": preview_t_max,
    }
    sample_kwargs = jax_utils.replicate(eval_sample_kwargs, devices=sample_devices)

    def replicate_for_sampling(state_value):
        if not single_device_training:
            return state_value
        return jax_utils.replicate(state_value, devices=sample_devices)

    def state_step_value(state_value):
        step_value = jax.device_get(state_value.step)
        if np.ndim(step_value) == 0:
            return int(step_value)
        return int(step_value[0])

    def metric_scalar(metric_value):
        metric_value = jax.device_get(metric_value)
        if np.ndim(metric_value) == 0:
            return bool(metric_value) if np.asarray(metric_value).dtype == np.bool_ else float(metric_value)
        first_value = np.asarray(metric_value)[0]
        return bool(first_value) if np.asarray(first_value).dtype == np.bool_ else float(first_value)

    sampling_uses_ema = (
        use_ema
        and not v_only_teacher_source_copies
        and not config.training.get("fid_use_online_only", False)
    )

    def log_preview_samples(state_for_logging, step_for_logging):
        num_images = min(int(config.fid.num_images_to_log), int(latent_manager.batch_size))
        grid_size = int(num_images ** 0.5)
        num_images = grid_size ** 2
        if num_images <= 0:
            raise ValueError(
                "Preview logging requires at least one square grid image on the first device."
            )
        log_for_0(
            "Logging %d preview samples at step %d for %s steps and omegas %s.",
            num_images,
            step_for_logging,
            preview_num_steps,
            preview_guidance_scales,
        )
        preview_image_groups = {}
        for omega in preview_guidance_scales:
            preview_images = {}
            for num_steps in preview_num_steps:
                preview_images[num_steps] = generate_preview_samples_eager(
                    state_for_logging,
                    model,
                    preview_latent_manager,
                    config,
                    num_steps,
                    omega,
                    preview_t_min,
                    preview_t_max,
                    ema=sampling_uses_ema,
                    num_samples=num_images,
                    param_dtype=get_sampling_param_dtype(config),
                )

            preview_panel = make_side_by_side_preview_panel(preview_images, grid_size)
            preview_image_groups[
                format_preview_guidance_label(omega, preview_t_min, preview_t_max)
            ] = [preview_panel]

        stacked_preview_panel = make_stacked_grid_panel(preview_image_groups, 1)
        writer.write_images(step_for_logging, {"image_grid": stacked_preview_panel})

    image_metric_evaluator = None
    best_fid_by_steps = {num_steps: float("inf") for num_steps in metric_num_steps}
    best_fd_dino_by_steps = {
        num_steps: float("inf") for num_steps in metric_num_steps
    }
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )
    save_best_fid_only = config.training.get("save_best_fid_only", False)
    save_eval_checkpoint_per_fid = config.training.get(
        "save_eval_checkpoint_per_fid", False
    )
    eval_ckpt_dir = os.path.join(
        workdir,
        config.training.get("eval_checkpoint_dir", "latest_eval"),
    )
    metric_mode = _primary_metric_mode(
        sampling_uses_ema and not config.training.get("fid_use_online_only", False)
    )

    ########### Training Loop ###########
    metrics_tracker = MetricsTracker()
    log_for_0("Initial compilation, this might take some minutes...")

    initial_step = state_step_value(state)
    if (
        initial_step == 0
        and config.training.sample_per_step > 0
        and bool(config.training.get("preview_at_step_zero", True))
    ):
        log_preview_samples(state, 0)

    should_stop = False
    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        ########### Train ###########
        timer = Timer()
        log_for_0("epoch {}...".format(epoch))
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            batch = input_pipeline.prepare_batch_data(batch)
            state, metrics = p_train_step(state, batch)
            current_step = state_step_value(state)
            did_update = bool(metric_scalar(metrics["did_update"]))

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                compilation_time = timer.elapse_with_reset()
                log_for_0("p_train_step compiled in {:.2f}s".format(compilation_time))

            ########### Metrics ###########
            metrics_tracker.update(metrics)  # stream one step in
            should_log = did_update and current_step > 0 and (
                current_step == 1 or current_step % config.training.log_per_step == 0
            )
            if should_log:
                summary = metrics_tracker.finalize()
                logged_steps = 1 if current_step == 1 else config.training.log_per_step
                summary["steps_per_second"] = (
                    logged_steps / timer.elapse_with_reset()
                )
                summary.pop("did_update", None)
                writer.write_scalars(current_step, summary)

                if config.training.get("debug_log_during_train", False):
                    log_for_0("Running debug logging on current batch at step %d.", current_step)
                    debug_outputs, debug_metrics = p_debug_step(state, batch)
                    debug_metrics = {
                        k: float(jnp.asarray(v).mean())
                        for k, v in jax.device_get(debug_metrics).items()
                    }
                    writer.write_scalars(current_step, debug_metrics)

                    debug_num_images = int(config.training.get("debug_num_images", 4))
                    debug_velocity_decode_scale = float(
                        config.training.get("debug_velocity_decode_scale", 0.1)
                    )
                    debug_grid_size = int(debug_num_images ** 0.5)
                    if debug_grid_size ** 2 != debug_num_images:
                        raise ValueError(
                            f"config.training.debug_num_images must be a perfect square, got {debug_num_images}"
                        )
                    x = jax.device_get(debug_outputs["x"]).reshape(-1, *debug_outputs["x"].shape[2:])[:debug_num_images]
                    z_t = jax.device_get(debug_outputs["z_t"]).reshape(-1, *debug_outputs["z_t"].shape[2:])[:debug_num_images]
                    v_u = jax.device_get(debug_outputs["v_u"]).reshape(-1, *debug_outputs["v_u"].shape[2:])[:debug_num_images]
                    v_c = jax.device_get(debug_outputs["v_c"]).reshape(-1, *debug_outputs["v_c"].shape[2:])[:debug_num_images]
                    v_pred = jax.device_get(debug_outputs["v_pred"]).reshape(-1, *debug_outputs["v_pred"].shape[2:])[:debug_num_images]
                    V = jax.device_get(debug_outputs["V"]).reshape(-1, *debug_outputs["V"].shape[2:])[:debug_num_images]
                    v_g = jax.device_get(debug_outputs["v_g"]).reshape(-1, *debug_outputs["v_g"].shape[2:])[:debug_num_images]

                    if config.training.get("debug_log_images", True):
                        debug_image_groups = {
                            "clean": jax.device_get(
                                _latents_to_uint8_images(latent_manager, jnp.asarray(x))
                            ),
                            "noisy": jax.device_get(
                                _latents_to_uint8_images(latent_manager, jnp.asarray(z_t))
                            ),
                            "v_u": jax.device_get(
                                _velocity_step_to_uint8_images(
                                    latent_manager,
                                    jnp.asarray(z_t),
                                    jnp.asarray(v_u),
                                    debug_velocity_decode_scale,
                                )
                            ),
                            "v_c": jax.device_get(
                                _velocity_step_to_uint8_images(
                                    latent_manager,
                                    jnp.asarray(z_t),
                                    jnp.asarray(v_c),
                                    debug_velocity_decode_scale,
                                )
                            ),
                            "v_pred": jax.device_get(
                                _velocity_step_to_uint8_images(
                                    latent_manager,
                                    jnp.asarray(z_t),
                                    jnp.asarray(v_pred),
                                    debug_velocity_decode_scale,
                                )
                            ),
                            "V": jax.device_get(
                                _velocity_step_to_uint8_images(
                                    latent_manager,
                                    jnp.asarray(z_t),
                                    jnp.asarray(V),
                                    debug_velocity_decode_scale,
                                )
                            ),
                            "v_g": jax.device_get(
                                _velocity_step_to_uint8_images(
                                    latent_manager,
                                    jnp.asarray(z_t),
                                    jnp.asarray(v_g),
                                    debug_velocity_decode_scale,
                                )
                            ),
                        }
                        debug_panel = _make_stacked_grid_panel(
                            debug_image_groups,
                            debug_grid_size,
                        )
                        writer.write_images(
                            current_step,
                            {"debug_image_grid": debug_panel},
                        )

            ########### Sampling ###########
            if (
                did_update
                and config.training.sample_per_step > 0
                and current_step > 0
                and current_step % config.training.sample_per_step == 0
            ):
                log_preview_samples(state, current_step)

            ########### FID ###########
            if did_update and current_step > 0 and _should_run_fid(current_step, config.training):
                if image_metric_evaluator is None:
                    image_metric_evaluator = get_image_metric_evaluator(
                        config, writer, latent_manager
                    )
                checkpoint_path_for_csv = ""
                if save_eval_checkpoint_per_fid:
                    save_best_checkpoint(replicate_for_sampling(state), eval_ckpt_dir)
                    checkpoint_path_for_csv = eval_ckpt_dir

                for metric_num_steps, p_metric_sample_step in p_metric_sample_steps.items():
                    log_for_0(
                        "Running metric evaluation at step %d with %d sampling steps.",
                        current_step,
                        metric_num_steps,
                    )
                    result = image_metric_evaluator(
                        replicate_for_sampling(state),
                        p_metric_sample_step,
                        current_step - 1,
                        metric_suffix=f"steps_{metric_num_steps}",
                        **sample_kwargs,
                    )
                    fid = result["fid"]
                    fd_dino = result.get("fd_dino", None)
                    is_best_fid = fid < best_fid_by_steps[metric_num_steps]
                    is_best_fd_dino = (
                        fd_dino is not None
                        and fd_dino < best_fd_dino_by_steps[metric_num_steps]
                    )
                    if is_best_fid:
                        best_fid_by_steps[metric_num_steps] = fid
                    if is_best_fd_dino:
                        best_fd_dino_by_steps[metric_num_steps] = fd_dino

                    row_checkpoint_path = checkpoint_path_for_csv
                    if (
                        metric_num_steps == int(config.sampling.num_steps)
                        and is_best_fid
                    ):
                        log_for_0(
                            "New best FID %.4f at step %d for %d sampling steps. "
                            "Saving best checkpoint to %s.",
                            fid,
                            current_step,
                            metric_num_steps,
                            best_fid_ckpt_dir,
                        )
                        save_best_checkpoint(
                            replicate_for_sampling(state), best_fid_ckpt_dir
                        )
                        row_checkpoint_path = best_fid_ckpt_dir

                    _write_eval_metrics_csv(
                        workdir,
                        eval_phase="train",
                        metric_mode=metric_mode,
                        training_step=current_step,
                        sampling_num_steps=metric_num_steps,
                        omega=eval_sample_kwargs["omega"],
                        t_min=eval_sample_kwargs["t_min"],
                        t_max=eval_sample_kwargs["t_max"],
                        fid=float(fid),
                        inception_score=float(result["is"]),
                        fd_dino="" if fd_dino is None else float(fd_dino),
                        is_best_fid=int(is_best_fid),
                        is_best_fd_dino=int(is_best_fd_dino),
                        checkpoint_path=(
                            os.path.abspath(row_checkpoint_path)
                            if row_checkpoint_path
                            else ""
                        ),
                    )

            if max_train_steps is not None and current_step >= max_train_steps:
                should_stop = True
                break

        ########### Save Checkpoint ###########
        if (
            not save_best_fid_only
            and (
                should_stop
                or (epoch + 1) % config.training.checkpoint_per_epoch == 0
                or (epoch + 1) == config.training.num_epochs
            )
        ):
            save_checkpoint(replicate_for_sampling(state), workdir)

        if should_stop:
            log_for_0("Reached max_train_steps=%d at step %d.", max_train_steps, current_step)
            break

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state

########################################################
#                    Evaluation                        #
########################################################

def just_evaluate(config: ml_collections.ConfigDict, workdir: str):

    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert (
        config.load_from != ""
    ), "config.load_from must be specified for just_evaluate"

    ########### Initialize ###########
    writer = Writer(config, workdir)

    _set_num_classes_from_data(config)

    image_size = config.dataset.image_size
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    use_ema = config.training.get("use_ema", True)
    if config.training.get("fid_use_online_only", False) or use_v_only_teacher_source_copies(config.model):
        use_ema = False
    metric_mode = _primary_metric_mode(use_ema)

    ########### Create Model ###########
    model_config = config.model.to_dict()
    valid_model_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    ignored_model_keys = sorted(set(model_config.keys()) - valid_model_keys)
    if ignored_model_keys:
        log_for_0(
            "Ignoring unsupported iMeanFlow model config keys during eval: %s",
            ignored_model_keys,
        )
    model_config = {
        key: value for key, value in model_config.items() if key in valid_model_keys
    }
    model = iMeanFlow(**model_config, eval=True)

    ########### Restore lightweight Eval State ###########
    state = restore_eval_checkpoint(config.load_from, use_ema=use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    ########### Create Latent Manager ###########

    latent_manager = LatentManager(
        config.dataset.vae,
        sample_device_bsz,
        image_size,
        decode_num_local_devices=sample_local_device_count,
    )

    ########### Create sample pmap ###########

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(99),
                config=config,
                device_batch_size=sample_device_bsz,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    image_metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)
    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in metric_num_steps
    }

    ############ Evaluate over CFG configs ###########
    best_fid = float("inf")
    best_is = float("-inf")
    best_fd_dino = float("inf")
    best_config = None
    best_fd_dino_config = None
    best_fd_dino_at_best_fid = None
    guidance_controllable = has_controllable_sampling_guidance(config.model)
    csv_rows = []
    for metric_num_step, p_metric_sample_step in p_metric_sample_steps.items():
        for omega, t_min, t_max in _get_eval_sampling_configs(config):
            kwargs = {"omega": omega, "t_min": t_min, "t_max": t_max}
            kwargs = jax_utils.replicate(kwargs, devices=sample_devices)
            result = image_metric_evaluator(
                state,
                p_metric_sample_step,
                step,
                not use_ema,
                metric_suffix=f"steps_{metric_num_step}",
                **kwargs,
            )
            fid = result["fid"]
            is_score = result["is"]
            fd_dino = result.get("fd_dino", None)

            row = {
                "sampling_num_steps": metric_num_step,
                "omega": omega,
                "t_min": t_min,
                "t_max": t_max,
                "fid": fid,
                "is": is_score,
                "fd_dino": fd_dino,
            }
            csv_rows.append(row)

            if fid < best_fid:
                best_fid = fid
                best_is = is_score
                best_config = (metric_num_step, omega, t_min, t_max)
                best_fd_dino_at_best_fid = fd_dino
            if fd_dino is not None and fd_dino < best_fd_dino:
                best_fd_dino = fd_dino
                best_fd_dino_config = (metric_num_step, omega, t_min, t_max)

    for row in csv_rows:
        row_config = (
            row["sampling_num_steps"],
            row["omega"],
            row["t_min"],
            row["t_max"],
        )
        _write_eval_metrics_csv(
            workdir,
            eval_phase="eval_only",
            metric_mode=metric_mode,
            training_step=step,
            sampling_num_steps=row["sampling_num_steps"],
            omega=row["omega"],
            t_min=row["t_min"],
            t_max=row["t_max"],
            fid=float(row["fid"]),
            inception_score=float(row["is"]),
            fd_dino="" if row["fd_dino"] is None else float(row["fd_dino"]),
            is_best_fid=int(row_config == best_config),
            is_best_fd_dino=int(
                best_fd_dino_config is not None and row_config == best_fd_dino_config
            ),
            checkpoint_path=os.path.abspath(config.load_from),
        )

    summary = {
        'best_fid': best_fid,
        'best_is': best_is,
    }
    if guidance_controllable:
        summary['sampling_num_steps'] = best_config[0]
        summary['omega'] = best_config[1]
        summary['t_min'] = best_config[2]
        summary['t_max'] = best_config[3]
        log_message = (
            f"Best FID achieved: {best_fid:.2f}, \n"
            f"IS achieved: {best_is:.2f}, \n"
            f"steps: {best_config[0]}, omega: {best_config[1]:.2f}, "
            f"t_min: {best_config[2]:.2f}, t_max: {best_config[3]:.2f}"
        )
    else:
        log_message = (
            f"Best FID achieved: {best_fid:.2f}, \n"
            f"IS achieved: {best_is:.2f}, \n"
            f"single-head SiT-DMF eval used the model's baked fixed-guidance output "
            f"(no sampling-time guidance control)."
        )
    if best_fd_dino_at_best_fid is not None:
        summary['best_fd_dino_at_best_fid'] = best_fd_dino_at_best_fid
        log_message += f", \nFD-DINO at best FID config: {best_fd_dino_at_best_fid:.2f}"
    if best_fd_dino_config is not None and guidance_controllable:
        summary['best_fd_dino'] = best_fd_dino
        summary['best_fd_dino_sampling_num_steps'] = best_fd_dino_config[0]
        summary['best_fd_dino_omega'] = best_fd_dino_config[1]
        summary['best_fd_dino_t_min'] = best_fd_dino_config[2]
        summary['best_fd_dino_t_max'] = best_fd_dino_config[3]
        log_message += (
            f", \nBest FD-DINO achieved: {best_fd_dino:.2f}, "
            f"steps: {best_fd_dino_config[0]}, "
            f"omega: {best_fd_dino_config[1]:.2f}, "
            f"t_min: {best_fd_dino_config[2]:.2f}, "
            f"t_max: {best_fd_dino_config[3]:.2f}"
        )
    elif best_fd_dino_config is not None:
        summary['best_fd_dino'] = best_fd_dino
        log_message += f", \nBest FD-DINO achieved: {best_fd_dino:.2f}"
    log_for_0(log_message)
    writer.write_scalars(step, summary)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
