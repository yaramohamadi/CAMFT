"""Shared preview-grid helpers for training-time sample logging."""

import numpy as np
from PIL import Image, ImageDraw

import jax
import jax.numpy as jnp

from imf import generate
from utils.logging_util import log_for_0
from utils.sample_util import _slice_local_device_axis


def make_uint8_image_grid(images, grid_size):
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


def make_side_by_side_preview_panel(step_to_images, grid_size, separator_width=8):
    ordered_steps = sorted(step_to_images.keys())
    grids = [
        make_uint8_image_grid(step_to_images[num_steps], grid_size)
        for num_steps in ordered_steps
    ]

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


def make_stacked_grid_panel(
    image_groups,
    grid_size,
    separator_height=8,
    header_height=24,
):
    ordered_names = list(image_groups.keys())
    grids = [make_uint8_image_grid(image_groups[name], grid_size) for name in ordered_names]

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


def format_preview_guidance_label(omega, t_min, t_max):
    return f"omega={float(omega):g}, t=[{float(t_min):g}, {float(t_max):g}]"


def generate_preview_samples_first_device(
    state,
    p_sample_step,
    latent_manager,
    ema=True,
    num_samples=None,
    param_dtype=None,
    sample_local_device_count=None,
    **kwargs,
):
    if num_samples is None:
        raise ValueError("num_samples must be provided for first-device preview generation.")

    num_iters = int(np.ceil(num_samples / latent_manager.batch_size))
    samples_all = []

    params = state.ema_params if ema else state.params
    if param_dtype is not None:
        params = jax.tree_util.tree_map(
            lambda x: x.astype(param_dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            params,
        )
    if sample_local_device_count is not None:
        params = _slice_local_device_axis(params, sample_local_device_count)
        kwargs = _slice_local_device_axis(kwargs, sample_local_device_count)
    variable = {"params": params}

    log_for_0("Note: the first preview sample may be significantly slower")
    for step in range(num_iters):
        preview_device_count = (
            jax.local_device_count()
            if sample_local_device_count is None
            else sample_local_device_count
        )
        sample_idx = jnp.arange(preview_device_count, dtype=jnp.int32)
        sample_idx = sample_idx + step * preview_device_count
        log_for_0(f"Preview sampling step {step} / {num_iters} on first local device...")

        latent = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
        latent = latent[0]
        decode_local_device_count = getattr(
            latent_manager,
            "decode_num_local_devices",
            jax.local_device_count(),
        )
        decode_total = latent_manager.batch_size * decode_local_device_count
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
    return samples_all[:num_samples]


def _first_local_device_tree(tree):
    local_device_count = jax.local_device_count()

    def maybe_take_first(x):
        if hasattr(x, "shape") and x.shape and x.shape[0] == local_device_count:
            return x[0]
        return x

    return jax.tree_util.tree_map(maybe_take_first, tree)


def generate_preview_samples_eager(
    state,
    model,
    latent_manager,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    ema=True,
    num_samples=None,
    param_dtype=None,
    rng_seed=99,
):
    """Generate preview samples without pmap/jit-heavy preview compilation."""
    if num_samples is None:
        raise ValueError("num_samples must be provided for eager preview generation.")

    params = state.ema_params if ema else state.params
    params = _first_local_device_tree(params)
    if param_dtype is not None:
        params = jax.tree_util.tree_map(
            lambda x: x.astype(param_dtype)
            if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
            else x,
            params,
        )
    variable = {"params": params}

    samples_all = []
    num_iters = int(np.ceil(num_samples / latent_manager.batch_size))
    for step in range(num_iters):
        batch_num_samples = min(
            latent_manager.batch_size,
            num_samples - step * latent_manager.batch_size,
        )
        rng = jax.random.fold_in(jax.random.PRNGKey(rng_seed + int(num_steps)), step)
        log_for_0(
            "Eager preview sampling step %d / %d with num_steps=%d.",
            step,
            num_iters,
            num_steps,
        )
        with jax.disable_jit():
            latent = generate(
                variable,
                model,
                rng,
                batch_num_samples,
                config,
                num_steps,
                float(omega),
                float(t_min),
                float(t_max),
                sample_idx=None,
            )

        latent = latent.transpose(0, 3, 1, 2)
        samples = latent_manager.decode(latent)
        samples = samples[:batch_num_samples]
        samples = samples.transpose(0, 2, 3, 1)
        samples = 127.5 * samples + 128.0
        samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)
        samples_all.append(np.asarray(jax.device_get(samples)))

    samples_all = np.concatenate(samples_all, axis=0)
    return samples_all[:num_samples]
