"""JAX training loop for pixel-space JiT MeanFlow-Transfer (MeFT).

This mirrors the VAE-latent MeFT loop in ``train.py`` (the DiT/SiT MeFT path),
but swaps the VAE latent data pipeline for the pixel-space image pipeline used
by ``train_jit.py``. The training objective is the iMeanFlow Decoupled MeanFlow
(DMF) forward/loss, driving the ``imfJiT_DMF`` backbone in ``models/jit.py``.

Key differences vs. ``train.py``:
  * No VAE. Images are RGB pixels in [-1, 1]; there is no ``LatentManager`` /
    ``cached_encode`` step. ``PixelImageManager`` is a thin decode/layout shim.
  * The model is initialized with 3 input channels (RGB) instead of 4 (VAE).
  * ``config.dataset.image_channels`` drives ``generate`` sample shapes.

It is intentionally scoped to the plain no-DogFit MeFT configuration (the same
one the SiT MeFT runs use): ``use_auxiliary_v_head=False``, no guidance
conditioning, ``source_params=None``.
"""

import dataclasses
import os
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
from flax import jax_utils
from jax import lax, random

import utils.input_pipeline as input_pipeline
from imf import iMeanFlow, generate
from utils.ckpt_util import (
    restore_checkpoint,
    restore_eval_checkpoint,
    restore_partial_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from utils.ema_util import ema_schedules, update_ema
from utils.eval_csv_util import append_eval_metrics_row
from utils.logging_util import MetricsTracker, Timer, Writer, log_for_0
from utils.lr_utils import lr_schedules
from utils.preview_util import (
    format_preview_guidance_label,
    generate_preview_samples_first_device,
    make_uint8_image_grid,
    make_side_by_side_preview_panel,
    make_stacked_grid_panel,
)
from utils.sample_util import (
    get_image_metric_evaluator,
    get_sample_device_batch_size,
    get_sample_devices,
    get_sample_local_device_count,
    get_sampling_param_dtype,
)
from utils.trainstate_util import EvalState, TrainState, create_train_state


def compute_metrics(dict_losses):
    metrics = {k: jnp.mean(v) for k, v in dict_losses.items()}
    return lax.pmean(metrics, axis_name="batch")


def infer_num_classes_from_images(dataset_root):
    train_root = os.path.join(dataset_root, "train")
    if not os.path.isdir(train_root):
        raise ValueError(f"Image train directory not found: {train_root}")
    class_dirs = [
        name
        for name in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, name))
    ]
    if not class_dirs:
        raise ValueError(f"No class folders found under: {train_root}")
    return len(class_dirs)


class PixelImageManager:
    """Decode helper for pixel-space samples.

    Shared metric/preview utilities expect a manager whose ``decode`` returns
    NCHW images in [-1, 1]. iMeanFlow ``generate`` already produces BHWC pixels,
    so ``decode`` is only a layout conversion (BHWC -> BCHW).
    """

    def __init__(self, batch_size, decode_num_local_devices=None):
        self.batch_size = int(batch_size)
        self.decode_num_local_devices = (
            jax.local_device_count()
            if decode_num_local_devices is None
            else int(decode_num_local_devices)
        )

    def decode(self, images_bhwc):
        images_bhwc = jnp.clip(images_bhwc, -1.0, 1.0)
        return jnp.transpose(images_bhwc, (0, 3, 1, 2))


def _build_model(config, eval_mode=False):
    model_config = config.model.to_dict()
    valid_model_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model_config = {
        key: value for key, value in model_config.items() if key in valid_model_keys
    }
    if eval_mode:
        return iMeanFlow(**model_config, eval=True)
    return iMeanFlow(**model_config)


def train_step(
    state,
    batch,
    rng_init,
    ema_fn,
    lr_fn,
    use_ema,
    grad_accum_steps,
):
    """Single pixel-space iMeanFlow train step (plain, no DogFit)."""
    rng_step = random.fold_in(rng_init, state.step)
    rng_base = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))

    images = batch["image"]  # [B, H, W, C] pixels in [-1, 1]
    labels = batch["label"]

    def loss_fn(params):
        return state.apply_fn(
            {"params": params},
            images=images,
            labels=labels,
            source_params=state.source_params,
            teacher_params=state.ema_params,
            current_step=state.step,
            rngs=dict(gen=rng_base),
        )

    aux, grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    lr_value = lr_fn(state.step)
    metrics = compute_metrics(aux[1])
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
            new_ema = update_ema(
                updated_state.ema_params, updated_state.params, ema_value
            )
            updated_state = updated_state.replace(ema_params=new_ema)
        reset_grad_accum = jax.tree_util.tree_map(jnp.zeros_like, accum_grads)
        return updated_state.replace(
            grad_accum=reset_grad_accum,
            grad_accum_step=jnp.array(0, dtype=jnp.int32),
        )

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


def sample_step(variable, sample_idx, model, rng_init, device_batch_size,
                config, num_steps, omega, t_min, t_max):
    rng_sample = random.fold_in(rng_init, sample_idx)
    # Return BHWC pixels. PixelImageManager.decode() (and get_image_metric_evaluator
    # when configured with a PixelImageManager) expects the BHWC pixel layout, the
    # same convention used by the plain pixel-space JiT path in train_jit.py. Unlike
    # the VAE-latent path (train.py), we do NOT transpose to BCHW here.
    images = generate(variable, model, rng_sample, device_batch_size,
                      config, num_steps, omega, t_min, t_max, sample_idx=sample_idx)
    return images


def _set_num_classes_from_data(config):
    if config.dataset.get("num_classes_from_data", False):
        inferred = infer_num_classes_from_images(config.dataset.root)
        config.dataset.num_classes = inferred
        config.model.num_classes = inferred
        config.sampling.num_classes = inferred
        log_for_0("Inferred dataset.num_classes from image data: %s", inferred)


def _should_run_fid(current_step, training_config):
    force_fid_per_step = int(training_config.get("force_fid_per_step", 0) or 0)
    if force_fid_per_step > 0:
        return current_step % force_fid_per_step == 0
    fid_per_step = int(training_config.get("fid_per_step", 0))
    return fid_per_step > 0 and current_step % fid_per_step == 0


def _parse_int_steps(value, *, fallback=()):
    if value is None or value == "":
        return tuple(fallback)
    if isinstance(value, str):
        return tuple(int(step) for step in value.replace(",", " ").split())
    if isinstance(value, (int, float)):
        return (int(value),)
    return tuple(int(step) for step in value)


def _write_eval_metrics_csv(workdir, **row):
    append_eval_metrics_row(workdir, row)


def _metrics_enabled(training_config):
    return (
        int(training_config.get("fid_per_step", 0)) > 0
        or int(training_config.get("force_fid_per_step", 0) or 0) > 0
    )


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    writer = Writer(config, workdir)
    if config.eval_only:
        # eval_only is handled by just_evaluate(); main_imf_jit routes there.
        return just_evaluate(config, workdir)
    _set_num_classes_from_data(config)

    rng = random.key(config.training.seed)
    image_size = int(config.dataset.image_size)
    image_channels = int(config.dataset.image_channels)
    use_ema = config.training.get("use_ema", True)
    max_train_steps = config.training.get("max_train_steps", None)
    grad_accum_steps = int(config.training.get("grad_accum_steps", 1))
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)

    log_for_0("config.training.batch_size: %s", config.training.batch_size)
    log_for_0("config.training.use_ema: %s", use_ema)
    log_for_0("config.training.max_train_steps: %s", max_train_steps)
    log_for_0("config.training.grad_accum_steps: %s", grad_accum_steps)
    log_for_0("image_channels (pixel space): %s", image_channels)

    local_batch_size = int(config.training.batch_size) // jax.process_count()
    if local_batch_size % jax.local_device_count() != 0:
        raise ValueError(
            "config.training.batch_size must make the per-host batch divisible "
            f"by local devices: local batch {local_batch_size}, local devices "
            f"{jax.local_device_count()}."
        )

    train_loader, steps_per_epoch = input_pipeline.create_image_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    log_for_0("Steps per Epoch: %s", steps_per_epoch)

    model = _build_model(config, eval_mode=False)
    lr_fn = lr_schedules(config, steps_per_epoch)
    ema_fn = ema_schedules(config)
    state = create_train_state(
        rng, config, model, image_size, lr_fn, input_channels=image_channels
    )

    if config.load_from != "":
        if config.get("partial_load", False):
            state = restore_partial_checkpoint(
                state,
                config.load_from,
                target_model_config=config.model,
            )
        else:
            state = restore_checkpoint(state, config.load_from)

    step = int(state.step)
    epoch_offset = step // steps_per_epoch
    state = jax_utils.replicate(state)

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            ema_fn=ema_fn,
            lr_fn=lr_fn,
            use_ema=use_ema,
            grad_accum_steps=grad_accum_steps,
        ),
        axis_name="batch",
        donate_argnums=(0,),
    )

    sample_model = _build_model(config, eval_mode=True)

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=sample_model,
                rng_init=random.PRNGKey(99),
                device_batch_size=sample_device_bsz,
                config=config,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    pixel_manager = PixelImageManager(
        sample_device_bsz,
        decode_num_local_devices=sample_local_device_count,
    )

    preview_num_steps = _parse_int_steps(
        config.training.get("preview_num_steps", ()),
        fallback=(int(config.sampling.num_steps),),
    )
    p_preview_sample_steps = {
        num_steps: build_p_sample_step(num_steps) for num_steps in preview_num_steps
    }
    preview_guidance_scales = config.training.get("preview_guidance_scales", [])
    preview_guidance_scales = (
        [float(omega) for omega in preview_guidance_scales]
        if preview_guidance_scales
        else [float(config.sampling.get("omega", 1.0))]
    )

    def log_preview_samples(state_for_logging, step_for_logging):
        num_images = int(config.fid.num_images_to_log)
        grid_size = int(num_images ** 0.5)
        num_images = grid_size ** 2
        if num_images <= 0:
            return
        preview_image_groups = {}
        for cfg_scale in preview_guidance_scales:
            preview_kwargs = jax_utils.replicate(
                {
                    "omega": cfg_scale,
                    "t_min": float(config.sampling.get("t_min", 0.0)),
                    "t_max": float(config.sampling.get("t_max", 1.0)),
                },
                devices=sample_devices,
            )
            preview_images = {}
            for num_steps, p_preview_step in p_preview_sample_steps.items():
                preview_images[num_steps] = generate_preview_samples_first_device(
                    state_for_logging,
                    p_preview_step,
                    pixel_manager,
                    use_ema,
                    num_samples=num_images,
                    param_dtype=get_sampling_param_dtype(config),
                    sample_local_device_count=sample_local_device_count,
                    **preview_kwargs,
                )
            preview_panel = make_side_by_side_preview_panel(preview_images, grid_size)
            preview_image_groups[
                format_preview_guidance_label(
                    cfg_scale,
                    float(config.sampling.get("t_min", 0.0)),
                    float(config.sampling.get("t_max", 1.0)),
                )
            ] = [preview_panel]
        writer.write_images(
            step_for_logging,
            {"image_grid": make_stacked_grid_panel(preview_image_groups, 1)},
        )

    image_metric_evaluator = (
        get_image_metric_evaluator(config, writer, pixel_manager)
        if _metrics_enabled(config.training)
        else None
    )
    best_fid = float("inf")
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )

    metrics_tracker = MetricsTracker()
    timer = Timer()
    should_stop = False
    log_for_0("Initial compilation, this might take some minutes...")

    initial_step = int(jax.device_get(state.step)[0])
    if (
        initial_step == 0
        and config.training.sample_per_step > 0
        and config.training.get("preview_at_step0", False)
    ):
        log_preview_samples(state, 0)

    sample_kwargs = jax_utils.replicate(
        {
            "omega": float(config.sampling.get("omega", 1.0)),
            "t_min": float(config.sampling.get("t_min", 0.0)),
            "t_max": float(config.sampling.get("t_max", 1.0)),
        },
        devices=sample_devices,
    )

    p_metric_sample_step = build_p_sample_step(int(config.sampling.num_steps))

    for epoch in range(epoch_offset, config.training.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch %s...", epoch)
        timer.reset()
        for n_batch, batch in enumerate(train_loader):
            batch = input_pipeline.prepare_batch_data(batch, batch_size=local_batch_size)
            state, metrics = p_train_step(state, batch)
            current_step = int(jax.device_get(state.step)[0])
            did_update = bool(jax.device_get(metrics["did_update"])[0])

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                log_for_0("p_train_step compiled in %.2fs", timer.elapse_with_reset())

            metrics_tracker.update(metrics)
            should_log = did_update and current_step > 0 and (
                current_step == 1 or current_step % config.training.log_per_step == 0
            )
            if should_log:
                summary = metrics_tracker.finalize()
                logged_steps = 1 if current_step == 1 else config.training.log_per_step
                summary["steps_per_second"] = logged_steps / timer.elapse_with_reset()
                summary.pop("did_update", None)
                writer.write_scalars(current_step, summary)
                log_for_0(
                    "step %d | loss %.5f",
                    current_step,
                    float(summary.get("loss", float("nan"))),
                )

            if (
                did_update
                and config.training.sample_per_step > 0
                and current_step > 0
                and current_step % config.training.sample_per_step == 0
            ):
                log_preview_samples(state, current_step)

            if (
                image_metric_evaluator is not None
                and did_update
                and current_step > 0
                and _should_run_fid(current_step, config.training)
            ):
                result = image_metric_evaluator(
                    state,
                    p_metric_sample_step,
                    current_step - 1,
                    ema_only=use_ema,
                    **sample_kwargs,
                )
                fid = float(result["fid"])
                is_best_fid = fid < best_fid
                if is_best_fid:
                    best_fid = fid
                    save_best_checkpoint(state, best_fid_ckpt_dir)
                _write_eval_metrics_csv(
                    workdir,
                    eval_phase="train",
                    training_step=current_step,
                    sampling_num_steps=int(config.sampling.num_steps),
                    omega=float(config.sampling.get("omega", 1.0)),
                    t_min=float(config.sampling.get("t_min", 0.0)),
                    t_max=float(config.sampling.get("t_max", 1.0)),
                    fid=fid,
                    inception_score=float(result["is"]),
                    fd_dino=(
                        "" if result.get("fd_dino") is None
                        else float(result["fd_dino"])
                    ),
                    is_best_fid=int(is_best_fid),
                )

            if max_train_steps is not None and current_step >= max_train_steps:
                should_stop = True
                break

        if (
            not config.training.get("save_best_fid_only", False)
            and (
                should_stop
                or (epoch + 1) % config.training.checkpoint_per_epoch == 0
                or (epoch + 1) == config.training.num_epochs
            )
        ):
            save_checkpoint(state, workdir)

        if should_stop:
            log_for_0("Reached max_train_steps=%d at step %d.", max_train_steps, current_step)
            break

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state


########################################################
#                 Evaluation (eval_only)               #
########################################################


def _get_eval_sampling_configs(config):
    """Scalar CFG only (JiT DMF is not guidance-controllable): (omega, t_min, t_max)."""
    sampling = config.sampling
    omega = sampling.get("omega", None)
    t_min = sampling.get("t_min", None)
    t_max = sampling.get("t_max", None)
    if omega is not None and t_min is not None and t_max is not None:
        return [(float(omega), float(t_min), float(t_max))]
    raise ValueError(
        "eval_only requires sampling.omega + sampling.t_min + sampling.t_max in the config."
    )


def _get_metric_num_steps(config):
    """NFE list to evaluate: force_metric_num_steps overrides; else metric_num_steps;
    else the training sampling.num_steps. Primary step is always included first."""
    forced = str(config.training.get("force_metric_num_steps", "") or "").strip()
    if forced:
        steps = [int(s) for s in forced.replace(",", " ").split()]
    else:
        configured = config.training.get("metric_num_steps", ())
        steps = [int(s) for s in configured] if configured else [int(config.sampling.num_steps)]
    primary = int(config.sampling.num_steps)
    ordered = []
    for s in [primary] + steps:
        if s < 1:
            raise ValueError("Metric sampling steps must be >= 1.")
        if s not in ordered:
            ordered.append(s)
    return tuple(ordered)


def _primary_metric_mode(use_ema):
    return "ema" if use_ema else "online"


def just_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Post-hoc multi-NFE FID/FDD/IS eval for a saved JiT DMF checkpoint.

    Mirrors train.py:just_evaluate on the pixel-space JiT path. Restores a
    lightweight EvalState from config.load_from (a best_fid/checkpoint_* dir),
    then evaluates every NFE in _get_metric_num_steps and writes eval_only rows.
    """
    assert config.eval_only, "config.eval_only must be True for just_evaluate"
    assert config.load_from != "", "config.load_from must be specified for just_evaluate"

    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)

    image_size = int(config.dataset.image_size)
    sample_device_bsz = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_devices = get_sample_devices(config)
    # These runs train with use_ema=False -> evaluate online params.
    use_ema = config.training.get("use_ema", True)
    metric_mode = _primary_metric_mode(use_ema)

    model = _build_model(config, eval_mode=True)

    state = restore_eval_checkpoint(config.load_from, use_ema=use_ema)
    step = int(state.step)
    state = jax_utils.replicate(state)

    pixel_manager = PixelImageManager(
        sample_device_bsz,
        decode_num_local_devices=sample_local_device_count,
    )

    def build_p_sample_step(num_steps):
        return jax.pmap(
            partial(
                sample_step,
                model=model,
                rng_init=random.PRNGKey(99),
                device_batch_size=sample_device_bsz,
                config=config,
                num_steps=num_steps,
            ),
            axis_name="batch",
            devices=sample_devices,
        )

    image_metric_evaluator = get_image_metric_evaluator(config, writer, pixel_manager)
    metric_num_steps = _get_metric_num_steps(config)
    p_metric_sample_steps = {
        n: build_p_sample_step(n) for n in metric_num_steps
    }

    best_fid = float("inf")
    best_config = None
    best_fd_dino = float("inf")
    best_fd_dino_config = None
    csv_rows = []
    for num_steps, p_sample_step in p_metric_sample_steps.items():
        for omega, t_min, t_max in _get_eval_sampling_configs(config):
            kwargs = jax_utils.replicate(
                {"omega": omega, "t_min": t_min, "t_max": t_max},
                devices=sample_devices,
            )
            result = image_metric_evaluator(
                state,
                p_sample_step,
                step,
                ema_only=use_ema,
                metric_suffix=f"steps_{num_steps}",
                **kwargs,
            )
            fid = float(result["fid"])
            is_score = float(result["is"])
            fd_dino = result.get("fd_dino", None)
            row = dict(sampling_num_steps=num_steps, omega=omega, t_min=t_min,
                       t_max=t_max, fid=fid, is_score=is_score, fd_dino=fd_dino)
            csv_rows.append(row)
            cfg_key = (num_steps, omega, t_min, t_max)
            if fid < best_fid:
                best_fid = fid
                best_config = cfg_key
            if fd_dino is not None and fd_dino < best_fd_dino:
                best_fd_dino = fd_dino
                best_fd_dino_config = cfg_key
            log_for_0("eval_only NFE=%d omega=%.2f -> FID=%.4f IS=%.4f FDD=%s",
                      num_steps, omega, fid, is_score,
                      "None" if fd_dino is None else f"{float(fd_dino):.4f}")

            # --- save a preview grid for this NFE (mirrors DiT eval) ---
            num_preview_images = int(config.fid.get("num_images_to_log", 16))
            preview_grid_size = int(num_preview_images ** 0.5)
            num_preview_images = preview_grid_size ** 2
            if num_preview_images > 0:
                preview = generate_preview_samples_first_device(
                    state,
                    p_sample_step,
                    pixel_manager,
                    use_ema,
                    num_samples=num_preview_images,
                    param_dtype=get_sampling_param_dtype(config),
                    sample_local_device_count=sample_local_device_count,
                    **kwargs,
                )
                writer.write_images(
                    step,
                    {f"image_grid_steps_{num_steps}": make_uint8_image_grid(preview, preview_grid_size)},
                )

    for row in csv_rows:
        cfg_key = (row["sampling_num_steps"], row["omega"], row["t_min"], row["t_max"])
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
            inception_score=float(row["is_score"]),
            fd_dino="" if row["fd_dino"] is None else float(row["fd_dino"]),
            is_best_fid=int(cfg_key == best_config),
            is_best_fd_dino=int(best_fd_dino_config is not None and cfg_key == best_fd_dino_config),
            checkpoint_path=os.path.abspath(config.load_from),
        )

    log_for_0("eval_only DONE. best FID=%.4f at %s", best_fid, str(best_config))
    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
