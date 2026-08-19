"""Discrete AFM post-training for a target-finetuned improved MeanFlow model."""

import dataclasses
import json
import os
from functools import partial

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from flax import jax_utils, serialization, struct
from flax.training import checkpoints
from jax import lax, random

import utils.input_pipeline as input_pipeline
from afm import (
    augment_latents,
    cosine_decay_weight,
    decayed_anchor_weight,
    discriminator_loss,
    finite_difference_penalty,
    generated_lower_endpoint,
    generator_loss,
    is_discriminator_step,
    linear_path,
    sample_time_pairs,
)
from imf import iMeanFlow
from models import imfDiT
from models.afm_discriminator import create_afm_discriminator
from train_caimf import (
    _cached_encode,
    _copy_device_tree,
    _copy_matching_generator_params,
    _create_sampling_model,
    _get_metric_num_steps,
    _metric_evaluation_enabled,
    _sample_step,
    _set_num_classes_from_data,
    _should_run_fid,
)
from utils.ckpt_util import (
    restore_checkpoint,
    restore_partial_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
)
from utils.ema_util import update_ema
from utils.eval_csv_util import append_eval_metrics_row
from utils.logging_util import Timer, Writer, log_for_0
from utils.sample_util import (
    get_image_metric_evaluator,
    get_sample_device_batch_size,
    get_sample_devices,
    get_sample_local_device_count,
    has_controllable_sampling_guidance,
    get_sampling_param_dtype,
)
from utils.preview_util import (
    generate_preview_samples_first_device,
    make_side_by_side_preview_panel,
)
from utils.trainstate_util import EvalState, create_train_state
from utils.vae_util import LatentManager


@struct.dataclass
class AFMTrainState:
    """Fully resumable two-optimizer AFM state."""

    step: object
    epoch: object
    gen_step: object
    dis_step: object
    real_images_seen: object
    params: object
    ema_params: object
    gen_opt_state: object
    dis_params: object
    dis_opt_state: object
    source_params: object
    rng: object


def _mean_metrics(metrics, distributed):
    return lax.pmean(metrics, "batch") if distributed else metrics


def _get_images_and_labels(batch, rng_vae, distributed):
    images = batch["image"] if distributed else batch["image"][0]
    labels = batch["label"] if distributed else batch["label"][0]
    return _cached_encode(images, rng_vae), labels


def _conditioning_values(config, batch_size, dtype):
    return (
        jnp.full((batch_size,), float(config.sampling.omega), dtype),
        jnp.full((batch_size,), float(config.sampling.t_min), dtype),
        jnp.full((batch_size,), float(config.sampling.t_max), dtype),
    )


def _predict_u(model, params, x_t, r, t, labels, config):
    omega, t_min, t_max = _conditioning_values(config, x_t.shape[0], x_t.dtype)
    return model.apply(
        {"params": params},
        x_t,
        r,
        t,
        omega,
        t_min,
        t_max,
        labels,
        method=model.afm_u_fn,
    )


def _optional_imf_loss(lambda_imf, compute_loss, dtype=jnp.float32):
    """Keep the expensive iMF/JVP branch completely absent when disabled."""
    if float(lambda_imf) == 0.0:
        return jnp.asarray(0.0, dtype)
    return compute_loss()


def validate_target_labels(labels, num_classes):
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("Target-label batch is empty.")
    if labels.min() < 0 or labels.max() >= int(num_classes):
        raise ValueError(
            f"Target labels outside [0,{int(num_classes)-1}]: "
            f"min={labels.min()} max={labels.max()}"
        )


def _make_endpoint_terms(model, params, images, labels, rng, config):
    rng_time, rng_noise = random.split(rng)
    r, t, interval, zero_mask = sample_time_pairs(
        rng_time,
        images.shape[0],
        min_interval=float(config.afm.min_interval),
        p_r_zero=float(config.afm.p_r_zero),
    )
    x1 = random.normal(rng_noise, images.shape, dtype=images.dtype)
    x_t = linear_path(images, x1, t)
    x_r = linear_path(images, x1, r)
    u = _predict_u(model, params, x_t, r, t, labels, config)
    x_r_fake = generated_lower_endpoint(x_t, u, r, t)
    return {
        "x_t": x_t,
        "x_r": x_r,
        "x_r_fake": x_r_fake,
        "u": u,
        "r": r,
        "t": t,
        "interval": interval,
        "zero_mask": zero_mask,
        "labels": labels,
    }


def _mask_discriminator_grads(
    grads, discriminator, trainable_blocks, freeze_backbone=False
):
    """Apply full, partial-block, or strict norm-and-head-only D training."""
    if trainable_blocks < 0 and not freeze_backbone:
        return grads
    state = serialization.to_state_dict(grads)
    shared_depth = discriminator.depth - discriminator.aux_head_depth
    first_trainable = max(discriminator.depth - trainable_blocks, 0)

    def transformer_index(name):
        for prefix, offset in (("shared_blocks_", 0), ("u_heads_", shared_depth)):
            if name.startswith(prefix):
                try:
                    return offset + int(name[len(prefix) :])
                except ValueError:
                    return None
        return None

    def mask(tree, path=()):
        if isinstance(tree, dict):
            return {key: mask(value, path + (key,)) for key, value in tree.items()}
        top = path[0] if path else ""
        if freeze_backbone:
            train = top in {"dis_norm", "dis_head"}
            return tree if train else jnp.zeros_like(tree)
        block_index = transformer_index(top)
        always_train = top in {
            "dis_norm",
            "dis_head",
            "h_embedder",
            "y_embedder",
            "time_tokens",
            "class_tokens",
        }
        train = always_train or (
            block_index is not None and block_index >= first_trainable
        )
        return tree if train else jnp.zeros_like(tree)

    return serialization.from_state_dict(grads, mask(state))


def discriminator_train_step(
    state,
    batch,
    *,
    model,
    discriminator,
    dis_tx,
    config,
    global_batch_size,
    distributed,
):
    step_rng, next_rng = random.split(state.rng)
    if distributed:
        step_rng = random.fold_in(step_rng, lax.axis_index("batch"))
    rng_vae, rng_endpoint, rng_real_aug, rng_fake_aug, rng_r1, rng_r2 = random.split(
        step_rng, 6
    )
    images, labels = _get_images_and_labels(batch, rng_vae, distributed)
    terms = _make_endpoint_terms(
        model, state.params, images, labels, rng_endpoint, config
    )
    terms = jax.tree_util.tree_map(lax.stop_gradient, terms)

    augmentation_probability = (
        float(config.afm.discriminator_augmentation_probability)
        if bool(config.afm.use_discriminator_augmentation)
        else 0.0
    )
    x_real = augment_latents(terms["x_r"], rng_real_aug, augmentation_probability)
    x_fake = augment_latents(
        terms["x_r_fake"], rng_fake_aug, augmentation_probability
    )

    def loss_fn(dis_params):
        d_real = discriminator.apply(
            {"params": dis_params}, x_real, terms["r"], labels
        )
        d_fake = discriminator.apply(
            {"params": dis_params}, x_fake, terms["r"], labels
        )
        if float(config.afm.lambda_gp) > 0.0:
            r1 = finite_difference_penalty(
                discriminator,
                dis_params,
                x_real,
                terms["r"],
                labels,
                jnp.maximum(terms["interval"], float(config.afm.interval_eps)),
                rng_r1,
                fd_epsilon=float(config.afm.fd_epsilon),
                batch_fraction=float(config.afm.gp_batch_fraction),
            )
            r2 = finite_difference_penalty(
                discriminator,
                dis_params,
                x_fake,
                terms["r"],
                labels,
                jnp.maximum(terms["interval"], float(config.afm.interval_eps)),
                rng_r2,
                fd_epsilon=float(config.afm.fd_epsilon),
                batch_fraction=float(config.afm.gp_batch_fraction),
            )
        else:
            r1 = jnp.asarray(0.0, images.dtype)
            r2 = jnp.asarray(0.0, images.dtype)
        return discriminator_loss(
            d_real,
            d_fake,
            r1,
            r2,
            lambda_gp=float(config.afm.lambda_gp),
            lambda_cp=float(config.afm.lambda_cp),
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.dis_params)
    if distributed:
        grads = lax.pmean(grads, "batch")
    grads = _mask_discriminator_grads(
        grads,
        discriminator,
        int(config.afm.discriminator_trainable_blocks),
        freeze_backbone=bool(config.afm.freeze_discriminator_backbone),
    )
    grad_norm = optax.global_norm(grads)
    updates, dis_opt_state = dis_tx.update(
        grads, state.dis_opt_state, state.dis_params
    )
    dis_params = optax.apply_updates(state.dis_params, updates)
    state = state.replace(
        step=state.step + 1,
        dis_step=state.dis_step + 1,
        real_images_seen=state.real_images_seen + global_batch_size,
        dis_params=dis_params,
        dis_opt_state=dis_opt_state,
        rng=next_rng,
    )
    metrics.update(
        {
            "phase_discriminator": jnp.asarray(1.0),
            "generator_grad_norm": jnp.asarray(0.0),
            "discriminator_grad_norm": grad_norm,
            "mean_interval": jnp.mean(terms["interval"]),
            "minimum_interval": jnp.min(terms["interval"]),
            "maximum_interval": jnp.max(terms["interval"]),
            "fraction_r_zero": jnp.mean(terms["zero_mask"].astype(jnp.float32)),
        }
    )
    return state, _finish_metrics(state, metrics, config, distributed)


def generator_train_step(
    state,
    batch,
    *,
    model,
    discriminator,
    gen_tx,
    config,
    global_batch_size,
    distributed,
):
    step_rng, next_rng = random.split(state.rng)
    if distributed:
        step_rng = random.fold_in(step_rng, lax.axis_index("batch"))
    rng_vae, rng_endpoint, rng_imf = random.split(step_rng, 3)
    images, labels = _get_images_and_labels(batch, rng_vae, distributed)
    needs_endpoint = any(
        float(value) > 0.0
        for value in (
            config.afm.lambda_adv,
            config.afm.lambda_ot,
            config.afm.lambda_anchor,
        )
    )

    def loss_fn(gen_params):
        if needs_endpoint:
            terms = _make_endpoint_terms(
                model, gen_params, images, labels, rng_endpoint, config
            )
        else:
            batch_size = images.shape[0]
            terms = {
                "x_t": images,
                "x_r": images,
                "x_r_fake": images,
                "u": jnp.zeros_like(images),
                "r": jnp.zeros((batch_size,), images.dtype),
                "t": jnp.ones((batch_size,), images.dtype),
                "interval": jnp.ones((batch_size,), images.dtype),
                "zero_mask": jnp.ones((batch_size,), dtype=bool),
                "labels": labels,
            }

        if float(config.afm.lambda_adv) > 0.0:
            d_real = lax.stop_gradient(
                discriminator.apply(
                    {"params": state.dis_params}, terms["x_r"], terms["r"], labels
                )
            )
            d_fake = discriminator.apply(
                {"params": state.dis_params},
                terms["x_r_fake"],
                terms["r"],
                labels,
            )
        else:
            d_real = jnp.zeros((images.shape[0],), images.dtype)
            d_fake = jnp.zeros((images.shape[0],), images.dtype)

        # This is a Python-static branch. With lambda_imf=0 the JVP function is
        # neither called nor traced into the AFM-only compiled computation.
        def compute_imf_loss():
            return model.apply(
                {"params": gen_params},
                images=images,
                labels=labels,
                source_params=None,
                teacher_params=state.ema_params,
                current_step=state.step,
                interval_eps=float(config.afm.interval_eps),
                rngs={"gen": rng_imf},
                method=model.forward_caimf_generator_terms,
            )["loss_imf"]

        loss_imf = _optional_imf_loss(
            config.afm.lambda_imf, compute_imf_loss, images.dtype
        )

        anchor_weight = decayed_anchor_weight(
            float(config.afm.lambda_anchor),
            state.gen_step,
            int(config.afm.anchor_decay_steps),
        )
        if float(config.afm.lambda_anchor) > 0.0:
            source_u = lax.stop_gradient(
                _predict_u(
                    model,
                    state.source_params,
                    terms["x_t"],
                    terms["r"],
                    terms["t"],
                    labels,
                    config,
                )
            )
            loss_anchor = jnp.mean((terms["u"] - source_u) ** 2)
        else:
            loss_anchor = jnp.asarray(0.0, images.dtype)

        lambda_ot = cosine_decay_weight(
            float(config.afm.lambda_ot),
            float(config.afm.get("lambda_ot_end", config.afm.lambda_ot)),
            state.step,
            int(config.afm.get("lambda_ot_decay_steps", 0)),
        )
        total, metrics = generator_loss(
            d_real,
            d_fake,
            terms["x_r_fake"],
            terms["x_t"],
            terms["interval"],
            lambda_adv=float(config.afm.lambda_adv),
            lambda_ot=lambda_ot,
            lambda_imf=float(config.afm.lambda_imf),
            loss_imf=loss_imf,
            lambda_anchor=anchor_weight,
            loss_anchor=loss_anchor,
            interval_eps=float(config.afm.interval_eps),
        )
        metrics["lambda_ot"] = lambda_ot
        return total, (metrics, terms)

    (loss_value, (metrics, terms)), grads = jax.value_and_grad(
        loss_fn, has_aux=True
    )(state.params)
    del loss_value
    if distributed:
        grads = lax.pmean(grads, "batch")
    grad_norm = optax.global_norm(grads)
    updates, gen_opt_state = gen_tx.update(grads, state.gen_opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema_params = state.ema_params
    if bool(config.training.use_ema):
        ema_params = update_ema(
            ema_params, params, float(config.training.ema_val)
        )
    state = state.replace(
        step=state.step + 1,
        gen_step=state.gen_step + 1,
        real_images_seen=state.real_images_seen + global_batch_size,
        params=params,
        ema_params=ema_params,
        gen_opt_state=gen_opt_state,
        rng=next_rng,
    )
    metrics.update(
        {
            "phase_discriminator": jnp.asarray(0.0),
            "generator_grad_norm": grad_norm,
            "discriminator_grad_norm": jnp.asarray(0.0),
            "mean_interval": jnp.mean(terms["interval"]),
            "minimum_interval": jnp.min(terms["interval"]),
            "maximum_interval": jnp.max(terms["interval"]),
            "fraction_r_zero": jnp.mean(terms["zero_mask"].astype(jnp.float32)),
        }
    )
    return state, _finish_metrics(state, metrics, config, distributed)


def _finish_metrics(state, metrics, config, distributed):
    generator_lr = float(config.afm.generator_learning_rate) * jnp.minimum(
        state.gen_step
        / float(max(int(config.afm.generator_lr_warmup_steps), 1)),
        1.0,
    )
    discriminator_lr = float(config.afm.discriminator_learning_rate) * jnp.minimum(
        state.dis_step
        / float(max(int(config.afm.discriminator_lr_warmup_steps), 1)),
        1.0,
    )
    metrics.update(
        {
            "batch_step": state.step.astype(jnp.float32),
            "generator_updates": state.gen_step.astype(jnp.float32),
            "discriminator_updates": state.dis_step.astype(jnp.float32),
            "real_images_seen": state.real_images_seen.astype(jnp.float32),
            "generator_learning_rate": generator_lr.astype(jnp.float32),
            "discriminator_learning_rate": discriminator_lr.astype(jnp.float32),
        }
    )
    return _mean_metrics(metrics, distributed)


def _create_models(config):
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model = iMeanFlow(
        **{key: value for key, value in model_config.items() if key in valid_keys}
    )
    if not str(config.model.model_str).startswith("imfDiT_"):
        raise ValueError("AFM currently supports model.model_str=imfDiT_* only.")
    net_fn = getattr(imfDiT, config.model.model_str)
    generator_net = net_fn(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        use_auxiliary_v_head=bool(config.model.use_auxiliary_v_head),
        eval=False,
    )
    depth = int(config.afm.discriminator_depth)
    discriminator = create_afm_discriminator(
        generator_net,
        width_multiplier=float(config.afm.discriminator_width_multiplier),
        depth=None if depth <= 0 else depth,
    )
    return model, generator_net, discriminator


def _optimizer(learning_rate, warmup_steps, beta1, beta2, weight_decay):
    # Optax 0.2.2 (the repository-pinned version) does not expose the newer
    # warmup_constant_schedule helper. Composing the two primitive schedules
    # is exactly the same policy and keeps this pipeline compatible with the
    # existing environment.
    warmup_steps = max(int(warmup_steps), 1)
    schedule = optax.join_schedules(
        schedules=(
            optax.linear_schedule(
                init_value=0.0,
                end_value=learning_rate,
                transition_steps=warmup_steps,
            ),
            optax.constant_schedule(learning_rate),
        ),
        boundaries=(warmup_steps,),
    )
    return optax.adamw(
        schedule,
        b1=beta1,
        b2=beta2,
        weight_decay=weight_decay,
    ), schedule


def _create_state(config, model, generator_net, discriminator, rng):
    afm_cfg = config.afm
    gen_tx, gen_lr_fn = _optimizer(
        float(afm_cfg.generator_learning_rate),
        int(afm_cfg.generator_lr_warmup_steps),
        float(afm_cfg.adam_beta1),
        float(afm_cfg.adam_beta2),
        float(afm_cfg.generator_weight_decay),
    )
    dis_tx, dis_lr_fn = _optimizer(
        float(afm_cfg.discriminator_learning_rate),
        int(afm_cfg.discriminator_lr_warmup_steps),
        float(afm_cfg.adam_beta1),
        float(afm_cfg.adam_beta2),
        float(afm_cfg.discriminator_weight_decay),
    )
    base_state = create_train_state(
        rng,
        config,
        model,
        int(config.dataset.image_size),
        lambda _: jnp.asarray(float(afm_cfg.generator_learning_rate)),
    )
    if not config.load_from:
        raise ValueError(
            "load_from/source_checkpoint must be the target-finetuned iMF checkpoint."
        )
    base_state = restore_partial_checkpoint(
        base_state,
        config.load_from,
        prefer_ema=bool(afm_cfg.load_generator_ema),
        target_model_config=config.model,
    )
    params = base_state.params
    ema_params = _copy_device_tree(params)
    source_params = (
        _copy_device_tree(params) if float(afm_cfg.lambda_anchor) > 0.0 else None
    )

    needs_discriminator = bool(afm_cfg.discriminator_updates) and float(
        afm_cfg.lambda_adv
    ) > 0.0
    if needs_discriminator:
        rng, rng_dis = random.split(rng)
        dummy_x = jnp.ones(
            (1, int(config.dataset.image_size), int(config.dataset.image_size), 4),
            jnp.float32,
        )
        dis_params = discriminator.init(
            {"params": rng_dis},
            dummy_x,
            jnp.full((1,), 0.5, jnp.float32),
            jnp.zeros((1,), jnp.int32),
        )["params"]
        if str(afm_cfg.discriminator_init).lower() == "generator":
            dis_params, copied = _copy_matching_generator_params(
                dis_params, params["net"]
            )
            log_for_0("Initialized AFM D from G: copied %d tensors.", copied)
        elif str(afm_cfg.discriminator_init).lower() != "random":
            raise ValueError("discriminator_init must be 'generator' or 'random'.")
        dis_opt_state = dis_tx.init(dis_params)
    else:
        dis_params = None
        dis_opt_state = None

    state = AFMTrainState(
        step=jnp.asarray(0, jnp.int32),
        epoch=jnp.asarray(0, jnp.int32),
        gen_step=jnp.asarray(0, jnp.int32),
        dis_step=jnp.asarray(0, jnp.int32),
        real_images_seen=jnp.asarray(0, jnp.int32),
        params=params,
        ema_params=ema_params,
        gen_opt_state=gen_tx.init(params),
        dis_params=dis_params,
        dis_opt_state=dis_opt_state,
        source_params=source_params,
        rng=rng,
    )
    resume_from = str(afm_cfg.get("resume_from", "") or "").strip()
    if resume_from:
        state = restore_checkpoint(state, resume_from)
        log_for_0(
            "Resumed AFM at batch=%d G=%d D=%d.",
            int(state.step),
            int(state.gen_step),
            int(state.dis_step),
        )
    return state, gen_tx, dis_tx, gen_lr_fn, dis_lr_fn


def _class_mapping(config):
    num_classes = int(config.dataset.num_classes)
    root = str(config.dataset.get("class_mapping_root", "") or "").strip()
    if root and os.path.isdir(root):
        names = sorted(
            entry
            for entry in os.listdir(root)
            if os.path.isdir(os.path.join(root, entry))
        )
        if len(names) != num_classes:
            raise ValueError(
                f"Found {len(names)} class folders under {root}, expected {num_classes}."
            )
        return {name: index for index, name in enumerate(names)}
    return {str(index): index for index in range(num_classes)}


def _metadata(config, mapping):
    return {
        "format": "imeanflow-afm-v1",
        "target_class_mapping": mapping,
        "afm_hyperparameters": config.afm.to_dict(),
        "source_checkpoint": os.path.abspath(str(config.load_from)),
        "target_data_root": os.path.abspath(str(config.dataset.root)),
        "target_num_classes": int(config.dataset.num_classes),
        "representation": "normalized SD-VAE latent, NHWC 32x32x4",
        "rng_reproduction": "state.rng plus state.step are stored in the Flax checkpoint",
    }


def _write_metadata(path, metadata):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "afm_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, sort_keys=True)


def _write_checkpoint_metadata(checkpoint_root, metadata):
    """Put the mapping/hyperparameters beside every checkpoint payload."""
    _write_metadata(checkpoint_root, metadata)
    if not os.path.isdir(checkpoint_root):
        return
    for entry in os.listdir(checkpoint_root):
        child = os.path.join(checkpoint_root, entry)
        if entry.startswith("checkpoint_") and os.path.isdir(child):
            _write_metadata(child, metadata)


def _step_value(state, field="step"):
    value = np.asarray(jax.device_get(getattr(state, field)))
    return int(value.reshape(-1)[0])


def _save_full_state(state, workdir, distributed, metadata):
    if distributed:
        save_checkpoint(state, workdir)
    else:
        # Single device: state has NO leading device axis. jax_utils.replicate
        # would allocate a 2nd full copy of the fat adversarial state (G + D +
        # both optimizers + EMA) on the GPU, on top of resident eval models ->
        # OOM at end-of-training. device_get pulls to host RAM (no 2nd GPU
        # copy); save directly. Mirrors train_caimf.save_state.
        host_state = jax.device_get(state)
        s_step = int(np.asarray(host_state.step).reshape(-1)[0])
        log_for_0("Saving full AFM state step %d (single-device).", s_step)
        checkpoints.save_checkpoint_multiprocess(
            os.path.abspath(workdir), host_state, s_step, keep=3
        )
        log_for_0("Full AFM state step %d saved.", s_step)
    step = _step_value(state)
    checkpoint_path = os.path.join(os.path.abspath(workdir), f"checkpoint_{step}")
    _write_metadata(checkpoint_path, metadata)
    _write_checkpoint_metadata(workdir, metadata)


def _validate_host_metrics(metrics, phase, config):
    for key, value in metrics.items():
        if not np.isfinite(value):
            raise FloatingPointError(f"Non-finite AFM metric {key}={value} in {phase} step.")
    grad_key = (
        "discriminator_grad_norm" if phase == "discriminator" else "generator_grad_norm"
    )
    if metrics.get(grad_key, 0.0) <= 0.0:
        raise FloatingPointError(f"Zero {grad_key} in {phase} step.")
    max_logit = float(config.afm.max_abs_discriminator_logit)
    if max(abs(metrics.get("d_real", 0.0)), abs(metrics.get("d_fake", 0.0))) > max_logit:
        raise FloatingPointError(
            f"Discriminator logit exceeded configured limit {max_logit}: "
            f"real={metrics.get('d_real')} fake={metrics.get('d_fake')}"
        )


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)
    if int(config.training.get("grad_accum_steps", 1)) != 1:
        raise ValueError("AFM currently requires training.grad_accum_steps=1.")
    if float(config.afm.lambda_imf) == 0.0 and str(config.afm.ablation) == "target_imf":
        raise ValueError("target_imf ablation requires lambda_imf > 0.")

    mapping = _class_mapping(config)
    metadata = _metadata(config, mapping)
    _write_metadata(workdir, metadata)

    local_batch_size = int(config.training.batch_size) // jax.process_count()
    train_loader, _ = input_pipeline.create_latent_split(
        config.dataset, local_batch_size, "train"
    )
    model, generator_net, discriminator = _create_models(config)
    # Legacy uint32 PRNG keys are serializable by the repository's Flax
    # checkpoint backend; typed key<fry> arrays are not.
    rng = random.PRNGKey(int(config.training.seed))
    state, gen_tx, dis_tx, _, _ = _create_state(
        config, model, generator_net, discriminator, rng
    )
    distributed = jax.local_device_count() > 1
    if distributed:
        state = jax_utils.replicate(state)

    common = dict(
        model=model,
        discriminator=discriminator,
        config=config,
        global_batch_size=int(config.training.batch_size),
        distributed=distributed,
    )
    if distributed:
        p_dis_step = jax.pmap(
            lambda s, b: discriminator_train_step(
                s, b, dis_tx=dis_tx, **common
            ),
            "batch",
            donate_argnums=(0,),
        )
        p_gen_step = jax.pmap(
            lambda s, b: generator_train_step(s, b, gen_tx=gen_tx, **common),
            "batch",
            donate_argnums=(0,),
        )
    else:
        p_dis_step = jax.jit(
            lambda s, b: discriminator_train_step(
                s, b, dis_tx=dis_tx, **common
            ),
            donate_argnums=(0,),
        )
        p_gen_step = jax.jit(
            lambda s, b: generator_train_step(s, b, gen_tx=gen_tx, **common),
            donate_argnums=(0,),
        )

    metric_enabled = _metric_evaluation_enabled(config.training)
    evaluator = None
    metric_steps = ()
    p_sample_steps = {}
    sample_kwargs = None
    latent_manager = None
    p_preview_sample_steps = {}
    preview_num_images = int(config.training.get("preview_num_images", 16))
    preview_grid_size = int(np.sqrt(max(preview_num_images, 1)))
    preview_num_images = preview_grid_size * preview_grid_size
    best_fid = {}
    best_fdd = {}
    best_dir = os.path.join(workdir, config.training.best_fid_checkpoint_dir)
    use_ema_metrics = bool(config.training.use_ema) and not bool(
        config.training.get("fid_use_online_only", False)
    )
    metric_mode = "ema" if use_ema_metrics else "online"
    if metric_enabled:
        sample_bsz = get_sample_device_batch_size(config)
        sample_devices = get_sample_devices(config)
        latent_manager = LatentManager(
            config.dataset.vae,
            sample_bsz,
            int(config.dataset.image_size),
            decode_num_local_devices=get_sample_local_device_count(config),
        )
        sampling_model = _create_sampling_model(config)

        def build_sample_step(num_steps):
            return jax.pmap(
                partial(
                    _sample_step,
                    model=sampling_model,
                    rng_init=random.PRNGKey(99),
                    device_batch_size=sample_bsz,
                    config=config,
                    num_steps=num_steps,
                ),
                "batch",
                devices=sample_devices,
            )

        metric_steps = _get_metric_num_steps(config)
        p_sample_steps = {step: build_sample_step(step) for step in metric_steps}
        preview_steps = tuple(
            int(step)
            for step in config.training.get("preview_num_steps", (1, 2, 4))
        )
        p_preview_sample_steps = {
            step: build_sample_step(step) for step in preview_steps if step >= 1
        }
        best_fid = {step: float("inf") for step in metric_steps}
        best_fdd = {step: float("inf") for step in metric_steps}
        controllable = has_controllable_sampling_guidance(config.model)
        sample_kwargs = jax_utils.replicate(
            {
                "omega": float(config.sampling.omega) if controllable else 1.0,
                "t_min": float(config.sampling.t_min),
                "t_max": float(config.sampling.t_max),
            },
            devices=sample_devices,
        )

    def write_preview(state_value, step_value):
        if not metric_enabled or not p_preview_sample_steps or preview_num_images <= 0:
            return
        preview_state = state_value if distributed else jax_utils.replicate(
            state_value, devices=get_sample_devices(config)
        )
        preview_images = {}
        for num_steps, p_preview_step in p_preview_sample_steps.items():
            preview_images[num_steps] = generate_preview_samples_first_device(
                preview_state,
                p_preview_step,
                latent_manager,
                ema=use_ema_metrics,
                num_samples=preview_num_images,
                param_dtype=get_sampling_param_dtype(config),
                sample_local_device_count=get_sample_local_device_count(config),
                **sample_kwargs,
            )
        writer.write_images(
            step_value,
            {"image_grid": make_side_by_side_preview_panel(
                preview_images, preview_grid_size
            )},
        )

    warmup = int(config.afm.discriminator_warmup_steps)
    d_steps = int(config.afm.d_steps_per_g_step)
    max_batches = int(config.afm.max_posttrain_batches)
    discriminator_updates = bool(config.afm.discriminator_updates) and float(
        config.afm.lambda_adv
    ) > 0.0
    log_every = int(config.training.log_per_step)
    timer = Timer()
    compiled = set()
    stop = False
    start_epoch = _step_value(state, "epoch")
    log_for_0(
        "AFM schedule: warmup=%d, %dD:1G, max_batches=%d; "
        "lambda_adv=%g lambda_imf=%g lambda_ot=%g->%g/%dsteps lambda_anchor=%g "
        "lambda_gp=%g lambda_cp=%g.",
        warmup,
        d_steps,
        max_batches,
        float(config.afm.lambda_adv),
        float(config.afm.lambda_imf),
        float(config.afm.lambda_ot),
        float(config.afm.get("lambda_ot_end", config.afm.lambda_ot)),
        int(config.afm.get("lambda_ot_decay_steps", 0)),
        float(config.afm.lambda_anchor),
        float(config.afm.lambda_gp),
        float(config.afm.lambda_cp),
    )

    if metric_enabled and int(config.training.get("sample_per_step", 0)) > 0:
        write_preview(state, 0)

    for epoch in range(start_epoch, int(config.training.num_epochs)):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        for raw_batch in train_loader:
            batch_step = _step_value(state)
            batch = input_pipeline.prepare_batch_data(raw_batch)
            validate_target_labels(batch["label"], config.dataset.num_classes)
            do_d = discriminator_updates and bool(
                np.asarray(is_discriminator_step(batch_step, warmup, d_steps))
            )
            phase = "discriminator" if do_d else "generator"
            state, metrics = (
                p_dis_step(state, batch) if do_d else p_gen_step(state, batch)
            )
            current_step = _step_value(state)

            sample_period = int(config.training.get("sample_per_step", 0))
            if (
                sample_period > 0
                and current_step > 0
                and current_step % sample_period == 0
            ):
                write_preview(state, current_step)
            if phase not in compiled:
                jax.tree_util.tree_leaves(metrics)[0].block_until_ready()
                log_for_0("Initial %s step compiled in %.2fs.", phase, timer.elapse_with_reset())
                compiled.add(phase)

            if current_step == 1 or current_step % log_every == 0:
                host_metrics = {
                    key: float(np.asarray(jax.device_get(value)).mean())
                    for key, value in metrics.items()
                }
                _validate_host_metrics(host_metrics, phase, config)
                host_metrics["batches_per_second"] = log_every / max(
                    timer.elapse_with_reset(), 1e-6
                )
                writer.write_scalars(current_step, host_metrics)

            if metric_enabled and current_step > 0 and _should_run_fid(
                current_step, config.training
            ):
                metric_state = EvalState(
                    step=state.step,
                    params=state.params,
                    ema_params=state.ema_params,
                )
                if not distributed:
                    metric_state = jax_utils.replicate(metric_state, devices=get_sample_devices(config))
                if evaluator is None:
                    evaluator = get_image_metric_evaluator(
                        config, writer, latent_manager
                    )
                for num_steps, p_sample in p_sample_steps.items():
                    result = evaluator(
                        metric_state,
                        p_sample,
                        current_step - 1,
                        ema_only=use_ema_metrics,
                        metric_suffix=f"steps_{num_steps}",
                        **sample_kwargs,
                    )
                    fid = float(result["fid"])
                    fdd = result.get("fd_dino")
                    is_best_fid = fid < best_fid[num_steps]
                    is_best_fdd = fdd is not None and float(fdd) < best_fdd[num_steps]
                    if is_best_fid:
                        best_fid[num_steps] = fid
                    if is_best_fdd:
                        best_fdd[num_steps] = float(fdd)
                    checkpoint_path = ""
                    if num_steps == int(config.sampling.num_steps) and is_best_fid:
                        checkpoint_state = state if distributed else jax_utils.replicate(state)
                        save_best_checkpoint(
                            checkpoint_state,
                            best_dir,
                            eval_state_only=bool(
                                config.training.get("save_best_fid_eval_state_only", True)
                            ),
                        )
                        _write_checkpoint_metadata(best_dir, metadata)
                        checkpoint_path = os.path.abspath(best_dir)
                    append_eval_metrics_row(
                        workdir,
                        {
                            "eval_phase": "train",
                            "metric_mode": metric_mode,
                            "training_step": current_step,
                            "sampling_num_steps": num_steps,
                            "omega": float(config.sampling.omega),
                            "t_min": float(config.sampling.t_min),
                            "t_max": float(config.sampling.t_max),
                            "fid": fid,
                            "inception_score": float(result["is"]),
                            "fd_dino": "" if fdd is None else float(fdd),
                            "is_best_fid": int(is_best_fid),
                            "is_best_fd_dino": int(is_best_fdd),
                            "checkpoint_path": checkpoint_path,
                        },
                    )

                # eval-path OOM fix (mirror train_caimf): on a single GPU the
                # jax_utils.replicate copies (metric_state, and checkpoint_state
                # on a best-FID save) of the fat adversarial state stay resident
                # and overflow the next dis_step. Free them before resuming.
                if not distributed:
                    to_free = [metric_state]
                    if "checkpoint_state" in locals():
                        to_free.append(checkpoint_state)
                    jax.tree_util.tree_map(
                        lambda x: x.delete() if hasattr(x, "delete") else None,
                        tuple(to_free),
                    )
                    del metric_state
                    if "checkpoint_state" in locals():
                        del checkpoint_state

            if current_step >= max_batches:
                stop = True
                break

        state = state.replace(epoch=state.epoch + 1)
        if stop or (epoch + 1) % int(config.training.checkpoint_per_epoch) == 0:
            _save_full_state(state, workdir, distributed, metadata)
        if stop:
            break

    export_state = EvalState(
        step=state.step,
        params=state.params,
        ema_params=state.ema_params,
    )
    if not distributed:
        export_state = jax_utils.replicate(export_state)
    export_dir = os.path.join(workdir, "generator_only")
    save_best_checkpoint(export_state, export_dir, eval_state_only=True)
    _write_checkpoint_metadata(export_dir, metadata)
    log_for_0("AFM finished at batch=%d G=%d D=%d.", _step_value(state), _step_value(state, "gen_step"), _step_value(state, "dis_step"))
    return state
