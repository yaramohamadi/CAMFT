"""Post-train an improved MeanFlow generator with finite-interval adversarial loss."""

import dataclasses
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

from caimf import (
    discriminator_loss,
    finite_fake_logit,
    finite_interval_logits,
    generator_loss,
    is_discriminator_step,
)
from imf import generate, iMeanFlow
from models import imfDiT
from models.caimf_discriminator import create_caimf_discriminator
import utils.input_pipeline as input_pipeline
from utils.ckpt_util import (
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
class CAIMFTrainState:
    """Two-optimizer state whose public params remain evaluation-compatible."""

    step: object
    gen_step: object
    dis_step: object
    params: object
    ema_params: object
    gen_opt_state: object
    dis_params: object
    dis_opt_state: object


def _cached_encode(cached_value, rng):
    """Sample normalized SD-VAE latents without constructing a VAE decoder."""
    mean, std = jnp.split(cached_value, 2, axis=-1)
    latent = mean + std * random.normal(rng, mean.shape)
    channel_mean = jnp.asarray([0.86488, -0.27787343, 0.21616915, 0.3738409])
    channel_std = jnp.asarray([4.85503674, 5.31922414, 3.93725398, 3.9870003])
    return (latent - channel_mean.reshape(1, 1, 1, 4)) / channel_std.reshape(
        1, 1, 1, 4
    )


def _copy_device_tree(tree):
    """Copy every array leaf so donated train-state fields never alias."""
    return jax.tree_util.tree_map(lambda value: jnp.array(value, copy=True), tree)


def _copy_matching_generator_params(discriminator_params, generator_net_params):
    """Copy every shape-compatible generator tensor into an independent D buffer."""
    target_state = serialization.to_state_dict(discriminator_params)
    source_state = serialization.to_state_dict(generator_net_params)
    loaded = 0

    def merge(target, source):
        nonlocal loaded
        if isinstance(target, dict):
            source = source if isinstance(source, dict) else {}
            return {key: merge(value, source.get(key)) for key, value in target.items()}
        if source is not None and hasattr(source, "shape") and target.shape == source.shape:
            loaded += 1
            # The full train state is donated to the compiled update. Reusing
            # the generator array object here would place the same device
            # buffer in both params and dis_params, which pmap rejects with
            # "Attempt to donate the same buffer twice".
            return jnp.array(source, copy=True)
        return target

    merged = merge(target_state, source_state)
    return serialization.from_state_dict(discriminator_params, merged), loaded


def _mask_discriminator_backbone_grads(grads):
    """Keep gradients only for the newly introduced norm and scalar head."""
    state = serialization.to_state_dict(grads)

    def mask(tree, path=()):
        if isinstance(tree, dict):
            return {key: mask(value, path + (key,)) for key, value in tree.items()}
        trainable = bool(path) and path[0] in {"dis_norm", "dis_head"}
        return tree if trainable else jnp.zeros_like(tree)

    return serialization.from_state_dict(grads, mask(state))


def _mean_metrics(metrics, distributed):
    if distributed:
        return lax.pmean(metrics, axis_name="batch")
    return metrics


def _should_run_fid(current_step, training_config):
    """Match the periodic metric schedule used by the base iMF trainer."""
    forced_steps = str(training_config.get("force_fid_steps", "") or "").strip()
    if forced_steps:
        return current_step in {
            int(step) for step in forced_steps.replace(",", " ").split()
        }

    forced_period = int(training_config.get("force_fid_per_step", 0) or 0)
    if forced_period > 0:
        return current_step % forced_period == 0

    schedule = training_config.get("fid_schedule", [])
    if schedule:
        for item in schedule:
            from_step = int(item.get("from_step", 0))
            until_step = item.get("until_step", None)
            every_steps = int(item["every_steps"])
            if current_step < from_step:
                continue
            if until_step is not None and current_step >= int(until_step):
                continue
            if (current_step - from_step) % every_steps == 0:
                return True
        return False

    period = int(training_config.get("fid_per_step", 0))
    return period > 0 and current_step % period == 0


def _metric_evaluation_enabled(training_config):
    return bool(
        str(training_config.get("force_fid_steps", "") or "").strip()
        or int(training_config.get("force_fid_per_step", 0) or 0) > 0
        or training_config.get("fid_schedule", [])
        or int(training_config.get("fid_per_step", 0)) > 0
    )


def _get_metric_num_steps(config):
    forced_steps = str(
        config.training.get("force_metric_num_steps", "") or ""
    ).strip()
    if forced_steps:
        configured = [
            int(step) for step in forced_steps.replace(",", " ").split()
        ]
    else:
        configured = [
            int(step)
            for step in config.training.get("metric_num_steps", ())
        ]

    primary_steps = int(config.sampling.num_steps)
    ordered_steps = []
    for num_steps in [primary_steps] + configured:
        if num_steps < 1:
            raise ValueError("Metric sampling steps must be >= 1.")
        if num_steps not in ordered_steps:
            ordered_steps.append(num_steps)
    return tuple(ordered_steps)


def _sample_step(
    variable,
    sample_idx,
    *,
    model,
    rng_init,
    device_batch_size,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
):
    """Generate one sharded latent batch using the standard iMF sampler."""
    rng_sample = random.fold_in(rng_init, sample_idx)
    images = generate(
        variable,
        model,
        rng_sample,
        device_batch_size,
        config,
        num_steps,
        omega,
        t_min,
        t_max,
        sample_idx=sample_idx,
    )
    return images.transpose(0, 3, 1, 2)


def _create_sampling_model(config):
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model_config = {
        key: value for key, value in model_config.items() if key in valid_keys
    }
    model_config["eval"] = True
    return iMeanFlow(**model_config)


def _get_images_and_labels(batch, rng_vae, distributed):
    images = batch["image"] if distributed else batch["image"][0]
    labels = batch["label"] if distributed else batch["label"][0]
    return _cached_encode(images, rng_vae), labels


def discriminator_train_step(
    state,
    batch,
    *,
    rng_init,
    model,
    discriminator,
    dis_tx,
    lambda_cp,
    cp_mode,
    interval_eps,
    freeze_backbone,
    distributed,
):
    rng_step = random.fold_in(rng_init, state.step)
    if distributed:
        rng_step = random.fold_in(rng_step, lax.axis_index("batch"))
    rng_gen, rng_vae = random.split(rng_step)
    images, labels = _get_images_and_labels(batch, rng_vae, distributed)

    samples = model.apply(
        {"params": state.params},
        images=images,
        labels=labels,
        current_step=state.step,
        interval_eps=interval_eps,
        rngs={"gen": rng_gen},
        method=model.forward_caimf_discriminator_samples,
    )
    samples = jax.tree_util.tree_map(lax.stop_gradient, samples)

    def loss_fn(dis_params):
        real_logit, fake_logit, potentials = finite_interval_logits(
            discriminator, dis_params, samples
        )
        return discriminator_loss(
            real_logit, fake_logit, potentials, lambda_cp=lambda_cp, cp_mode=cp_mode
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.dis_params)
    if distributed:
        grads = lax.pmean(grads, axis_name="batch")
    if freeze_backbone:
        grads = _mask_discriminator_backbone_grads(grads)
    updates, dis_opt_state = dis_tx.update(
        grads, state.dis_opt_state, state.dis_params
    )
    dis_params = optax.apply_updates(state.dis_params, updates)
    state = state.replace(
        step=state.step + 1,
        dis_step=state.dis_step + 1,
        dis_params=dis_params,
        dis_opt_state=dis_opt_state,
    )
    metrics["phase_discriminator"] = jnp.asarray(1.0)
    metrics["batch_step"] = state.step.astype(jnp.float32)
    metrics["gen_step"] = state.gen_step.astype(jnp.float32)
    metrics["dis_step"] = state.dis_step.astype(jnp.float32)
    return state, _mean_metrics(metrics, distributed)


def generator_train_step(
    state,
    batch,
    *,
    rng_init,
    model,
    discriminator,
    gen_tx,
    lambda_imf,
    lambda_adv,
    lambda_ot,
    interval_eps,
    use_ema,
    ema_decay,
    distributed,
):
    rng_step = random.fold_in(rng_init, state.step)
    if distributed:
        rng_step = random.fold_in(rng_step, lax.axis_index("batch"))
    rng_gen, rng_vae = random.split(rng_step)
    images, labels = _get_images_and_labels(batch, rng_vae, distributed)

    def loss_fn(gen_params):
        terms = model.apply(
            {"params": gen_params},
            images=images,
            labels=labels,
            source_params=None,
            teacher_params=state.ema_params,
            current_step=state.step,
            interval_eps=interval_eps,
            rngs={"gen": rng_gen},
            method=model.forward_caimf_generator_terms,
        )
        if lambda_adv > 0.0:
            fake_logit = finite_fake_logit(
                discriminator, state.dis_params, terms
            )
        else:
            fake_logit = jnp.zeros((images.shape[0],), dtype=images.dtype)
        return generator_loss(
            terms,
            fake_logit,
            lambda_imf=lambda_imf,
            lambda_adv=lambda_adv,
            lambda_ot=lambda_ot,
        )

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    if distributed:
        grads = lax.pmean(grads, axis_name="batch")
    updates, gen_opt_state = gen_tx.update(grads, state.gen_opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    ema_params = state.ema_params
    if use_ema:
        ema_params = update_ema(ema_params, params, ema_decay)
    state = state.replace(
        step=state.step + 1,
        gen_step=state.gen_step + 1,
        params=params,
        ema_params=ema_params,
        gen_opt_state=gen_opt_state,
    )
    metrics["phase_discriminator"] = jnp.asarray(0.0)
    metrics["batch_step"] = state.step.astype(jnp.float32)
    metrics["gen_step"] = state.gen_step.astype(jnp.float32)
    metrics["dis_step"] = state.dis_step.astype(jnp.float32)
    return state, _mean_metrics(metrics, distributed)


def _create_models(config):
    model_config = config.model.to_dict()
    valid_keys = {field.name for field in dataclasses.fields(iMeanFlow)}
    model = iMeanFlow(**{key: value for key, value in model_config.items() if key in valid_keys})

    if not str(config.model.model_str).startswith("imfDiT_"):
        raise ValueError("CA-iMF currently supports model.model_str=imfDiT_* only.")
    net_fn = getattr(imfDiT, config.model.model_str)
    generator_net = net_fn(
        name="net",
        num_classes=int(config.model.num_classes),
        use_null_class=bool(config.model.target_use_null_class),
        use_auxiliary_v_head=bool(config.model.use_auxiliary_v_head),
        eval=False,
    )
    discriminator = create_caimf_discriminator(generator_net)
    return model, discriminator


def _create_state(config, model, discriminator, rng):
    ca_cfg = config.caimf
    gen_lr = float(ca_cfg.gen_learning_rate)
    dis_lr = float(ca_cfg.dis_learning_rate)
    beta1 = float(ca_cfg.adam_beta1)
    beta2 = float(ca_cfg.adam_beta2)
    weight_decay = float(ca_cfg.weight_decay)
    gen_tx = optax.adamw(gen_lr, b1=beta1, b2=beta2, weight_decay=weight_decay)
    dis_tx = optax.adamw(dis_lr, b1=beta1, b2=beta2, weight_decay=weight_decay)

    # The temporary base state reuses the repository's tested partial loader.
    base_state = create_train_state(
        rng,
        config,
        model,
        int(config.dataset.image_size),
        lambda _: jnp.asarray(gen_lr),
    )
    if not config.load_from:
        raise ValueError("load_from must point to an existing iMF checkpoint.")
    base_state = restore_partial_checkpoint(
        base_state,
        config.load_from,
        prefer_ema=bool(ca_cfg.load_generator_ema),
        target_model_config=config.model,
    )
    params = base_state.params
    # deepcopy does not guarantee a distinct backing buffer for JAX arrays.
    # EMA and online params coexist in a donated train state, so force an
    # actual device-array copy for every leaf.
    ema_params = _copy_device_tree(params)

    discriminator_updates = bool(ca_cfg.discriminator_updates)
    if discriminator_updates or float(ca_cfg.lambda_adv) > 0.0:
        rng, rng_dis = random.split(rng)
        batch_size = 1
        image_size = int(config.dataset.image_size)
        dummy_x = jnp.ones((batch_size, image_size, image_size, 4), jnp.float32)
        dummy_time = jnp.full((batch_size,), 0.5, jnp.float32)
        dummy_r = jnp.full((batch_size,), 0.25, jnp.float32)
        dummy_y = jnp.zeros((batch_size,), jnp.int32)
        dis_params = discriminator.init(
            {"params": rng_dis},
            dummy_x,
            dummy_time,
            dummy_r,
            dummy_time,
            dummy_y,
        )["params"]
        dis_params, loaded = _copy_matching_generator_params(
            dis_params, params["net"]
        )
        log_for_0(
            "Initialized discriminator from generator backbone: copied %d tensors; "
            "new scalar head remains freshly initialized.",
            loaded,
        )
        dis_opt_state = dis_tx.init(dis_params)
    else:
        dis_params = None
        dis_opt_state = None

    state = CAIMFTrainState(
        step=jnp.asarray(0, jnp.int32),
        gen_step=jnp.asarray(0, jnp.int32),
        dis_step=jnp.asarray(0, jnp.int32),
        params=params,
        ema_params=ema_params,
        gen_opt_state=gen_tx.init(params),
        dis_params=dis_params,
        dis_opt_state=dis_opt_state,
    )
    return state, gen_tx, dis_tx


def _set_num_classes_from_data(config):
    if not config.dataset.get("num_classes_from_data", False):
        return
    train_root = os.path.join(config.dataset.root, "train")
    labels = []
    import torch

    for filename in os.listdir(train_root):
        if filename.endswith(".pt"):
            sample = torch.load(
                os.path.join(train_root, filename), map_location="cpu", weights_only=False
            )
            labels.append(int(sample["label"]))
    if not labels:
        raise ValueError(f"No latent .pt files found under {train_root}")
    num_classes = max(labels) + 1
    config.dataset.num_classes = num_classes
    config.model.num_classes = num_classes
    config.sampling.num_classes = num_classes
    log_for_0("Inferred dataset.num_classes=%d from target latents.", num_classes)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    """Run CA-iMF post-training and save evaluation-compatible checkpoints."""
    writer = Writer(config, workdir)
    _set_num_classes_from_data(config)
    ca_cfg = config.caimf
    if int(config.training.get("grad_accum_steps", 1)) != 1:
        raise ValueError("CA-iMF currently requires training.grad_accum_steps=1.")

    local_batch_size = int(config.training.batch_size) // jax.process_count()
    train_loader, steps_per_epoch = input_pipeline.create_latent_split(
        config.dataset, local_batch_size, split="train"
    )
    model, discriminator = _create_models(config)
    rng = random.key(int(config.training.seed))
    state, gen_tx, dis_tx = _create_state(config, model, discriminator, rng)

    # --- Full-state resume (e.g. 150k -> 300k continuation) --------------
    # load_from restores generator params ONLY (step reset to 0, fresh
    # optimizer + freshly-initialized discriminator). caimf.resume_from
    # instead restores the COMPLETE adversarial CAIMFTrainState -- generator
    # and EMA params, both optimizer states, the discriminator params/opt,
    # and the step/gen_step/dis_step counters -- so training continues
    # exactly where it stopped, preserving the D/G equilibrium and the
    # post-warmup D:G cadence. Point it at a workdir (latest checkpoint_* is
    # chosen) or a specific checkpoint_* directory.
    resume_from = str(ca_cfg.get("resume_from", "") or "")
    if resume_from:
        resume_from = os.path.abspath(resume_from)
        restored = checkpoints.restore_checkpoint(resume_from, state)
        restored_step = int(np.asarray(restored.step).reshape(-1)[0])
        if restored_step <= 0:
            raise ValueError(
                f"caimf.resume_from={resume_from} restored step "
                f"{restored_step}; expected a full-state checkpoint (step>0). "
                "Point it at a periodic checkpoint_* dir, not best_fid/."
            )
        state = restored
        log_for_0(
            "Resumed FULL CA-iMF state from %s at step %d "
            "(continuing to max_posttrain_batches=%d).",
            resume_from,
            restored_step,
            int(ca_cfg.max_posttrain_batches),
        )
    # ---------------------------------------------------------------------

    distributed = jax.local_device_count() > 1
    if distributed:
        state = jax_utils.replicate(state)

    common_d = dict(
        rng_init=rng,
        model=model,
        discriminator=discriminator,
        dis_tx=dis_tx,
        lambda_cp=float(ca_cfg.lambda_cp),
        cp_mode=str(ca_cfg.get("cp_mode", "full")),
        interval_eps=float(ca_cfg.interval_eps),
        freeze_backbone=bool(ca_cfg.freeze_discriminator_backbone),
        distributed=distributed,
    )
    common_g = dict(
        rng_init=rng,
        model=model,
        discriminator=discriminator,
        gen_tx=gen_tx,
        lambda_imf=float(ca_cfg.lambda_imf),
        lambda_adv=float(ca_cfg.lambda_adv),
        lambda_ot=float(ca_cfg.lambda_ot),
        interval_eps=float(ca_cfg.interval_eps),
        use_ema=bool(config.training.use_ema),
        ema_decay=float(config.training.ema_val),
        distributed=distributed,
    )
    if distributed:
        p_dis_step = jax.pmap(
            lambda state_value, batch_value: discriminator_train_step(
                state_value, batch_value, **common_d
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
        p_gen_step = jax.pmap(
            lambda state_value, batch_value: generator_train_step(
                state_value, batch_value, **common_g
            ),
            axis_name="batch",
            donate_argnums=(0,),
        )
    else:
        p_dis_step = jax.jit(
            lambda state_value, batch_value: discriminator_train_step(
                state_value, batch_value, **common_d
            ),
            donate_argnums=(0,),
        )
        p_gen_step = jax.jit(
            lambda state_value, batch_value: generator_train_step(
                state_value, batch_value, **common_g
            ),
            donate_argnums=(0,),
        )

    def step_value(state_value):
        value = np.asarray(jax.device_get(state_value.step))
        return int(value.reshape(-1)[0])

    def save_state(state_value):
        if distributed:
            save_checkpoint(state_value, workdir)
        else:
            # Single device: state has no leading device axis. Replicating
            # here would allocate a 2nd full copy of the fat adversarial
            # state on the GPU, on top of resident eval models -> OOM.
            # device_get pulls to host RAM (no 2nd GPU copy); save directly.
            host_state = jax.device_get(state_value)
            step = int(np.asarray(host_state.step).reshape(-1)[0])
            log_for_0("Saving checkpoint step %d (single-device).", step)
            checkpoints.save_checkpoint_multiprocess(
                os.path.abspath(workdir), host_state, step, keep=3
            )
            log_for_0("Checkpoint step %d saved.", step)

    metric_evaluation_enabled = _metric_evaluation_enabled(config.training)
    image_metric_evaluator = None
    metric_num_steps = ()
    p_metric_sample_steps = {}
    sample_kwargs = None
    latent_manager = None
    p_preview_sample_steps = {}
    preview_num_images = int(config.training.get("preview_num_images", 16))
    preview_grid_size = int(np.sqrt(max(preview_num_images, 1)))
    preview_num_images = preview_grid_size * preview_grid_size
    best_fid_by_steps = {}
    best_fd_dino_by_steps = {}
    best_fid_ckpt_dir = os.path.join(
        workdir,
        config.training.get("best_fid_checkpoint_dir", "best_fid"),
    )
    eval_ckpt_dir = os.path.join(
        workdir,
        config.training.get("eval_checkpoint_dir", "latest_eval"),
    )
    use_ema_for_metrics = bool(config.training.use_ema) and not bool(
        config.training.get("fid_use_online_only", False)
    )
    metric_mode = "ema" if use_ema_for_metrics else "online"

    if metric_evaluation_enabled:
        sample_device_bsz = get_sample_device_batch_size(config)
        sample_local_device_count = get_sample_local_device_count(config)
        sample_devices = get_sample_devices(config)
        sampling_model = _create_sampling_model(config)
        latent_manager = LatentManager(
            config.dataset.vae,
            sample_device_bsz,
            int(config.dataset.image_size),
            decode_num_local_devices=sample_local_device_count,
        )

        def build_p_sample_step(num_steps):
            return jax.pmap(
                partial(
                    _sample_step,
                    model=sampling_model,
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
            num_steps: build_p_sample_step(num_steps)
            for num_steps in metric_num_steps
        }
        preview_steps = tuple(
            int(step)
            for step in config.training.get("preview_num_steps", (1, 2, 4))
        )
        p_preview_sample_steps = {
            step: build_p_sample_step(step) for step in preview_steps if step >= 1
        }
        best_fid_by_steps = {
            num_steps: float("inf") for num_steps in metric_num_steps
        }
        best_fd_dino_by_steps = {
            num_steps: float("inf") for num_steps in metric_num_steps
        }
        controllable_guidance = has_controllable_sampling_guidance(config.model)
        eval_sample_kwargs = {
            "omega": (
                float(config.sampling.omega) if controllable_guidance else 1.0
            ),
            "t_min": float(config.sampling.t_min),
            "t_max": float(config.sampling.t_max),
        }
        sample_kwargs = jax_utils.replicate(
            eval_sample_kwargs, devices=sample_devices
        )
        log_for_0(
            "CA-iMF metric evaluation enabled for sampling steps %s; mode=%s.",
            metric_num_steps,
            metric_mode,
        )

    def write_preview(state_value, step_value):
        if (
            not metric_evaluation_enabled
            or not p_preview_sample_steps
            or preview_num_images <= 0
        ):
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
                ema=use_ema_for_metrics,
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

    warmup = int(ca_cfg.discriminator_warmup_batches)
    dis_steps = int(ca_cfg.discriminator_steps_per_cycle)
    discriminator_updates = bool(ca_cfg.discriminator_updates)
    max_batches = int(ca_cfg.max_posttrain_batches)
    log_every = int(config.training.log_per_step)
    timer = Timer()
    compiled_phases = set()
    should_stop = False

    log_for_0(
        "CA-iMF schedule: warmup=%d D-only batches, then %dD:1G; "
        "lambda_imf=%g lambda_adv=%g lambda_ot=%g.",
        warmup,
        dis_steps,
        float(ca_cfg.lambda_imf),
        float(ca_cfg.lambda_adv),
        float(ca_cfg.lambda_ot),
    )

    if metric_evaluation_enabled and int(config.training.get("sample_per_step", 0)) > 0:
        write_preview(state, 0)

    for epoch in range(int(config.training.num_epochs)):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            batch_step = step_value(state)
            batch = input_pipeline.prepare_batch_data(batch)
            do_discriminator = discriminator_updates and bool(
                np.asarray(is_discriminator_step(batch_step, warmup, dis_steps))
            )
            phase = "discriminator" if do_discriminator else "generator"
            if do_discriminator:
                state, metrics = p_dis_step(state, batch)
            else:
                state, metrics = p_gen_step(state, batch)
            current_step = step_value(state)

            sample_period = int(config.training.get("sample_per_step", 0))
            if (
                sample_period > 0
                and current_step > 0
                and current_step % sample_period == 0
            ):
                write_preview(state, current_step)

            if phase not in compiled_phases:
                jax.tree_util.tree_leaves(metrics)[0].block_until_ready()
                log_for_0(
                    "Initial %s step compiled in %.2fs.",
                    phase,
                    timer.elapse_with_reset(),
                )
                compiled_phases.add(phase)

            if current_step == 1 or current_step % log_every == 0:
                host_metrics = {
                    key: float(np.asarray(jax.device_get(value)).mean())
                    for key, value in metrics.items()
                }
                host_metrics["batches_per_second"] = log_every / max(
                    timer.elapse_with_reset(), 1e-6
                )
                writer.write_scalars(current_step, host_metrics)

            if (
                metric_evaluation_enabled
                and current_step > 0
                and _should_run_fid(current_step, config.training)
            ):
                metric_state = EvalState(
                    step=state.step,
                    params=state.params,
                    ema_params=state.ema_params,
                )
                if not distributed:
                    metric_state = jax_utils.replicate(
                        metric_state, devices=sample_devices
                    )
                checkpoint_state = (
                    state if distributed else jax_utils.replicate(state)
                )
                if image_metric_evaluator is None:
                    image_metric_evaluator = get_image_metric_evaluator(
                        config, writer, latent_manager
                    )

                checkpoint_path_for_csv = ""
                if config.training.get("save_eval_checkpoint_per_fid", False):
                    save_best_checkpoint(checkpoint_state, eval_ckpt_dir)
                    checkpoint_path_for_csv = eval_ckpt_dir

                for metric_num_steps, p_metric_sample_step in (
                    p_metric_sample_steps.items()
                ):
                    log_for_0(
                        "Running CA-iMF metric evaluation at batch step %d "
                        "with %d sampling steps.",
                        current_step,
                        metric_num_steps,
                    )
                    result = image_metric_evaluator(
                        metric_state,
                        p_metric_sample_step,
                        current_step - 1,
                        ema_only=use_ema_for_metrics,
                        metric_suffix=f"steps_{metric_num_steps}",
                        **sample_kwargs,
                    )
                    fid = float(result["fid"])
                    fd_dino = result.get("fd_dino", None)
                    is_best_fid = fid < best_fid_by_steps[metric_num_steps]
                    is_best_fd_dino = (
                        fd_dino is not None
                        and float(fd_dino)
                        < best_fd_dino_by_steps[metric_num_steps]
                    )
                    if is_best_fid:
                        best_fid_by_steps[metric_num_steps] = fid
                    if is_best_fd_dino:
                        best_fd_dino_by_steps[metric_num_steps] = float(fd_dino)

                    row_checkpoint_path = checkpoint_path_for_csv
                    if (
                        metric_num_steps == int(config.sampling.num_steps)
                        and is_best_fid
                    ):
                        log_for_0(
                            "New best CA-iMF FID %.4f at batch step %d; "
                            "saving to %s.",
                            fid,
                            current_step,
                            best_fid_ckpt_dir,
                        )
                        save_best_checkpoint(
                            checkpoint_state,
                            best_fid_ckpt_dir,
                            eval_state_only=bool(
                                config.training.get(
                                    "save_best_fid_eval_state_only", True
                                )
                            ),
                        )
                        row_checkpoint_path = best_fid_ckpt_dir

                    append_eval_metrics_row(
                        workdir,
                        {
                            "eval_phase": "train",
                            "metric_mode": metric_mode,
                            "training_step": current_step,
                            "sampling_num_steps": metric_num_steps,
                            "omega": float(config.sampling.omega),
                            "t_min": float(config.sampling.t_min),
                            "t_max": float(config.sampling.t_max),
                            "fid": fid,
                            "inception_score": float(result["is"]),
                            "fd_dino": (
                                "" if fd_dino is None else float(fd_dino)
                            ),
                            "is_best_fid": int(is_best_fid),
                            "is_best_fd_dino": int(is_best_fd_dino),
                            "checkpoint_path": (
                                os.path.abspath(row_checkpoint_path)
                                if row_checkpoint_path
                                else ""
                            ),
                        },
                    )

                # cp-ablation eval-path OOM fix: on a single GPU the two
                # jax_utils.replicate copies (metric_state, checkpoint_state)
                # of the fat adversarial state stay resident and overflow the
                # next p_dis_step. Free them before resuming training.
                if not distributed:
                    jax.tree_util.tree_map(
                        lambda x: x.delete()
                        if hasattr(x, "delete")
                        else None,
                        (metric_state, checkpoint_state),
                    )
                    del metric_state, checkpoint_state

            if current_step >= max_batches:
                should_stop = True
                break

        if (
            should_stop
            or (epoch + 1) % int(config.training.checkpoint_per_epoch) == 0
            or (epoch + 1) == int(config.training.num_epochs)
        ):
            save_state(state)
        if should_stop:
            log_for_0("Reached max_posttrain_batches=%d.", max_batches)
            break

    jax.random.normal(jax.random.key(0), ()).block_until_ready()
    return state
