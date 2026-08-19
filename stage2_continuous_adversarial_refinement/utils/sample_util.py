import os
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
from utils import dino_util
from utils import fid_util
from utils.logging_util import log_for_0
from utils.imf_param_util import use_v_only_teacher_source_copies


def get_sample_local_device_count(config):
    fid_config = config.get("fid", {})
    local_device_count = jax.local_device_count()
    if fid_config.get("sample_first_device_only", False):
        return 1

    requested = int(fid_config.get("sample_num_local_devices", 0))
    if requested <= 0:
        return local_device_count
    if requested > local_device_count:
        raise ValueError(
            f"Requested {requested} sample local devices, but only "
            f"{local_device_count} local devices are visible."
        )
    return requested


def get_sample_devices(config):
    return jax.local_devices()[: get_sample_local_device_count(config)]


def _slice_local_device_axis(tree, num_local_devices):
    if num_local_devices >= jax.local_device_count():
        return tree

    def maybe_slice(x):
        if hasattr(x, "shape") and x.shape and x.shape[0] == jax.local_device_count():
            return x[:num_local_devices]
        return x

    return jax.tree_util.tree_map(maybe_slice, tree)


def has_controllable_sampling_guidance(model_config):
    if model_config.get("uses_classifier_free_guidance", False):
        return True
    if model_config.get("use_auxiliary_v_head", True):
        return True
    if model_config.get("use_context_guidance_conditioning", False):
        return True
    if model_config.get("use_adaln_guidance_scale_conditioning", False):
        return True
    return not (
        model_config.get("use_training_guidance", True)
        and model_config.get("guidance_scale_strategy", "sampled") == "fixed"
    )


def run_p_sample_step(
    p_sample_step,
    state,
    sample_idx,
    latent_manager,
    ema=True,
    sample_local_device_count=None,
    **kwargs,
):
    """
    Run one p_sample_step to get samples from the model.
    """
    params = state.ema_params if ema else state.params
    if sample_local_device_count is not None:
        params = _slice_local_device_axis(params, sample_local_device_count)
        kwargs = _slice_local_device_axis(kwargs, sample_local_device_count)

    variable = {"params": params}
    latent = p_sample_step(variable, sample_idx=sample_idx, **kwargs)
    latent = latent.reshape(-1, *latent.shape[2:])

    samples = latent_manager.decode(latent)
    assert not jnp.any(
        jnp.isnan(samples)
    ), f"There is nan in decoded samples! Latent range: {latent.min()}, {latent.max()}. nan in latent: {jnp.any(jnp.isnan(latent))}"

    samples = samples.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    samples = 127.5 * samples + 128.0
    samples = jnp.clip(samples, 0, 255).astype(jnp.uint8)

    jax.random.normal(random.key(0), ()).block_until_ready()  # dist sync
    return samples


def get_sample_device_batch_size(config):
    sample_device_batch_size = int(config.fid.get("sample_device_batch_size", -1))
    if sample_device_batch_size <= 0:
        sample_device_batch_size = config.fid.device_batch_size
    return int(sample_device_batch_size)


def _get_half_precision_dtype(dtype_name):
    dtype_name = str(dtype_name).lower()
    if dtype_name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    if dtype_name in ("fp16", "float16"):
        return jnp.float16
    raise ValueError(
        "half precision dtype must be one of: bfloat16, bf16, float16, fp16. "
        f"Got {dtype_name!r}."
    )


def get_training_param_dtype(config):
    if not config.training.get("half_precision", False):
        return None

    return _get_half_precision_dtype(config.training.get("half_precision_dtype", "float16"))


def get_sampling_param_dtype(config):
    sampling = config.get("sampling", {})
    if not sampling.get("half_precision", False):
        return None

    return _get_half_precision_dtype(
        sampling.get(
            "half_precision_dtype",
            config.training.get("half_precision_dtype", "float16"),
        )
    )


def _cast_floating_tree(tree, dtype):
    if tree is None or dtype is None:
        return tree

    def maybe_cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(maybe_cast, tree)


def maybe_cast_state_for_sampling(state, config):
    dtype = get_sampling_param_dtype(config)
    if dtype is None:
        return state

    log_for_0("Casting sampling params to %s.", dtype)
    return state.replace(
        params=_cast_floating_tree(state.params, dtype),
        ema_params=_cast_floating_tree(state.ema_params, dtype),
    )


def generate_fid_samples(
    state, config, p_sample_step, run_p_sample_step, ema=True, num_samples=None, **kwargs
):
    """
    Generate samples for FID evaluation or preview logging.
    """
    target_num_samples = config.fid.num_samples if num_samples is None else num_samples
    sample_device_batch_size = get_sample_device_batch_size(config)
    sample_local_device_count = get_sample_local_device_count(config)
    sample_global_device_count = sample_local_device_count * jax.process_count()
    state = maybe_cast_state_for_sampling(state, config)
    num_steps = np.ceil(
        target_num_samples / sample_device_batch_size / sample_global_device_count
    ).astype(int)
    log_sample_every = max(1, int(config.fid.get("sample_log_every", 1)))

    samples_all = []

    log_for_0("Note: the first sample may be significant slower")
    log_for_0(
        "Generating %d samples with sample_device_batch_size=%d, global_batch_size=%d, batches=%d",
        target_num_samples,
        sample_device_batch_size,
        sample_device_batch_size * sample_global_device_count,
        num_steps,
    )
    for step in range(num_steps):
        sample_idx = jax.process_index() * sample_local_device_count + jnp.arange(
            sample_local_device_count
        )
        sample_idx = sample_global_device_count * step + sample_idx
        should_log = step == 0 or step + 1 == num_steps or step % log_sample_every == 0
        if should_log:
            log_for_0(f"Sampling step {step} / {num_steps}...")
        sample_start = time.time()
        samples = run_p_sample_step(
            p_sample_step,
            state,
            sample_idx=sample_idx,
            ema=ema,
            sample_local_device_count=sample_local_device_count,
            **kwargs,
        )
        samples.block_until_ready()
        device_seconds = time.time() - sample_start
        transfer_start = time.time()
        samples = jax.device_get(samples)
        transfer_seconds = time.time() - transfer_start
        if should_log:
            log_for_0(
                "Sampling batch %d / %d timing: device %.2fs, device_get %.2fs",
                step,
                num_steps,
                device_seconds,
                transfer_seconds,
            )
        samples_all.append(samples)

    samples_all = np.concatenate(samples_all, axis=0)

    return samples_all[:target_num_samples]


def _get_eval_descriptor(kwargs, mode_str, guidance_controllable=True, metric_suffix=""):
    omega = kwargs.get("omega", None)[0]
    t_min = kwargs.get("t_min", None)[0]
    t_max = kwargs.get("t_max", None)[0]
    if guidance_controllable:
        descriptor = f"omega_{omega:.2f}_tmin_{t_min:.2f}_tmax_{t_max:.2f}_{mode_str}"
    else:
        descriptor = f"single_head_{mode_str}"
    if metric_suffix:
        descriptor = f"{descriptor}_{metric_suffix}"
    return descriptor, omega, t_min, t_max


def get_image_metric_evaluator(config, writer, latent_manager):
    """
    Create a single evaluator that logs FID, Inception Score, and optionally FD-DINO.
    """
    sample_local_device_count = get_sample_local_device_count(config)
    inception_batch_size = config.fid.device_batch_size * sample_local_device_count
    fid_device = os.environ.get("IMF_FID_DEVICE", "").strip().lower() or None
    inception_net = fid_util.build_jax_inception(
        batch_size=inception_batch_size,
        device=fid_device,
    )
    fid_stats_ref = fid_util.get_reference(config.fid.cache_ref)

    fd_dino_config = config.get("fd_dino", None)
    fd_dino_enabled = bool(fd_dino_config and fd_dino_config.get("cache_ref", ""))
    dino_net = None
    fd_dino_stats_ref = None
    if fd_dino_enabled:
        # FD-DINO is optional.  Do not abort an otherwise valid FID/IS run
        # merely because the local auxiliary DINO weights were not installed.
        # Resolve defaults relative to the repository, since callers may start
        # from a different working directory (screen/slurm jobs commonly do).
        weights_path = os.environ.get(
            "DINOV2_JAX_WEIGHTS",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "files", "weights", "dinov2_base_hf.safetensors"),
        )
        posembed_path = os.environ.get(
            "DINOV2_JAX_POSEMBED",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "files", "weights", "dinov2_base_posembed_224.npy"),
        )
        if not (os.path.isfile(weights_path) and os.path.isfile(posembed_path)):
            if os.environ.get("REQUIRE_FD_DINO", "0").lower() in {"1", "true", "yes"}:
                raise FileNotFoundError(
                    "FD-DINO is enabled but its local weights are missing. "
                    f"Expected {weights_path} and {posembed_path}."
                )
            log_for_0(
                "FD-DINO disabled: missing local weights (%s, %s). "
                "FID and IS evaluation will continue.",
                weights_path,
                posembed_path,
            )
            fd_dino_enabled = False
        else:
            # Pass absolute paths into the loader so evaluation is independent
            # of the process working directory.
            os.environ.setdefault("DINOV2_JAX_WEIGHTS", weights_path)
            os.environ.setdefault("DINOV2_JAX_POSEMBED", posembed_path)
    if fd_dino_enabled:
        dino_net = dino_util.build_jax_dinov2(
            arch=fd_dino_config.get("arch", "vitb14"),
            model_name=fd_dino_config.get("model_name", None),
            batch_size=config.fid.device_batch_size
            * sample_local_device_count
            * jax.process_count(),
        )
        fd_dino_stats_ref = fid_util.get_reference(fd_dino_config.cache_ref)

    run_p_sample_step_inner = partial(run_p_sample_step, latent_manager=latent_manager)
    use_ema = config.training.get("use_ema", True)
    if use_v_only_teacher_source_copies(config.model):
        use_ema = False
    fid_use_online_only = config.training.get("fid_use_online_only", False)
    guidance_controllable = has_controllable_sampling_guidance(config.model)

    def _evaluate_one_mode(state, p_sample_step, ema, metric_suffix="", **kwargs):
        samples_all = generate_fid_samples(
            state, config, p_sample_step, run_p_sample_step_inner, ema, **kwargs
        )

        fid_stats = fid_util.compute_stats(
            samples_all,
            inception_net,
            batch_size=config.fid.device_batch_size,
            fid_samples=config.fid.num_samples,
            num_local_devices=sample_local_device_count,
        )
        metric = {}
        result = {}

        mode_str = "ema" if ema else "online"
        descriptor, omega, t_min, t_max = _get_eval_descriptor(
            kwargs,
            mode_str,
            guidance_controllable=guidance_controllable,
            metric_suffix=metric_suffix,
        )
        if guidance_controllable:
            log_for_0(
                f"Computing image metrics at omega={omega:.2f}, t_min={t_min:.2f}, "
                f"t_max={t_max:.2f}, mode={mode_str}..."
            )
        else:
            log_for_0(
                f"Computing image metrics for single-head evaluation without "
                f"sampling-time guidance control, mode={mode_str}..."
            )

        fid = fid_util.compute_fid(
            fid_stats_ref["mu"],
            fid_stats["mu"],
            fid_stats_ref["sigma"],
            fid_stats["sigma"],
        )
        is_score, _ = fid_util.compute_inception_score(fid_stats["logits"])

        metric[f"FID_{descriptor}"] = fid
        metric[f"IS_{descriptor}"] = is_score
        result["fid"] = fid
        result["is"] = is_score

        if fd_dino_enabled:
            dino_stats = fid_util.compute_dinov2_stats(
                samples_all,
                dino_net,
                batch_size=config.fid.device_batch_size,
                fid_samples=config.fid.num_samples,
                num_local_devices=sample_local_device_count,
            )
            fd_dino = fid_util.compute_fid(
                fd_dino_stats_ref["mu"],
                dino_stats["mu"],
                fd_dino_stats_ref["sigma"],
                dino_stats["sigma"],
            )
            metric[f"FD_DINO_{descriptor}"] = fd_dino
            result["fd_dino"] = fd_dino

        return metric, result

    def evaluator(
        state,
        p_sample_step,
        step,
        ema_only=False,
        metric_suffix="",
        **kwargs,
    ):
        metric_dict = {}
        primary_result = None
        if fid_use_online_only:
            metric, primary_result = _evaluate_one_mode(
                state,
                p_sample_step,
                False,
                metric_suffix=metric_suffix,
                **kwargs,
            )
            metric_dict.update(metric)
        elif use_ema:
            metric, primary_result = _evaluate_one_mode(
                state,
                p_sample_step,
                True,
                metric_suffix=metric_suffix,
                **kwargs,
            )
            metric_dict.update(metric)
            if not ema_only:
                metric, _ = _evaluate_one_mode(
                    state,
                    p_sample_step,
                    False,
                    metric_suffix=metric_suffix,
                    **kwargs,
                )
                metric_dict.update(metric)
        else:
            metric, primary_result = _evaluate_one_mode(
                state,
                p_sample_step,
                False,
                metric_suffix=metric_suffix,
                **kwargs,
            )
            metric_dict.update(metric)

        writer.write_scalars(step + 1, metric_dict)
        return primary_result

    return evaluator


def get_fid_evaluator(config, writer, latent_manager):
    """
    Backward-compatible wrapper that returns FID and IS.
    """
    metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        result = metric_evaluator(
            state, p_sample_step, step, ema_only=ema_only, **kwargs
        )
        return result["fid"], result["is"]

    return evaluator


def get_fd_dino_evaluator(config, writer, latent_manager):
    """
    Backward-compatible wrapper that returns only FD-DINO.
    """
    fd_dino_config = config.get("fd_dino", None)
    if fd_dino_config is None or not fd_dino_config.get("cache_ref", ""):
        raise ValueError("config.fd_dino.cache_ref must be set to use FD-DINO evaluation.")

    metric_evaluator = get_image_metric_evaluator(config, writer, latent_manager)

    def evaluator(state, p_sample_step, step, ema_only=False, **kwargs):
        result = metric_evaluator(
            state, p_sample_step, step, ema_only=ema_only, **kwargs
        )
        return result["fd_dino"]

    return evaluator
