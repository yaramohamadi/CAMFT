"""Sampling utilities for the dedicated plain SiT path."""

import jax
from jax import random
import jax.numpy as jnp

from utils.sit_transport_jax import create_transport


def _get_sampling_dtype(config):
    sampling = config.get("sampling", {})
    if not sampling.get("half_precision", False):
        return jnp.float32

    dtype_name = str(
        sampling.get(
            "half_precision_dtype",
            config.training.get("half_precision_dtype", "float16"),
        )
    ).lower()
    if dtype_name in ("fp16", "float16"):
        return jnp.float16
    if dtype_name in ("bf16", "bfloat16"):
        return jnp.bfloat16
    raise ValueError(
        "training.half_precision_dtype must be one of: bfloat16, bf16, float16, fp16. "
        f"Got {dtype_name!r}."
    )


def _make_sample_labels(num_samples, num_classes, sample_idx=None):
    if sample_idx is not None:
        labels = jnp.arange(num_samples, dtype=jnp.int32)
        labels = labels + jnp.asarray(sample_idx, dtype=jnp.int32) * num_samples
        return labels % num_classes

    return random.randint(
        random.PRNGKey(0), (num_samples,), 0, num_classes, dtype=jnp.int32
    )


def _make_random_sample_labels(rng, num_samples, num_classes):
    return random.randint(
        rng,
        (num_samples,),
        0,
        num_classes,
        dtype=jnp.int32,
    )


def _guided_velocity(model, variable, x, t_model, labels, cfg_scale):
    num_samples = x.shape[0]

    def conditional_velocity(_):
        return model.apply(variable, x, t_model, labels)

    def classifier_free_guided_velocity(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)

        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t_model, t_model], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)

        out = model.apply(variable, x_cat, t_cat, y_cat)
        cond, uncond = jnp.split(out, 2, axis=0)
        return uncond + cfg_scale * (cond - uncond)

    return jax.lax.cond(
        jnp.equal(cfg_scale, 1.0),
        conditional_velocity,
        classifier_free_guided_velocity,
        operand=None,
    )


def _guided_native_output(model, variable, x, t_model, labels, cfg_scale):
    num_samples = x.shape[0]

    def conditional_output(_):
        return model.apply(
            variable,
            x,
            t_model,
            labels,
            method=model.predict_native_output,
        )

    def classifier_free_guided_output(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)

        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t_model, t_model], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)

        out = model.apply(
            variable,
            x_cat,
            t_cat,
            y_cat,
            method=model.predict_native_output,
        )
        cond, uncond = jnp.split(out, 2, axis=0)
        return uncond + cfg_scale * (cond - uncond)

    return jax.lax.cond(
        jnp.equal(cfg_scale, 1.0),
        conditional_output,
        classifier_free_guided_output,
        operand=None,
    )


def _guided_meanflow(model, variable, x, t_model, r_model, labels, cfg_scale):
    num_samples = x.shape[0]

    def conditional_meanflow(_):
        return model.apply(variable, x, t_model, labels, r=r_model)

    def classifier_free_guided_meanflow(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)

        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t_model, t_model], axis=0)
        r_cat = jnp.concatenate([r_model, r_model], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)

        out = model.apply(variable, x_cat, t_cat, y_cat, r=r_cat)
        cond, uncond = jnp.split(out, 2, axis=0)
        return uncond + cfg_scale * (cond - uncond)

    return jax.lax.cond(
        jnp.equal(cfg_scale, 1.0),
        conditional_meanflow,
        classifier_free_guided_meanflow,
        operand=None,
    )


def _use_interval_meanflow_sampling(config):
    return str(config.transport.get("objective", "sit")) == "power_meanflow"


def _create_sampling_transport(config):
    prediction = config.transport.prediction
    train_eps = config.transport.train_eps
    sample_eps = config.transport.sample_eps

    output_prediction_space = str(
        config.model.get("sit_output_prediction_space", "velocity")
    )
    if (
        str(config.transport.get("objective", "sit")) == "sit"
        and output_prediction_space != "velocity"
    ):
        min_eps = max(float(config.model.get("sit_wrapper_eps", 1e-6)), 1e-3)
        train_eps = min_eps if train_eps is None else max(float(train_eps), min_eps)
        sample_eps = min_eps if sample_eps is None else max(float(sample_eps), min_eps)

    return create_transport(
        path_type=config.transport.path_type,
        prediction=prediction,
        loss_weight=config.transport.loss_weight,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )


def _use_native_prediction_sampling(config):
    return bool(config.sampling.get("plain_sit_native_prediction_sampling", False))


def _get_native_prediction_space(config):
    return str(config.model.get("sit_output_prediction_space", "velocity"))


def _native_prediction_to_drift(transport, prediction_space, native_output, x, t):
    if prediction_space == "velocity":
        return native_output

    drift_mean, drift_var = transport.path_sampler.compute_drift(x, t)
    if prediction_space == "noise":
        sigma_t, _ = transport.path_sampler.compute_sigma_t(
            t.reshape((t.shape[0],) + (1,) * (x.ndim - 1))
        )
        score = native_output / -sigma_t
        return -drift_mean + drift_var * score

    raise ValueError(
        "Native plain SiT prediction sampling currently supports "
        f"'velocity' and 'noise', got {prediction_space!r}."
    )


def _convert_final_sample_to_data(
    model,
    variable,
    x,
    t_model,
    labels,
    cfg_scale,
    *,
    native_prediction_sampling,
    prediction_space,
):
    if not native_prediction_sampling or prediction_space == "velocity":
        return x

    native_output = _guided_native_output(
        model,
        variable,
        x,
        t_model,
        labels,
        cfg_scale,
    )
    return model.apply(
        variable,
        native_output,
        x,
        t_model,
        method=model.convert_native_output_to_data,
    )


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    sample_idx=None,
):
    """Generate latent samples with plain SiT and fixed-step ODE integration."""
    del t_min, t_max
    if num_steps < 1:
        raise ValueError(
            f"Plain SiT sampling requires num_steps >= 1, got {num_steps}."
        )

    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels
    sample_dtype = _get_sampling_dtype(config)
    x = jax.random.normal(
        rng,
        (n_sample, img_size, img_size, img_channels),
        dtype=sample_dtype,
    )
    labels = _make_sample_labels(
        n_sample,
        config.dataset.num_classes,
        sample_idx=sample_idx,
    )

    transport = _create_sampling_transport(config)
    tau0, tau1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    times = jnp.linspace(float(tau0), float(tau1), num_steps + 1, dtype=sample_dtype)

    method = str(config.sampling.get("method", "euler")).lower()
    flip_time = bool(config.sampling.get("flip_time", False))
    cfg_scale = jnp.asarray(omega, dtype=sample_dtype)
    native_prediction_sampling = _use_native_prediction_sampling(config)
    prediction_space = _get_native_prediction_space(config)

    def euler_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output = _guided_native_output(
                model,
                variable,
                x_cur,
                t_cur,
                labels,
                cfg_scale,
            )
            drift = _native_prediction_to_drift(
                transport, prediction_space, native_output, x_cur, t_cur
            )
        else:
            drift = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        return (x_cur + dt * drift).astype(sample_dtype)

    def heun_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output = _guided_native_output(
                model,
                variable,
                x_cur,
                t_cur,
                labels,
                cfg_scale,
            )
            drift = _native_prediction_to_drift(
                transport, prediction_space, native_output, x_cur, t_cur
            )
        else:
            drift = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        x_pred = x_cur + dt * drift
        t_next = jnp.full(
            (n_sample,),
            1.0 - tau_next if flip_time else tau_next,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output_next = _guided_native_output(
                model,
                variable,
                x_pred,
                t_next,
                labels,
                cfg_scale,
            )
            drift_next = _native_prediction_to_drift(
                transport, prediction_space, native_output_next, x_pred, t_next
            )
        else:
            drift_next = _guided_velocity(
                model, variable, x_pred, t_next, labels, cfg_scale
            )
        return (x_cur + 0.5 * dt * (drift + drift_next)).astype(sample_dtype)

    def meanflow_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        r_next = jnp.full(
            (n_sample,),
            1.0 - tau_next if flip_time else tau_next,
            dtype=sample_dtype,
        )
        meanflow = _guided_meanflow(
            model,
            variable,
            x_cur,
            t_cur,
            r_next,
            labels,
            cfg_scale,
        )
        delta_t = (r_next - t_cur).reshape((n_sample, 1, 1, 1))
        return (x_cur + delta_t * meanflow).astype(sample_dtype)

    if _use_interval_meanflow_sampling(config):
        return jax.lax.fori_loop(0, num_steps, meanflow_step, x)
    if method == "euler":
        x_final = jax.lax.fori_loop(0, num_steps, euler_step, x)
    elif method == "heun":
        x_final = jax.lax.fori_loop(0, num_steps, heun_step, x)
    else:
        raise ValueError(f"Unsupported plain SiT sampling method: {method}")

    t_final = times[-1]
    t_model_final = jnp.full(
        (n_sample,),
        1.0 - t_final if flip_time else t_final,
        dtype=sample_dtype,
    )
    final_data = _convert_final_sample_to_data(
        model,
        variable,
        x_final,
        t_model_final,
        labels,
        cfg_scale,
        native_prediction_sampling=native_prediction_sampling,
        prediction_space=prediction_space,
    )
    return final_data.astype(sample_dtype)


def generate_with_initial_noise(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
    sample_idx=None,
):
    """Generate latents and also return the initial Gaussian latents used."""
    del t_min, t_max
    if num_steps < 1:
        raise ValueError(
            f"Plain SiT sampling requires num_steps >= 1, got {num_steps}."
        )

    img_size = config.dataset.image_size
    img_channels = config.dataset.image_channels
    sample_dtype = _get_sampling_dtype(config)
    rng_noise, rng_labels = random.split(rng)
    x = jax.random.normal(
        rng_noise,
        (n_sample, img_size, img_size, img_channels),
        dtype=sample_dtype,
    )
    x_init = x

    labels = _make_random_sample_labels(
        rng_labels,
        n_sample,
        config.dataset.num_classes,
    )

    transport = _create_sampling_transport(config)
    tau0, tau1 = transport.check_interval(
        transport.train_eps,
        transport.sample_eps,
        sde=False,
        eval=True,
        reverse=False,
        last_step_size=0.0,
    )
    times = jnp.linspace(float(tau0), float(tau1), num_steps + 1, dtype=sample_dtype)

    method = str(config.sampling.get("method", "euler")).lower()
    flip_time = bool(config.sampling.get("flip_time", False))
    cfg_scale = jnp.asarray(omega, dtype=sample_dtype)
    native_prediction_sampling = _use_native_prediction_sampling(config)
    prediction_space = _get_native_prediction_space(config)

    def euler_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output = _guided_native_output(
                model,
                variable,
                x_cur,
                t_cur,
                labels,
                cfg_scale,
            )
            drift = _native_prediction_to_drift(
                transport, prediction_space, native_output, x_cur, t_cur
            )
        else:
            drift = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        return (x_cur + dt * drift).astype(sample_dtype)

    def heun_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        dt = tau_next - tau_cur
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output = _guided_native_output(
                model,
                variable,
                x_cur,
                t_cur,
                labels,
                cfg_scale,
            )
            drift = _native_prediction_to_drift(
                transport, prediction_space, native_output, x_cur, t_cur
            )
        else:
            drift = _guided_velocity(model, variable, x_cur, t_cur, labels, cfg_scale)
        x_pred = x_cur + dt * drift
        t_next = jnp.full(
            (n_sample,),
            1.0 - tau_next if flip_time else tau_next,
            dtype=sample_dtype,
        )
        if native_prediction_sampling:
            native_output_next = _guided_native_output(
                model,
                variable,
                x_pred,
                t_next,
                labels,
                cfg_scale,
            )
            drift_next = _native_prediction_to_drift(
                transport, prediction_space, native_output_next, x_pred, t_next
            )
        else:
            drift_next = _guided_velocity(
                model, variable, x_pred, t_next, labels, cfg_scale
            )
        return (x_cur + 0.5 * dt * (drift + drift_next)).astype(sample_dtype)

    def meanflow_step(i, x_cur):
        tau_cur = times[i]
        tau_next = times[i + 1]
        t_cur = jnp.full(
            (n_sample,),
            1.0 - tau_cur if flip_time else tau_cur,
            dtype=sample_dtype,
        )
        r_next = jnp.full(
            (n_sample,),
            1.0 - tau_next if flip_time else tau_next,
            dtype=sample_dtype,
        )
        meanflow = _guided_meanflow(
            model,
            variable,
            x_cur,
            t_cur,
            r_next,
            labels,
            cfg_scale,
        )
        delta_t = (r_next - t_cur).reshape((n_sample, 1, 1, 1))
        return (x_cur + delta_t * meanflow).astype(sample_dtype)

    if _use_interval_meanflow_sampling(config):
        x_final = jax.lax.fori_loop(0, num_steps, meanflow_step, x)
        return x_init.astype(sample_dtype), x_final.astype(sample_dtype)
    if method == "euler":
        x_final = jax.lax.fori_loop(0, num_steps, euler_step, x)
    elif method == "heun":
        x_final = jax.lax.fori_loop(0, num_steps, heun_step, x)
    else:
        raise ValueError(f"Unsupported plain SiT sampling method: {method}")

    t_final = times[-1]
    t_model_final = jnp.full(
        (n_sample,),
        1.0 - t_final if flip_time else t_final,
        dtype=sample_dtype,
    )
    final_data = _convert_final_sample_to_data(
        model,
        variable,
        x_final,
        t_model_final,
        labels,
        cfg_scale,
        native_prediction_sampling=native_prediction_sampling,
        prediction_space=prediction_space,
    )
    return x_init.astype(sample_dtype), final_data.astype(sample_dtype)


def sample_step_with_initial_noise(
    variable,
    sample_idx,
    model,
    rng_init,
    device_batch_size,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
):
    """PMapped sampling step returning initial noise and final latents in BCHW."""
    rng_sample = random.fold_in(rng_init, sample_idx)
    initial_noise, images = generate_with_initial_noise(
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
    return initial_noise.transpose(0, 3, 1, 2), images.transpose(0, 3, 1, 2)


def sample_step(
    variable,
    sample_idx,
    model,
    rng_init,
    device_batch_size,
    config,
    num_steps,
    omega,
    t_min,
    t_max,
):
    """PMapped sampling step returning latents in BCHW format."""
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
