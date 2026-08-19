"""Sampling utilities for the JAX DiT diffusion path."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from utils.dit_diffusion import create_diffusion, space_timesteps


def get_default_cfg_scale(config):
    """Return the config-level DiT classifier-free guidance scale.

    The project uses `sampling.omega` as the common guidance-scale name across
    SiT and DiT. `sampling.cfg_scale` is accepted as a DiT-compatible fallback.
    """
    sampling = config.sampling
    has_omega = "omega" in sampling
    has_cfg_scale = "cfg_scale" in sampling
    if has_omega and has_cfg_scale:
        omega = float(sampling.omega)
        cfg_scale = float(sampling.cfg_scale)
        if abs(omega - cfg_scale) > 1e-8:
            raise ValueError(
                "DiT config has both sampling.omega and sampling.cfg_scale, "
                f"but they differ: omega={omega}, cfg_scale={cfg_scale}. "
                "Set them to the same value or remove one to avoid ambiguous guidance."
            )
        return omega
    if has_omega:
        return float(sampling.omega)
    if has_cfg_scale:
        return float(sampling.cfg_scale)
    return 1.0


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
    raise ValueError(f"Unsupported half precision dtype: {dtype_name}")


def _make_sample_labels(num_samples, num_classes, sample_idx=None):
    if sample_idx is not None:
        labels = jnp.arange(num_samples, dtype=jnp.int32)
        labels = labels + jnp.asarray(sample_idx, dtype=jnp.int32) * num_samples
        return labels % num_classes
    return random.randint(
        random.PRNGKey(0), (num_samples,), 0, num_classes, dtype=jnp.int32
    )


def _guided_model_output(model, variable, x, t, labels, cfg_scale):
    num_samples = x.shape[0]

    def unwrap_prediction(model_output, x_input, t_input):
        raw_prediction, model_var_values = jnp.split(model_output, 2, axis=-1)
        eps_prediction = model._unwrap_prediction(
            x_input,
            t_input.astype(jnp.int32),
            raw_prediction,
        )
        return jnp.concatenate([eps_prediction, model_var_values], axis=-1)

    def conditional(_):
        out = model.apply(variable, x, t.astype(jnp.float32), labels)
        return unwrap_prediction(out, x, t)

    def guided(_):
        null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)
        x_cat = jnp.concatenate([x, x], axis=0)
        t_cat = jnp.concatenate([t, t], axis=0)
        y_cat = jnp.concatenate([labels, null_labels], axis=0)
        out = model.apply(variable, x_cat, t_cat.astype(jnp.float32), y_cat)
        cond, uncond = jnp.split(out, 2, axis=0)
        prediction_c, model_var_values_c = jnp.split(cond, 2, axis=-1)
        prediction_u, _ = jnp.split(uncond, 2, axis=-1)
        # Apply CFG to every latent channel. Restricting it to the first three
        # channels is only appropriate for RGB pixel-space outputs, not 4-channel
        # DiT latent predictions.
        guided_wrapped = prediction_u + cfg_scale * (prediction_c - prediction_u)
        guided_out = jnp.concatenate([guided_wrapped, model_var_values_c], axis=-1)
        return unwrap_prediction(guided_out, x, t)

    return jax.lax.cond(jnp.equal(cfg_scale, 1.0), conditional, guided, operand=None)


def _guided_eps_prediction(model, variable, x, t, labels, cfg_scale):
    model_output = _guided_model_output(model, variable, x, t, labels, cfg_scale)
    eps_prediction, _ = jnp.split(model_output, 2, axis=-1)
    return eps_prediction


def _unwrap_eps_prediction(model, x_input, t_input, model_output):
    raw_prediction, _ = jnp.split(model_output, 2, axis=-1)
    return model._unwrap_prediction(
        x_input,
        t_input.astype(jnp.int32),
        raw_prediction,
    )


def _conditional_eps_prediction(model, variable, x, t, labels):
    model_output = model.apply(variable, x, t.astype(jnp.float32), labels)
    return _unwrap_eps_prediction(model, x, t, model_output)


def _conditional_unconditional_eps_predictions(model, variable, x, t, labels):
    num_samples = x.shape[0]
    null_labels = jnp.full((num_samples,), model.num_classes, dtype=jnp.int32)
    x_cat = jnp.concatenate([x, x], axis=0)
    t_cat = jnp.concatenate([t, t], axis=0)
    y_cat = jnp.concatenate([labels, null_labels], axis=0)
    out = model.apply(variable, x_cat, t_cat.astype(jnp.float32), y_cat)
    cond, uncond = jnp.split(out, 2, axis=0)
    cond_eps = _unwrap_eps_prediction(model, x, t, cond)
    uncond_eps = _unwrap_eps_prediction(model, x, t, uncond)
    return cond_eps, uncond_eps


def _broadcast_schedule_value(value, x):
    value = jnp.asarray(value, dtype=jnp.float32)
    while value.ndim < x.ndim:
        value = value[..., None]
    return value + jnp.zeros(x.shape, dtype=jnp.float32)


def _build_native_velocity_schedule(num_steps, diffusion_steps):
    if num_steps < 1:
        raise ValueError(
            f"DiT native-velocity sampling requires num_steps >= 1, got {num_steps}."
        )
    # Euler integration needs both endpoints so that num_steps matches NFEs.
    schedule = sorted(space_timesteps(diffusion_steps, [num_steps + 1]))
    return np.asarray(schedule, dtype=np.int32)


def _sample_with_native_velocity(
    model,
    variable,
    noise,
    labels,
    cfg_scale,
    *,
    num_steps,
    config,
):
    diffusion_steps = int(config.diffusion.diffusion_steps)
    base_diffusion = create_diffusion(
        "",
        noise_schedule=config.diffusion.noise_schedule,
        learn_sigma=config.diffusion.learn_sigma,
        predict_xstart=config.diffusion.predict_xstart,
        rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        diffusion_steps=diffusion_steps,
    )
    schedule = _build_native_velocity_schedule(num_steps, diffusion_steps)
    schedule_t = jnp.asarray(schedule, dtype=jnp.int32)
    tau = schedule_t.astype(jnp.float32) / jnp.maximum(diffusion_steps - 1, 1)
    alpha = jnp.asarray(base_diffusion.sqrt_alphas_cumprod[schedule], dtype=jnp.float32)
    sigma = jnp.asarray(
        base_diffusion.sqrt_one_minus_alphas_cumprod[schedule],
        dtype=jnp.float32,
    )
    x = noise.astype(jnp.float32)
    alpha_eps = jnp.asarray(1e-6, dtype=jnp.float32)
    num_intervals = schedule_t.shape[0] - 1
    native_velocity_cfg_space = str(
        config.sampling.get("native_velocity_cfg_space", "epsilon")
    ).lower()
    native_velocity_derivative_mode = str(
        config.sampling.get("native_velocity_derivative_mode", "finite_difference")
    ).lower()
    native_velocity_sigma_clamp = jnp.asarray(
        float(config.sampling.get("native_velocity_sigma_clamp", 1e-6)),
        dtype=jnp.float32,
    )
    if native_velocity_cfg_space not in {"epsilon", "velocity"}:
        raise ValueError(
            "native_velocity_cfg_space must be 'epsilon' or 'velocity', got "
            f"{native_velocity_cfg_space!r}."
        )
    if native_velocity_derivative_mode not in {"finite_difference", "analytic"}:
        raise ValueError(
            "native_velocity_derivative_mode must be 'finite_difference' or "
            f"'analytic', got {native_velocity_derivative_mode!r}."
        )

    beta_start = jnp.asarray(1e-4, dtype=jnp.float32)
    beta_end = jnp.asarray(2e-2, dtype=jnp.float32)
    time_scale = jnp.asarray(max(diffusion_steps - 1, 1), dtype=jnp.float32)

    def eps_to_velocity(x_t, eps_hat, current_pos, next_pos):
        alpha_t = _broadcast_schedule_value(alpha[current_pos], x_t)
        sigma_t = _broadcast_schedule_value(sigma[current_pos], x_t)
        dt = tau[next_pos] - tau[current_pos]
        if native_velocity_derivative_mode == "analytic":
            tau_cur = tau[current_pos]
            # `tau` is the normalized discrete index t / (T - 1), so
            # d/dtau = (T - 1) d/dt_index. Without this factor the analytic
            # velocity is under-scaled by ~1000x for a 1000-step DiT schedule.
            beta_tau = time_scale * (beta_start + (beta_end - beta_start) * tau_cur)
            alpha_scalar = alpha[current_pos]
            sigma_scalar = sigma[current_pos]
            sigma_safe = jnp.maximum(sigma_scalar, native_velocity_sigma_clamp)
            alpha_dot = -0.5 * beta_tau * alpha_scalar
            sigma_dot = 0.5 * beta_tau * (alpha_scalar ** 2) / sigma_safe
        else:
            alpha_dot = (alpha[next_pos] - alpha[current_pos]) / dt
            sigma_dot = (sigma[next_pos] - sigma[current_pos]) / dt
        alpha_dot_t = _broadcast_schedule_value(alpha_dot, x_t)
        sigma_dot_t = _broadcast_schedule_value(sigma_dot, x_t)
        x0_hat = (x_t - sigma_t * eps_hat) / jnp.maximum(alpha_t, alpha_eps)
        v_hat = alpha_dot_t * x0_hat + sigma_dot_t * eps_hat
        return v_hat, dt

    def step_fn(i, x_t):
        current_pos = num_intervals - i
        next_pos = current_pos - 1

        model_t = jnp.full((x_t.shape[0],), schedule_t[current_pos], dtype=jnp.int32)

        if native_velocity_cfg_space == "velocity":
            cond_eps, uncond_eps = _conditional_unconditional_eps_predictions(
                model,
                variable,
                x_t,
                model_t,
                labels,
            )
            cond_v, dt = eps_to_velocity(
                x_t,
                cond_eps.astype(jnp.float32),
                current_pos,
                next_pos,
            )
            uncond_v, _ = eps_to_velocity(
                x_t,
                uncond_eps.astype(jnp.float32),
                current_pos,
                next_pos,
            )
            v_hat = uncond_v + cfg_scale.astype(jnp.float32) * (cond_v - uncond_v)
        else:
            eps_hat = _guided_eps_prediction(
                model,
                variable,
                x_t,
                model_t,
                labels,
                cfg_scale,
            ).astype(jnp.float32)
            v_hat, dt = eps_to_velocity(x_t, eps_hat, current_pos, next_pos)

        return x_t + _broadcast_schedule_value(dt, x_t) * v_hat

    return jax.lax.fori_loop(0, num_intervals, step_fn, x)


def _sample_with_transport_velocity(
    model,
    variable,
    noise,
    labels,
    cfg_scale,
    *,
    num_steps,
    config,
):
    diffusion_steps = int(config.diffusion.diffusion_steps)
    base_diffusion = create_diffusion(
        "",
        noise_schedule=config.diffusion.noise_schedule,
        learn_sigma=config.diffusion.learn_sigma,
        predict_xstart=config.diffusion.predict_xstart,
        rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        diffusion_steps=diffusion_steps,
    )
    ddpm_alpha = jnp.asarray(base_diffusion.sqrt_alphas_cumprod, dtype=jnp.float32)
    ddpm_sigma = jnp.asarray(
        base_diffusion.sqrt_one_minus_alphas_cumprod,
        dtype=jnp.float32,
    )
    eps_value = float(config.sampling.get("transport_velocity_eps", 1e-3))
    if eps_value <= 0.0 or eps_value >= 1.0:
        raise ValueError(
            "sampling.transport_velocity_eps must be in (0, 1), got "
            f"{eps_value}."
        )
    eps = jnp.asarray(eps_value, dtype=jnp.float32)

    cfg_space = str(
        config.sampling.get(
            "transport_velocity_cfg_space",
            config.sampling.get("native_velocity_cfg_space", "velocity"),
        )
    ).lower()
    if cfg_space not in {"epsilon", "velocity"}:
        raise ValueError(
            "transport_velocity_cfg_space must be 'epsilon' or 'velocity', got "
            f"{cfg_space!r}."
        )

    time_map = str(
        config.sampling.get("transport_velocity_time_map", "noise_ratio")
    ).lower()
    if time_map not in {"noise_ratio", "flipped_linear", "linear", "diff2flow"}:
        raise ValueError(
            "transport_velocity_time_map must be 'noise_ratio', "
            f"'flipped_linear', 'linear', or 'diff2flow', got {time_map!r}."
        )

    scale_input = bool(config.sampling.get("transport_velocity_scale_input", True))
    x = noise.astype(jnp.float32)
    t_steps = jnp.linspace(eps, 1.0, num_steps + 1, dtype=jnp.float32)
    ddpm_log_noise_ratio = jnp.log(jnp.maximum(ddpm_sigma, eps)) - jnp.log(
        jnp.maximum(ddpm_alpha, eps)
    )
    ddpm_time_scale = jnp.asarray(max(diffusion_steps - 1, 1), dtype=jnp.float32)
    ddpm_indices = jnp.arange(diffusion_steps, dtype=jnp.float32)
    diff2flow_t_fm = ddpm_alpha / jnp.maximum(ddpm_alpha + ddpm_sigma, eps)
    diff2flow_t_fm_asc = diff2flow_t_fm[::-1]
    ddpm_indices_asc = ddpm_indices[::-1]
    ddpm_alpha_asc = ddpm_alpha[::-1]
    ddpm_sigma_asc = ddpm_sigma[::-1]

    def mapped_ddpm_timestep(t_linear, alpha_t, sigma_t):
        if time_map == "noise_ratio":
            target_log_noise_ratio = jnp.log(jnp.maximum(sigma_t, eps)) - jnp.log(
                jnp.maximum(alpha_t, eps)
            )
            return jnp.argmin(
                jnp.abs(ddpm_log_noise_ratio - target_log_noise_ratio)
            ).astype(jnp.int32)
        if time_map == "flipped_linear":
            tau = 1.0 - t_linear
        else:
            tau = t_linear
        return jnp.clip(
            jnp.rint(tau * ddpm_time_scale).astype(jnp.int32),
            0,
            diffusion_steps - 1,
        )

    def diff2flow_alignment(x_t, t_linear):
        t_query = jnp.clip(
            t_linear,
            diff2flow_t_fm_asc[0],
            diff2flow_t_fm_asc[-1],
        )
        tau_model = jnp.interp(t_query, diff2flow_t_fm_asc, ddpm_indices_asc)
        alpha_tau = jnp.interp(t_query, diff2flow_t_fm_asc, ddpm_alpha_asc)
        sigma_tau = jnp.interp(t_query, diff2flow_t_fm_asc, ddpm_sigma_asc)
        # Diff2Flow aligns both the DDPM time and the DDPM state before the
        # source epsilon model is queried, rather than only matching a timestep.
        model_x = _broadcast_schedule_value(alpha_tau + sigma_tau, x_t) * x_t
        model_t = jnp.full((x_t.shape[0],), tau_model, dtype=jnp.float32)
        return model_x, model_t, alpha_tau, sigma_tau

    def prepare_model_input(x_t, alpha_t, sigma_t):
        if not scale_input:
            return x_t
        path_norm = jnp.sqrt(jnp.maximum(alpha_t ** 2 + sigma_t ** 2, eps))
        return x_t / path_norm

    def eps_to_linear_velocity(x_t, eps_hat, alpha_t, sigma_t):
        alpha_b = _broadcast_schedule_value(alpha_t, x_t)
        sigma_b = _broadcast_schedule_value(sigma_t, x_t)
        x0_hat = (x_t - sigma_b * eps_hat) / jnp.maximum(alpha_b, eps)
        return x0_hat - eps_hat

    def step_fn(i, x_t):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        alpha_t = t_cur
        sigma_t = 1.0 - t_cur
        dt = t_next - t_cur
        if time_map == "diff2flow":
            model_x, model_t, velocity_alpha, velocity_sigma = diff2flow_alignment(
                x_t, t_cur
            )
            velocity_x = model_x
        else:
            model_x = prepare_model_input(x_t, alpha_t, sigma_t)
            model_t_idx = mapped_ddpm_timestep(t_cur, alpha_t, sigma_t)
            model_t = jnp.full((x_t.shape[0],), model_t_idx, dtype=jnp.int32)
            velocity_x = x_t
            velocity_alpha = alpha_t
            velocity_sigma = sigma_t

        if cfg_space == "velocity":
            cond_eps, uncond_eps = _conditional_unconditional_eps_predictions(
                model,
                variable,
                model_x,
                model_t,
                labels,
            )
            cond_v = eps_to_linear_velocity(
                velocity_x,
                cond_eps.astype(jnp.float32),
                velocity_alpha,
                velocity_sigma,
            )
            uncond_v = eps_to_linear_velocity(
                velocity_x,
                uncond_eps.astype(jnp.float32),
                velocity_alpha,
                velocity_sigma,
            )
            v_hat = uncond_v + cfg_scale.astype(jnp.float32) * (cond_v - uncond_v)
        else:
            eps_hat = _guided_eps_prediction(
                model,
                variable,
                model_x,
                model_t,
                labels,
                cfg_scale,
            ).astype(jnp.float32)
            v_hat = eps_to_linear_velocity(
                velocity_x,
                eps_hat,
                velocity_alpha,
                velocity_sigma,
            )

        return x_t + _broadcast_schedule_value(dt, x_t) * v_hat

    return jax.lax.fori_loop(0, num_steps, step_fn, x)


def generate(
    variable,
    model,
    rng,
    n_sample,
    config,
    num_steps,
    omega,
    t_min=None,
    t_max=None,
    sample_idx=None,
):
    del t_min, t_max
    if num_steps < 1:
        raise ValueError(f"DiT sampling requires num_steps >= 1, got {num_steps}.")

    img_size = int(config.dataset.image_size)
    img_channels = int(config.dataset.image_channels)
    sample_dtype = _get_sampling_dtype(config)
    noise = random.normal(
        rng,
        (n_sample, img_size, img_size, img_channels),
        dtype=sample_dtype,
    )
    labels = _make_sample_labels(
        n_sample,
        int(config.dataset.num_classes),
        sample_idx=sample_idx,
    )
    cfg_scale_value = omega
    if cfg_scale_value is None:
        cfg_scale_value = get_default_cfg_scale(config)
    cfg_scale = jnp.asarray(cfg_scale_value, dtype=sample_dtype)
    sampling_method = str(config.sampling.get("method", "p_sample"))

    if sampling_method == "native_velocity":
        return _sample_with_native_velocity(
            model,
            variable,
            noise,
            labels,
            cfg_scale,
            num_steps=num_steps,
            config=config,
        ).astype(sample_dtype)

    if sampling_method == "transport_velocity":
        return _sample_with_transport_velocity(
            model,
            variable,
            noise,
            labels,
            cfg_scale,
            num_steps=num_steps,
            config=config,
        ).astype(sample_dtype)

    diffusion = create_diffusion(
        str(num_steps),
        noise_schedule=config.diffusion.noise_schedule,
        learn_sigma=config.diffusion.learn_sigma,
        predict_xstart=config.diffusion.predict_xstart,
        rescale_learned_sigmas=config.diffusion.rescale_learned_sigmas,
        diffusion_steps=config.diffusion.diffusion_steps,
    )

    def model_fn(x, t):
        return _guided_model_output(model, variable, x, t, labels, cfg_scale)

    return diffusion.p_sample_loop(
        model_fn,
        noise,
        rng=random.fold_in(rng, 1),
        clip_denoised=False,
        dtype=sample_dtype,
    )


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
