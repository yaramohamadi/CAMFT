"""JAX implementation of the original DiT Gaussian diffusion objective/sampler."""

import enum
import math

import jax
import jax.numpy as jnp
import numpy as np


def mean_flat(x):
    return jnp.mean(x, axis=tuple(range(1, x.ndim)))


class ModelMeanType(enum.Enum):
    PREVIOUS_X = enum.auto()
    START_X = enum.auto()
    EPSILON = enum.auto()


class ModelVarType(enum.Enum):
    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()
    RESCALED_MSE = enum.auto()
    KL = enum.auto()
    RESCALED_KL = enum.auto()


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + jnp.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * jnp.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    return 0.5 * (
        1.0 + jnp.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * jnp.power(x, 3)))
    )


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    centered_x = x - means
    inv_stdv = jnp.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = jnp.log(jnp.clip(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = jnp.log(jnp.clip(1.0 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    return jnp.where(
        x < -0.999,
        log_cdf_plus,
        jnp.where(
            x > 0.999,
            log_one_minus_cdf_min,
            jnp.log(jnp.clip(cdf_delta, min=1e-12)),
        ),
    )


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    if beta_schedule != "linear":
        raise NotImplementedError(beta_schedule)
    return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    if schedule_name == "linear":
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    if schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = jnp.asarray(arr, dtype=jnp.float32)[timesteps]
    while res.ndim < len(broadcast_shape):
        res = res[..., None]
    return res + jnp.zeros(broadcast_shape, dtype=jnp.float32)


class GaussianDiffusion:
    def __init__(self, *, betas, model_mean_type, model_var_type, loss_type):
        betas = np.array(betas, dtype=np.float64)
        if len(betas.shape) != 1:
            raise ValueError("betas must be 1-D")
        if not ((betas > 0).all() and (betas <= 1).all()):
            raise ValueError("betas must be in (0, 1]")

        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.betas = betas
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        if self.posterior_variance.shape[0] == 1:
            self.posterior_log_variance_clipped = np.log(
                np.maximum(self.posterior_variance, 1e-20)
            )
        else:
            self.posterior_log_variance_clipped = np.log(
                np.append(self.posterior_variance[1], self.posterior_variance[1:])
            )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise):
        shape = x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        shape = x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(self, x_t, t, eps):
        shape = x_t.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        shape = x_t.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, shape)

    def alpha_sigma(self, x_t, t):
        shape = x_t.shape
        alpha_t = _extract_into_tensor(self.sqrt_alphas_cumprod, t, shape)
        sigma_t = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, shape)
        return alpha_t, sigma_t

    def predict_xstart_from_eps(self, x_t, t, eps):
        return self._predict_xstart_from_eps(x_t, t, eps)

    def predict_velocity_from_eps(self, x_t, t, eps):
        alpha_t, sigma_t = self.alpha_sigma(x_t, t)
        pred_xstart = self._predict_xstart_from_eps(x_t, t, eps)
        return alpha_t * eps - sigma_t * pred_xstart

    def predict_eps_from_velocity(self, x_t, t, velocity):
        alpha_t, sigma_t = self.alpha_sigma(x_t, t)
        return sigma_t * x_t + alpha_t * velocity

    def p_mean_variance(self, model_fn, x, t, clip_denoised=True):
        x = x.astype(jnp.float32)
        model_output = model_fn(x, t).astype(jnp.float32)
        batch, height, width, channels = x.shape

        if self.model_var_type in (ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE):
            expected = (batch, height, width, channels * 2)
            if model_output.shape != expected:
                raise ValueError(f"expected model output {expected}, got {model_output.shape}")
            model_output, model_var_values = jnp.split(model_output, 2, axis=-1)
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            frac = (model_var_values + 1.0) / 2.0
            model_log_variance = frac * max_log + (1.0 - frac) * min_log
            model_variance = jnp.exp(model_log_variance)
        else:
            raise NotImplementedError("Only learned-range variance is used for DiT.")

        if self.model_mean_type == ModelMeanType.START_X:
            pred_xstart = model_output
        elif self.model_mean_type == ModelMeanType.EPSILON:
            pred_xstart = self._predict_xstart_from_eps(x, t, model_output)
        else:
            raise NotImplementedError(self.model_mean_type)

        if clip_denoised:
            pred_xstart = jnp.clip(pred_xstart, -1.0, 1.0)
        model_mean, _, _ = self.q_posterior_mean_variance(pred_xstart, x, t)
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample(self, model_fn, x, t, rng, clip_denoised=True):
        out = self.p_mean_variance(model_fn, x, t, clip_denoised=clip_denoised)
        noise = jax.random.normal(rng, x.shape, dtype=jnp.float32)
        nonzero_mask = (t != 0).astype(jnp.float32).reshape(
            (-1,) + (1,) * (x.ndim - 1)
        )
        sample = out["mean"] + nonzero_mask * jnp.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(self, model_fn, noise, rng, clip_denoised=False, dtype=jnp.float32):
        def loop_body(i, carry):
            img, key = carry
            key, step_key = jax.random.split(key)
            step_idx = self.num_timesteps - 1 - i
            t = jnp.full((img.shape[0],), step_idx, dtype=jnp.int32)
            out = self.p_sample(
                model_fn,
                img,
                t,
                step_key,
                clip_denoised=clip_denoised,
            )
            return out["sample"].astype(dtype), key

        final_img, _ = jax.lax.fori_loop(
            0,
            self.num_timesteps,
            loop_body,
            (noise.astype(dtype), rng),
        )
        return final_img

    def _vb_terms_bpd(self, model_fn, x_start, x_t, t, clip_denoised=True):
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start,
            x_t=x_t,
            t=t,
        )
        out = self.p_mean_variance(
            model_fn,
            x_t,
            t,
            clip_denoised=clip_denoised,
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out["mean"], out["log_variance"])
        kl = mean_flat(kl) / np.log(2.0)
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start,
            means=out["mean"],
            log_scales=0.5 * out["log_variance"],
        )
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)
        output = jnp.where(t == 0, decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model_fn, x_start, t, rng):
        x_start = x_start.astype(jnp.float32)
        noise = jax.random.normal(rng, x_start.shape, dtype=jnp.float32)
        x_t = self.q_sample(x_start, t, noise=noise)

        if self.loss_type not in (LossType.MSE, LossType.RESCALED_MSE):
            raise NotImplementedError(self.loss_type)

        model_output = model_fn(x_t, t).astype(jnp.float32)
        batch, height, width, channels = x_t.shape
        if model_output.shape != (batch, height, width, channels * 2):
            raise ValueError(
                f"expected model output {(batch, height, width, channels * 2)}, "
                f"got {model_output.shape}"
            )

        model_output, model_var_values = jnp.split(model_output, 2, axis=-1)
        frozen_out = jnp.concatenate(
            [jax.lax.stop_gradient(model_output), model_var_values],
            axis=-1,
        )
        vb = self._vb_terms_bpd(
            model_fn=lambda _x, _t: frozen_out,
            x_start=x_start,
            x_t=x_t,
            t=t,
            clip_denoised=False,
        )["output"]
        if self.loss_type == LossType.RESCALED_MSE:
            vb = vb * (self.num_timesteps / 1000.0)

        if self.model_mean_type == ModelMeanType.EPSILON:
            target = noise
        elif self.model_mean_type == ModelMeanType.START_X:
            target = x_start
        else:
            target = self.q_posterior_mean_variance(x_start, x_t, t)[0]

        mse = mean_flat((target - model_output) ** 2)
        return {"loss": mse + vb, "mse": mse, "vb": vb, "t": t}


def space_timesteps(num_timesteps, section_counts):
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for stride in range(1, num_timesteps):
                if len(range(0, num_timesteps, stride)) == desired_count:
                    return set(range(0, num_timesteps, stride))
            raise ValueError(
                f"cannot create exactly {desired_count} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        frac_stride = 1 if section_count <= 1 else (size - 1) / (section_count - 1)
        cur_idx = 0.0
        for _ in range(section_count):
            all_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def _wrap_model(self, model_fn):
        if isinstance(model_fn, _WrappedModel):
            return model_fn
        return _WrappedModel(model_fn, self.timestep_map)

    def p_mean_variance(self, model_fn, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model_fn), *args, **kwargs)

    def training_losses(self, model_fn, *args, **kwargs):
        return super().training_losses(self._wrap_model(model_fn), *args, **kwargs)


class _WrappedModel:
    def __init__(self, model_fn, timestep_map):
        self.model_fn = model_fn
        self.timestep_map = tuple(timestep_map)

    def __call__(self, x, t):
        map_tensor = jnp.asarray(self.timestep_map, dtype=jnp.int32)
        return self.model_fn(x, map_tensor[t])


def create_diffusion(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    learn_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
):
    del sigma_small
    if not learn_sigma:
        raise ValueError("The original DiT-XL/2 checkpoint expects learn_sigma=True.")
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON if not predict_xstart else ModelMeanType.START_X,
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=loss_type,
    )
