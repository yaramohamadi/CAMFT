"""JAX transport utilities for plain SiT training."""

import enum

import jax
import jax.numpy as jnp


def expand_t_like_x(t, x):
    """Reshape a `[B]` time vector to broadcast over `x`."""
    dims = (1,) * (x.ndim - 1)
    return t.reshape((t.shape[0],) + dims)


def mean_flat(x):
    """Mean over all non-batch dimensions."""
    return jnp.mean(x.reshape((x.shape[0], -1)), axis=1)


class ICPlan:
    """Linear interpolant plan from the official SiT repo."""

    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        t = jnp.asarray(t)
        return t, jnp.ones_like(t)

    def compute_sigma_t(self, t):
        t = jnp.asarray(t)
        return 1.0 - t, -jnp.ones_like(t)

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1.0 / t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        norm = jnp.asarray(norm, dtype=x.dtype)
        if form == "constant":
            return jnp.ones_like(t) * norm
        if form == "SBDM":
            return norm * self.compute_drift(x, t)[1]
        if form == "sigma":
            return norm * self.compute_sigma_t(t)[0]
        if form == "linear":
            return norm * (1.0 - t)
        if form == "decreasing":
            return 0.25 * (norm * jnp.cos(jnp.pi * t) + 1.0) ** 2
        if form == "increasing-decreasing":
            return norm * jnp.sin(jnp.pi * t) ** 2
        raise NotImplementedError(f"Diffusion form {form} not implemented")

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t ** 2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - mean) / var

    def compute_mu_t(self, t, x0, x1):
        t = expand_t_like_x(t, x1)
        alpha_t, _ = self.compute_alpha_t(t)
        sigma_t, _ = self.compute_sigma_t(t)
        return alpha_t * x1 + sigma_t * x0

    def compute_xt(self, t, x0, x1):
        return self.compute_mu_t(t, x0, x1)

    def compute_ut(self, t, x0, x1, xt):
        del xt
        t = expand_t_like_x(t, x1)
        _, d_alpha_t = self.compute_alpha_t(t)
        _, d_sigma_t = self.compute_sigma_t(t)
        return d_alpha_t * x1 + d_sigma_t * x0

    def plan(self, t, x0, x1):
        xt = self.compute_xt(t, x0, x1)
        ut = self.compute_ut(t, x0, x1, xt)
        return t, xt, ut


class VPCPlan(ICPlan):
    """VP interpolant plan from the official SiT repo."""

    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = (
            lambda t: -0.25 * ((1.0 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1.0 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = (
            lambda t: 0.5 * (1.0 - t) * (self.sigma_max - self.sigma_min)
            + 0.5 * self.sigma_min
        )

    def compute_alpha_t(self, t):
        alpha_t = jnp.exp(self.log_mean_coeff(t))
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        p_sigma_t = 2.0 * self.log_mean_coeff(t)
        sigma_t = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(p_sigma_t), 1e-12))
        d_sigma_t = (
            jnp.exp(p_sigma_t) * (2.0 * self.d_log_mean_coeff(t)) / (-2.0 * sigma_t)
        )
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1.0 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2.0


class GVPCPlan(ICPlan):
    """GVP interpolant plan from the official SiT repo."""

    def compute_alpha_t(self, t):
        alpha_t = jnp.sin(t * jnp.pi / 2.0)
        d_alpha_t = jnp.pi / 2.0 * jnp.cos(t * jnp.pi / 2.0)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = jnp.cos(t * jnp.pi / 2.0)
        d_sigma_t = -jnp.pi / 2.0 * jnp.sin(t * jnp.pi / 2.0)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return jnp.pi / (2.0 * jnp.tan(t * jnp.pi / 2.0))


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    """JAX port of the official SiT transport training loss."""

    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: ICPlan,
            PathType.GVP: GVPCPlan,
            PathType.VP: VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0.0
        t1 = 1.0
        eps = train_eps if not eval else sample_eps

        if isinstance(self.path_sampler, VPCPlan):
            t1 = 1.0 - eps if (not sde or last_step_size == 0.0) else 1.0 - last_step_size
        elif isinstance(self.path_sampler, (ICPlan, GVPCPlan)) and (
            self.model_type != ModelType.VELOCITY or sde or eps > 0.0
        ):
            t0 = eps if (
                (diffusion_form == "SBDM" and sde)
                or self.model_type != ModelType.VELOCITY
                or eps > 0.0
            ) else 0.0
            if self.model_type == ModelType.VELOCITY and not sde and eps > 0.0:
                t1 = 1.0
            else:
                t1 = 1.0 - eps if (not sde or last_step_size == 0.0) else 1.0 - last_step_size

        if reverse:
            t0, t1 = 1.0 - t0, 1.0 - t1
        return t0, t1

    def sample(self, x1, rng):
        """Sample `x0` and `t` based on the shape of `x1`."""
        rng_x0, rng_t = jax.random.split(rng)
        x0 = jax.random.normal(rng_x0, x1.shape, dtype=x1.dtype)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = jax.random.uniform(
            rng_t,
            (x1.shape[0],),
            minval=t0,
            maxval=t1,
            dtype=x1.dtype,
        )
        return t, x0, x1

    def training_losses(self, model, x1, rng, model_kwargs=None):
        """Official SiT transport loss."""
        if model_kwargs is None:
            model_kwargs = {}

        t, x0, x1 = self.sample(x1, rng)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)

        if model_output.shape != xt.shape:
            raise ValueError(
                f"Model output shape {model_output.shape} must match xt {xt.shape}."
            )

        terms = {
            "pred": model_output,
            "target": ut,
            "t": t,
        }

        if self.model_type == ModelType.VELOCITY:
            terms["loss"] = mean_flat((model_output - ut) ** 2)
            return terms

        _, drift_var = self.path_sampler.compute_drift(xt, t)
        sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, xt))

        if self.loss_type == WeightType.VELOCITY:
            weight = (drift_var / sigma_t) ** 2
        elif self.loss_type == WeightType.LIKELIHOOD:
            weight = drift_var / (sigma_t ** 2)
        elif self.loss_type == WeightType.NONE:
            weight = 1.0
        else:
            raise NotImplementedError()

        if self.model_type == ModelType.NOISE:
            terms["loss"] = mean_flat(weight * ((model_output - x0) ** 2))
        else:
            terms["loss"] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        return terms


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
):
    """Create a transport object with official SiT defaults."""
    if prediction == "noise":
        model_type = ModelType.NOISE
    elif prediction == "score":
        model_type = ModelType.SCORE
    else:
        model_type = ModelType.VELOCITY

    if loss_weight == "velocity":
        loss_type = WeightType.VELOCITY
    elif loss_weight == "likelihood":
        loss_type = WeightType.LIKELIHOOD
    else:
        loss_type = WeightType.NONE

    path_choice = {
        "Linear": PathType.LINEAR,
        "GVP": PathType.GVP,
        "VP": PathType.VP,
    }
    path_enum = path_choice[path_type]

    if path_enum == PathType.VP:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_enum in (PathType.GVP, PathType.LINEAR) and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0.0 if train_eps is None else train_eps
        sample_eps = 0.0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=path_enum,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
