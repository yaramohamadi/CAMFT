import enum

import numpy as np
import torch

try:
    from torchdiffeq import odeint as _torchdiffeq_odeint
except ImportError:  # pragma: no cover - optional dependency
    _torchdiffeq_odeint = None


def expand_t_like_x(t, x):
    dims = [1] * (len(x.size()) - 1)
    return t.view(t.size(0), *dims)


class ICPlan:
    def __init__(self, sigma=0.0):
        self.sigma = sigma

    def compute_alpha_t(self, t):
        return t, 1

    def compute_sigma_t(self, t):
        return 1 - t, -1

    def compute_d_alpha_alpha_ratio_t(self, t):
        return 1 / t

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        alpha_ratio = self.compute_d_alpha_alpha_ratio_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        drift = alpha_ratio * x
        diffusion = alpha_ratio * (sigma_t ** 2) - sigma_t * d_sigma_t
        return -drift, diffusion

    def compute_diffusion(self, x, t, form="constant", norm=1.0):
        t = expand_t_like_x(t, x)
        choices = {
            "constant": norm,
            "SBDM": norm * self.compute_drift(x, t)[1],
            "sigma": norm * self.compute_sigma_t(t)[0],
            "linear": norm * (1 - t),
            "decreasing": 0.25 * (norm * torch.cos(np.pi * t) + 1) ** 2,
            "increasing-decreasing": norm * torch.sin(np.pi * t) ** 2,
        }
        if form not in choices:
            raise NotImplementedError(f"Diffusion form {form} not implemented")
        return choices[form]

    def get_score_from_velocity(self, velocity, x, t):
        t = expand_t_like_x(t, x)
        alpha_t, d_alpha_t = self.compute_alpha_t(t)
        sigma_t, d_sigma_t = self.compute_sigma_t(t)
        mean = x
        reverse_alpha_ratio = alpha_t / d_alpha_t
        var = sigma_t**2 - reverse_alpha_ratio * d_sigma_t * sigma_t
        return (reverse_alpha_ratio * velocity - mean) / var


class VPCPlan(ICPlan):
    def __init__(self, sigma_min=0.1, sigma_max=20.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.log_mean_coeff = (
            lambda t: -0.25 * ((1 - t) ** 2) * (self.sigma_max - self.sigma_min)
            - 0.5 * (1 - t) * self.sigma_min
        )
        self.d_log_mean_coeff = (
            lambda t: 0.5 * (1 - t) * (self.sigma_max - self.sigma_min)
            + 0.5 * self.sigma_min
        )

    def compute_alpha_t(self, t):
        alpha_t = torch.exp(self.log_mean_coeff(t))
        d_alpha_t = alpha_t * self.d_log_mean_coeff(t)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        p_sigma_t = 2 * self.log_mean_coeff(t)
        sigma_t = torch.sqrt(1 - torch.exp(p_sigma_t))
        d_sigma_t = torch.exp(p_sigma_t) * (2 * self.d_log_mean_coeff(t)) / (-2 * sigma_t)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return self.d_log_mean_coeff(t)

    def compute_drift(self, x, t):
        t = expand_t_like_x(t, x)
        beta_t = self.sigma_min + (1 - t) * (self.sigma_max - self.sigma_min)
        return -0.5 * beta_t * x, beta_t / 2


class GVPCPlan(ICPlan):
    def compute_alpha_t(self, t):
        alpha_t = torch.sin(t * np.pi / 2)
        d_alpha_t = np.pi / 2 * torch.cos(t * np.pi / 2)
        return alpha_t, d_alpha_t

    def compute_sigma_t(self, t):
        sigma_t = torch.cos(t * np.pi / 2)
        d_sigma_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
        return sigma_t, d_sigma_t

    def compute_d_alpha_alpha_ratio_t(self, t):
        return np.pi / (2 * torch.tan(t * np.pi / 2))


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
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if isinstance(self.path_sampler, VPCPlan):
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif isinstance(self.path_sampler, (ICPlan, GVPCPlan)) and (
            self.model_type != ModelType.VELOCITY or sde
        ):
            t0 = eps if ((diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY) else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def get_drift(self):
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            if model_output.shape != x.shape:
                raise ValueError("Output shape from ODE solver must match input shape")
            return model_output

        return body_fn

    def get_score(self):
        if self.model_type == ModelType.NOISE:
            return lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(expand_t_like_x(t, x))[0]
        if self.model_type == ModelType.SCORE:
            return lambda x, t, model, **kwargs: model(x, t, **kwargs)
        if self.model_type == ModelType.VELOCITY:
            return lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        raise NotImplementedError()


class _ODE:
    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol):
        self.drift = drift
        self.t = torch.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def _fixed_step_sample(self, x, model, **model_kwargs):
        xs = []
        cur = x
        for idx in range(len(self.t) - 1):
            t_cur = self.t[idx]
            t_next = self.t[idx + 1]
            dt = t_next - t_cur
            t_batch = torch.ones(cur.size(0), device=cur.device) * t_cur
            if self.sampler_type == "euler":
                cur = cur + dt * self.drift(cur, t_batch, model, **model_kwargs)
            elif self.sampler_type == "heun":
                k1 = self.drift(cur, t_batch, model, **model_kwargs)
                pred = cur + dt * k1
                t_next_batch = torch.ones(cur.size(0), device=cur.device) * t_next
                k2 = self.drift(pred, t_next_batch, model, **model_kwargs)
                cur = cur + 0.5 * dt * (k1 + k2)
            else:
                raise NotImplementedError(f"Unsupported fixed-step ODE sampler: {self.sampler_type}")
            xs.append(cur)
        return xs

    def sample(self, x, model, **model_kwargs):
        method = self.sampler_type.lower()
        if method in {"euler", "heun"}:
            return self._fixed_step_sample(x, model, **model_kwargs)
        if _torchdiffeq_odeint is None:
            raise ImportError(
                "torchdiffeq is required for adaptive ODE solvers such as dopri5. "
                "Install torchdiffeq or use --sampling-method euler/heun."
            )

        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, state):
            batch = state[0].size(0) if isinstance(state, tuple) else state.size(0)
            t_batch = torch.ones(batch, device=device) * t
            return self.drift(state, t_batch, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return _torchdiffeq_odeint(_fn, x, t, method=method, atol=atol, rtol=rtol)


class _SDE:
    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        if not t0 < t1:
            raise ValueError("SDE sampler has to be in forward time")
        self.num_timesteps = num_steps
        self.t = torch.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def _euler_maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = torch.randn(x.size(), device=x.device)
        t_batch = torch.ones(x.size(0), device=x.device) * t
        dw = w_cur * torch.sqrt(self.dt)
        drift = self.drift(x, t_batch, model, **model_kwargs)
        diffusion = self.diffusion(x, t_batch)
        mean_x = x + drift * self.dt
        x = mean_x + torch.sqrt(2 * diffusion) * dw
        return x, mean_x

    def _heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = torch.randn(x.size(), device=x.device)
        dw = w_cur * torch.sqrt(self.dt)
        t_cur = torch.ones(x.size(0), device=x.device) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + torch.sqrt(2 * diffusion) * dw
        k1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * k1
        k2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (k1 + k2), xhat

    def sample(self, init, model, **model_kwargs):
        x = init
        mean_x = init
        samples = []
        step_fn = {
            "Euler": self._euler_maruyama_step,
            "Heun": self._heun_step,
        }.get(self.sampler_type)
        if step_fn is None:
            raise NotImplementedError(f"SDE sampler {self.sampler_type} not implemented")
        for ti in self.t[:-1]:
            with torch.no_grad():
                x, mean_x = step_fn(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def _get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(
                x, t, form=diffusion_form, norm=diffusion_norm
            )

        sde_drift = (
            lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs)
            + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
        )
        return sde_drift, diffusion_fn

    def _get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            return lambda x, t, model, **model_kwargs: x
        if last_step == "Mean":
            return lambda x, t, model, **model_kwargs: x + sde_drift(
                x, t, model, **model_kwargs
            ) * last_step_size
        if last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (
                sigma(t)[0][0] ** 2
            ) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        if last_step == "Euler":
            return lambda x, t, model, **model_kwargs: x + self.drift(
                x, t, model, **model_kwargs
            ) * last_step_size
        raise NotImplementedError()

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        if last_step is None:
            last_step_size = 0.0
        sde_drift, sde_diffusion = self._get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )
        sde_solver = _SDE(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )
        last_step_fn = self._get_last_step(
            sde_drift,
            last_step=last_step,
            last_step_size=last_step_size,
        )

        def _sample(init, model, **model_kwargs):
            xs = sde_solver.sample(init, model, **model_kwargs)
            ts = torch.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            return xs

        return _sample

    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )
        ode_solver = _ODE(
            drift=self.drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        return ode_solver.sample


def create_transport(
    path_type="Linear",
    prediction="velocity",
    loss_weight=None,
    train_eps=None,
    sample_eps=None,
):
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

    if path_enum in [PathType.VP]:
        train_eps = 1e-5 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    elif path_enum in [PathType.GVP, PathType.LINEAR] and model_type != ModelType.VELOCITY:
        train_eps = 1e-3 if train_eps is None else train_eps
        sample_eps = 1e-3 if sample_eps is None else sample_eps
    else:
        train_eps = 0 if train_eps is None else train_eps
        sample_eps = 0 if sample_eps is None else sample_eps

    return Transport(
        model_type=model_type,
        path_type=path_enum,
        loss_type=loss_type,
        train_eps=train_eps,
        sample_eps=sample_eps,
    )
