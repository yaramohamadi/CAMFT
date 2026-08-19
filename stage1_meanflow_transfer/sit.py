"""Plain SiT wrapper for official transport-style training."""

import jax
import jax.numpy as jnp
import flax.linen as nn

from models import imfDiT
from utils.dit_diffusion import create_diffusion
from utils.sit_transport_jax import create_transport


class PlainSiT(nn.Module):
    """Dedicated plain SiT training wrapper around the exact Flax SiT backbone."""

    model_str: str
    dtype: jnp.dtype = jnp.float32
    num_classes: int = 1000
    class_dropout_prob: float = 0.1
    target_use_null_class: bool = True
    path_type: str = "Linear"
    prediction: str = "velocity"
    loss_weight: str = None
    train_eps: float = None
    sample_eps: float = None
    objective: str = "sit"
    path_power_k: float = 1.0
    P_mean: float = -0.4
    P_std: float = 1.0
    data_proportion: float = 0.5
    output_prediction_space: str = "velocity"
    velocity_map_mode: str = "transport"
    input_alignment_mode: str = "none"
    native_velocity_derivative_mode: str = "finite_difference"
    native_diffusion_steps: int = 1000
    native_noise_schedule: str = "linear"
    wrapper_eps: float = 1e-6
    wrapped_loss_weight: str = "none"
    model_time_scale: float = 1.0
    model_time_flip: bool = False
    eval: bool = False

    def setup(self):
        self._validate_output_prediction_space()
        self._validate_wrapper_configuration()
        if not (
            self.model_str.startswith("flaxSiT")
            or self.model_str.startswith("flaxDiT")
        ):
            raise ValueError(
                "PlainSiT expects a flaxSiT_* or flaxDiT_* backbone, got "
                f"{self.model_str!r}."
            )

        net_fn = getattr(imfDiT, self.model_str)
        self.net = net_fn(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            use_r_conditioning=(self.objective == "power_meanflow"),
            eval=self.eval,
        )
        self.transport = create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=self.train_eps,
            sample_eps=self.sample_eps,
        )
        self.transport = self._maybe_adjust_transport_eps(self.transport)
        if self.objective not in {"sit", "power_meanflow"}:
            raise ValueError(
                "PlainSiT objective must be one of ['sit', 'power_meanflow'], got "
                f"{self.objective!r}."
            )
        if self._needs_native_diffusion_schedule():
            native_diffusion = create_diffusion(
                "",
                noise_schedule=self.native_noise_schedule,
                learn_sigma=True,
                predict_xstart=False,
                rescale_learned_sigmas=False,
                diffusion_steps=self.native_diffusion_steps,
            )
            self.native_alpha = jnp.asarray(
                native_diffusion.sqrt_alphas_cumprod, dtype=jnp.float32
            )
            self.native_sigma = jnp.asarray(
                native_diffusion.sqrt_one_minus_alphas_cumprod, dtype=jnp.float32
            )
            self.native_time_scale = jnp.asarray(
                max(int(self.native_diffusion_steps) - 1, 1), dtype=jnp.float32
            )
            self.native_tau = (
                jnp.arange(int(self.native_diffusion_steps), dtype=jnp.float32)
                / self.native_time_scale
            )
            diff2flow_t_fm = self.native_alpha / jnp.maximum(
                self.native_alpha + self.native_sigma, self.wrapper_eps
            )
            self.diff2flow_t_fm_asc = diff2flow_t_fm[::-1]
            self.native_indices_asc = jnp.arange(
                int(self.native_diffusion_steps), dtype=jnp.float32
            )[::-1]
            self.native_alpha_asc = self.native_alpha[::-1]
            self.native_sigma_asc = self.native_sigma[::-1]

    def _validate_output_prediction_space(self):
        if self.output_prediction_space not in {"velocity", "data", "noise"}:
            raise ValueError(
                "PlainSiT output_prediction_space must be one of "
                "['velocity', 'data', 'noise'], got "
                f"{self.output_prediction_space!r}."
            )
        if self.output_prediction_space != "velocity" and self.objective != "sit":
            raise ValueError(
                "PlainSiT non-velocity output wrappers are only supported for "
                f"objective='sit', got objective={self.objective!r}."
            )

    def _validate_wrapper_configuration(self):
        if self.velocity_map_mode not in {"transport", "dit_native"}:
            raise ValueError(
                "PlainSiT velocity_map_mode must be one of "
                "['transport', 'dit_native'], got "
                f"{self.velocity_map_mode!r}."
            )
        if self.input_alignment_mode not in {"none", "diff2flow"}:
            raise ValueError(
                "PlainSiT input_alignment_mode must be one of "
                "['none', 'diff2flow'], got "
                f"{self.input_alignment_mode!r}."
            )
        if self.native_velocity_derivative_mode not in {
            "finite_difference",
            "analytic",
        }:
            raise ValueError(
                "PlainSiT native_velocity_derivative_mode must be one of "
                "['finite_difference', 'analytic'], got "
                f"{self.native_velocity_derivative_mode!r}."
            )
        if self.output_prediction_space == "velocity":
            if self.velocity_map_mode != "transport":
                raise ValueError(
                    "PlainSiT wrapped velocity_map_mode only applies to native "
                    f"noise/data outputs, got velocity_map_mode={self.velocity_map_mode!r}."
                )
            if self.input_alignment_mode != "none":
                raise ValueError(
                    "PlainSiT input_alignment_mode requires a native noise wrapper, "
                    f"got output_prediction_space={self.output_prediction_space!r}."
                )
        if self.output_prediction_space == "data" and self.velocity_map_mode != "transport":
            raise ValueError(
                "PlainSiT data outputs only support velocity_map_mode='transport', got "
                f"{self.velocity_map_mode!r}."
            )
        if self.input_alignment_mode == "diff2flow":
            if self.path_type != "Linear":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires path_type='Linear', "
                    f"got {self.path_type!r}."
                )
            if self.output_prediction_space != "noise":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires "
                    f"output_prediction_space='noise', got {self.output_prediction_space!r}."
                )
            if self.velocity_map_mode != "transport":
                raise ValueError(
                    "PlainSiT diff2flow alignment currently requires "
                    f"velocity_map_mode='transport', got {self.velocity_map_mode!r}."
                )

    def _needs_nonzero_transport_eps(self):
        return self.output_prediction_space != "velocity"

    def _needs_native_diffusion_schedule(self):
        return self.output_prediction_space == "noise" and (
            self.velocity_map_mode == "dit_native"
            or self.input_alignment_mode == "diff2flow"
        )

    def _maybe_adjust_transport_eps(self, transport):
        if not self._needs_nonzero_transport_eps():
            return transport

        train_eps = float(transport.train_eps)
        sample_eps = float(transport.sample_eps)
        min_eps = max(float(self.wrapper_eps), 1e-3)
        if train_eps >= min_eps and sample_eps >= min_eps:
            return transport

        return create_transport(
            path_type=self.path_type,
            prediction=self.prediction,
            loss_weight=self.loss_weight,
            train_eps=max(train_eps, min_eps),
            sample_eps=max(sample_eps, min_eps),
        )

    def _broadcast_scalar(self, value, ref):
        value = jnp.asarray(value, dtype=jnp.float32)
        return value.reshape((value.shape[0],) + (1,) * (ref.ndim - 1))

    def _map_model_time(self, t):
        t_model = t.astype(self.dtype)
        if self.model_time_flip:
            t_model = 1.0 - t_model
        return t_model * jnp.asarray(self.model_time_scale, dtype=self.dtype)

    def _compute_transport_schedule(self, t, xt):
        t_expanded = t.reshape((t.shape[0],) + (1,) * (xt.ndim - 1))
        alpha_t, d_alpha_t = self.transport.path_sampler.compute_alpha_t(t_expanded)
        sigma_t, d_sigma_t = self.transport.path_sampler.compute_sigma_t(t_expanded)
        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def _diff2flow_schedule_values(self, t):
        t_query = jnp.clip(
            t.astype(jnp.float32),
            self.diff2flow_t_fm_asc[0],
            self.diff2flow_t_fm_asc[-1],
        )
        tau_model = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_indices_asc
        )
        alpha_tau = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_alpha_asc
        )
        sigma_tau = jnp.interp(
            t_query, self.diff2flow_t_fm_asc, self.native_sigma_asc
        )
        return t_query, tau_model, alpha_tau, sigma_tau

    def _prepare_prediction_context(self, xt, t):
        xt = xt.astype(self.dtype)
        t = t.astype(self.dtype)
        context = {
            "model_x": xt,
            "model_t": self._map_model_time(t),
            "velocity_x": xt,
            "velocity_alpha": None,
            "velocity_sigma": None,
        }
        if self.input_alignment_mode != "diff2flow":
            return context

        _, tau_model, alpha_tau, sigma_tau = self._diff2flow_schedule_values(t)
        velocity_x = self._broadcast_scalar(alpha_tau + sigma_tau, xt) * xt.astype(
            jnp.float32
        )
        context["model_x"] = velocity_x.astype(self.dtype)
        context["model_t"] = tau_model.astype(self.dtype)
        context["velocity_x"] = velocity_x.astype(self.dtype)
        context["velocity_alpha"] = alpha_tau.astype(jnp.float32)
        context["velocity_sigma"] = sigma_tau.astype(jnp.float32)
        return context

    def _run_backbone(self, x, t, y, r=None):
        return self.net(
            x.astype(self.dtype),
            t.astype(self.dtype),
            y,
            r=r,
        )

    def _predict_backbone_output(self, x, t, y, r=None):
        r_model = None
        if r is not None:
            r_model = self._map_model_time(r)
        return self._run_backbone(x, self._map_model_time(t), y, r=r_model)

    def _noise_to_data(self, raw_output, xt, alpha_t, sigma_t):
        alpha_b = self._broadcast_scalar(alpha_t, xt)
        sigma_b = self._broadcast_scalar(sigma_t, xt)
        return (xt - sigma_b * raw_output) / jnp.maximum(alpha_b, self.wrapper_eps)

    def _data_to_transport_velocity(self, raw_output, xt, t):
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._compute_transport_schedule(t, xt)
        sigma_safe = jnp.where(
            jnp.abs(sigma_t) > self.wrapper_eps,
            sigma_t,
            jnp.where(sigma_t >= 0.0, self.wrapper_eps, -self.wrapper_eps),
        )
        x1_hat = raw_output
        x0_hat = (xt - alpha_t * x1_hat) / sigma_safe
        return d_alpha_t * x1_hat + d_sigma_t * x0_hat

    def _noise_to_transport_velocity(
        self,
        raw_output,
        xt,
        t,
        *,
        alpha_override=None,
        sigma_override=None,
    ):
        if alpha_override is not None and sigma_override is not None:
            x1_hat = self._noise_to_data(raw_output, xt, alpha_override, sigma_override)
            return x1_hat - raw_output

        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self._compute_transport_schedule(t, xt)
        x1_hat = self._noise_to_data(raw_output, xt, alpha_t.reshape((alpha_t.shape[0],)), sigma_t.reshape((sigma_t.shape[0],)))
        return d_alpha_t * x1_hat + d_sigma_t * raw_output

    def _noise_to_dit_native_velocity(self, raw_output, xt, t):
        diffusion_steps = max(int(self.native_diffusion_steps), 1)
        time_scale = self.native_time_scale
        model_time_scale = jnp.asarray(self.model_time_scale, dtype=jnp.float32)
        tau = jnp.asarray(t, dtype=jnp.float32) * (model_time_scale / time_scale)
        dtau_dt = model_time_scale / time_scale
        if self.model_time_flip:
            tau = 1.0 - tau
            dtau_dt = -dtau_dt
        tau = jnp.clip(tau, 0.0, 1.0)

        schedule_pos = jnp.clip(
            jnp.rint(tau * time_scale).astype(jnp.int32),
            0,
            diffusion_steps - 1,
        )
        alpha_scalar = self.native_alpha[schedule_pos]
        sigma_scalar = self.native_sigma[schedule_pos]

        if self.native_velocity_derivative_mode == "analytic":
            if self.native_noise_schedule != "linear":
                raise ValueError(
                    "PlainSiT analytic native velocity derivatives currently require "
                    f"native_noise_schedule='linear', got {self.native_noise_schedule!r}."
                )
            beta_start = jnp.asarray(1e-4, dtype=jnp.float32)
            beta_end = jnp.asarray(2e-2, dtype=jnp.float32)
            beta_tau = time_scale * (beta_start + (beta_end - beta_start) * tau)
            sigma_safe = jnp.maximum(sigma_scalar, self.wrapper_eps)
            alpha_dot_tau = -0.5 * beta_tau * alpha_scalar
            sigma_dot_tau = 0.5 * beta_tau * (alpha_scalar ** 2) / sigma_safe
        else:
            next_pos = jnp.where(schedule_pos > 0, schedule_pos - 1, schedule_pos + 1)
            alpha_next = self.native_alpha[next_pos]
            sigma_next = self.native_sigma[next_pos]
            tau_next = self.native_tau[next_pos]
            tau_cur = self.native_tau[schedule_pos]
            dt = tau_next - tau_cur
            dt = jnp.where(
                jnp.abs(dt) > self.wrapper_eps,
                dt,
                jnp.where(dt >= 0.0, self.wrapper_eps, -self.wrapper_eps),
            )
            alpha_dot_tau = (alpha_next - alpha_scalar) / dt
            sigma_dot_tau = (sigma_next - sigma_scalar) / dt

        alpha_b = self._broadcast_scalar(alpha_scalar, xt)
        sigma_b = self._broadcast_scalar(sigma_scalar, xt)
        alpha_dot_b = self._broadcast_scalar(alpha_dot_tau * dtau_dt, xt)
        sigma_dot_b = self._broadcast_scalar(sigma_dot_tau * dtau_dt, xt)
        x0_hat = (xt - sigma_b * raw_output) / jnp.maximum(alpha_b, self.wrapper_eps)
        return alpha_dot_b * x0_hat + sigma_dot_b * raw_output

    def _compute_wrapped_velocity(self, raw_output, xt, t, context=None):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "velocity":
            return raw_output

        if self.output_prediction_space == "data":
            return self._data_to_transport_velocity(raw_output, xt, t)
        if self.velocity_map_mode == "dit_native":
            return self._noise_to_dit_native_velocity(raw_output, xt, t)
        if context is not None and context["velocity_alpha"] is not None:
            return self._noise_to_transport_velocity(
                raw_output,
                context["velocity_x"].astype(self.dtype),
                t,
                alpha_override=context["velocity_alpha"],
                sigma_override=context["velocity_sigma"],
            )
        return self._noise_to_transport_velocity(raw_output, xt, t)

    def _compute_data_prediction(self, raw_output, xt, t, context=None):
        self._validate_output_prediction_space()
        if self.output_prediction_space == "data":
            return raw_output
        if self.output_prediction_space == "noise":
            if context is not None and context["velocity_alpha"] is not None:
                return self._noise_to_data(
                    raw_output,
                    context["velocity_x"].astype(self.dtype),
                    context["velocity_alpha"],
                    context["velocity_sigma"],
                )
            alpha_t, sigma_t, _, _ = self._compute_transport_schedule(t, xt)
            return self._noise_to_data(
                raw_output,
                xt,
                alpha_t.reshape((alpha_t.shape[0],)),
                sigma_t.reshape((sigma_t.shape[0],)),
            )
        raise ValueError(
            "PlainSiT data reconstruction is only defined for output_prediction_space "
            f"'data' or 'noise', got {self.output_prediction_space!r}."
        )

    def _wrapped_velocity_loss_weight(self, t):
        if self.wrapped_loss_weight in {"", "none", None}:
            return 1.0
        if self.wrapped_loss_weight != "denom_squared":
            raise ValueError(
                "PlainSiT wrapped_loss_weight must be one of "
                "['none', 'denom_squared'], got "
                f"{self.wrapped_loss_weight!r}."
            )
        if self.output_prediction_space == "velocity":
            return 1.0

        if self.output_prediction_space == "noise" and self.input_alignment_mode == "diff2flow":
            _, _, alpha_t, sigma_t = self._diff2flow_schedule_values(t.astype(self.dtype))
        else:
            t_expanded = t.reshape((t.shape[0],) + (1,) * 0)
            alpha_t, _ = self.transport.path_sampler.compute_alpha_t(t_expanded)
            sigma_t, _ = self.transport.path_sampler.compute_sigma_t(t_expanded)
        denom = sigma_t if self.output_prediction_space == "data" else alpha_t
        denom = jnp.maximum(jnp.abs(denom), self.wrapper_eps)
        return jnp.square(denom)

    def _predict_transport_output(self, x, t, y, r=None):
        if r is not None:
            raw_output = self._predict_backbone_output(x, t, y, r=r)
            return self._compute_wrapped_velocity(
                raw_output, x.astype(self.dtype), t.astype(self.dtype)
            )

        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        raw_output = self._run_backbone(context["model_x"], context["model_t"], y)
        return self._compute_wrapped_velocity(
            raw_output, x.astype(self.dtype), t.astype(self.dtype), context=context
        )

    def predict_native_output(self, x, t, y, r=None):
        """Return the backbone output in its configured native prediction space."""
        if r is not None:
            return self._predict_backbone_output(x, t, y, r=r)
        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._run_backbone(context["model_x"], context["model_t"], y)

    def predict_data(self, x, t, y, r=None):
        """Reconstruct x1/data from a native non-velocity prediction."""
        raw_output = self.predict_native_output(x, t, y, r=r)
        context = None if r is not None else self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._compute_data_prediction(
            raw_output, x.astype(self.dtype), t.astype(self.dtype), context=context
        )

    def convert_native_output_to_data(self, raw_output, x, t):
        """Reconstruct x1/data from a provided native backbone prediction."""
        context = self._prepare_prediction_context(
            x.astype(self.dtype), t.astype(self.dtype)
        )
        return self._compute_data_prediction(
            raw_output,
            x.astype(self.dtype),
            t.astype(self.dtype),
            context=context,
        )

    def debug_noise_reconstruction(self, images, labels):
        """Return training-path noise prediction diagnostics for one batch."""
        if self.objective != "sit" or self.output_prediction_space != "noise":
            raise ValueError(
                "debug_noise_reconstruction is only defined for objective='sit' "
                "with output_prediction_space='noise'."
            )

        x1 = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)
        t, x0, x1 = self.transport.sample(x1, self.make_rng("gen"))
        t, xt, _ = self.transport.path_sampler.plan(t, x0, x1)
        context = self._prepare_prediction_context(xt, t)
        x0_hat = self._run_backbone(context["model_x"], context["model_t"], labels)
        x1_hat = self._compute_data_prediction(x0_hat, xt, t, context=context)

        mse_x0 = jnp.mean(jnp.square(x0_hat - x0), axis=tuple(range(1, x0.ndim)))
        mse_x1 = jnp.mean(jnp.square(x1_hat - x1), axis=tuple(range(1, x1.ndim)))
        return {
            "x0": x0,
            "x1": x1,
            "xt": xt,
            "x0_hat": x0_hat,
            "x1_hat": x1_hat,
            "t": t,
            "mse_x0": jnp.mean(mse_x0),
            "mse_x1": jnp.mean(mse_x1),
        }

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)
        return t, r, fm_mask

    def _drop_labels(self, labels, rng):
        if (
            (not self.target_use_null_class)
            or self.class_dropout_prob <= 0.0
            or self.eval
        ):
            return labels

        drop_mask = jax.random.uniform(rng, labels.shape, dtype=jnp.float32)
        drop_mask = drop_mask < self.class_dropout_prob
        null_labels = jnp.full(labels.shape, self.num_classes, dtype=jnp.int32)
        return jnp.where(drop_mask, null_labels, labels)

    def forward(self, images, labels):
        """Compute the official SiT transport loss."""
        if self.objective == "power_meanflow":
            return self.forward_power_meanflow(images, labels)

        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)

        rng_drop, rng_loss = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)

        def model_fn(xt, t, y):
            return self._predict_transport_output(xt, t, y)

        terms = self.transport.training_losses(
            model_fn,
            x,
            rng=rng_loss,
            model_kwargs={"y": labels},
        )
        loss_weight = self._wrapped_velocity_loss_weight(terms["t"])
        weighted_loss = terms["loss"] * loss_weight
        loss = jnp.mean(weighted_loss)
        dict_losses = {
            "loss": loss,
            "loss_transport": loss,
            "loss_transport_unweighted": jnp.mean(terms["loss"]),
            "wrapped_loss_weight_mean": jnp.mean(loss_weight),
            "t_mean": jnp.mean(terms["t"]),
        }
        return loss, dict_losses

    def forward_power_meanflow(self, images, labels):
        """Compute the experimental power-geometry mean-flow loss."""
        x = images.astype(self.dtype)
        labels = labels.astype(jnp.int32)
        bz = x.shape[0]

        rng_drop, rng_eps = jax.random.split(self.make_rng("gen"))
        labels = self._drop_labels(labels, rng_drop)
        eps = jax.random.normal(rng_eps, x.shape, dtype=self.dtype)

        t, r, fm_mask = self.sample_tr(bz)
        t_scalar = t.reshape((bz,))
        r_scalar = r.reshape((bz,))
        one_minus_t = jnp.clip(1.0 - t, 1e-6, 1.0)
        one_minus_r = jnp.clip(1.0 - r, 1e-6, 1.0)
        t_clamped = jnp.clip(t, 1e-6, 1.0)
        r_clamped = jnp.clip(r, 1e-6, 1.0)
        k = jnp.asarray(self.path_power_k, dtype=self.dtype)

        a_t = one_minus_t**k
        b_t = t_clamped**k
        z_t = a_t * x + b_t * eps

        inst_target = (
            -k * (one_minus_t ** (k - 1.0)) * x
            + k * (t_clamped ** (k - 1.0)) * eps
        )

        a_r = one_minus_r**k
        b_r = r_clamped**k
        z_r = a_r * x + b_r * eps
        denom = jnp.maximum(jnp.abs(t - r), 1e-6)
        mf_target = (z_t - z_r) / denom
        target = jnp.where(fm_mask, inst_target, mf_target)

        pred = self.net(
            z_t.astype(self.dtype),
            t_scalar.astype(self.dtype),
            labels,
            r=r_scalar.astype(self.dtype),
        )
        sq_error = (pred - target.astype(pred.dtype)) ** 2
        per_example_loss = jnp.mean(sq_error.reshape((bz, -1)), axis=1)
        inst_mask = fm_mask.reshape((bz,)).astype(self.dtype)
        mf_mask = 1.0 - inst_mask
        inst_denom = jnp.maximum(jnp.sum(inst_mask), 1.0)
        mf_denom = jnp.maximum(jnp.sum(mf_mask), 1.0)
        inst_loss = jnp.sum(per_example_loss * inst_mask) / inst_denom
        mf_loss = jnp.sum(per_example_loss * mf_mask) / mf_denom
        loss = jnp.mean(per_example_loss)

        dict_losses = {
            "loss": loss,
            "loss_power_meanflow": loss,
            "loss_instantaneous": inst_loss,
            "loss_meanflow": mf_loss,
            "diag_fraction": jnp.mean(inst_mask),
            "t_mean": jnp.mean(t_scalar),
            "r_mean": jnp.mean(r_scalar),
            "interval_mean": jnp.mean(jnp.abs(t_scalar - r_scalar)),
        }
        return loss, dict_losses

    def __call__(self, x, t, y, r=None):
        """Initialization-only forward that mirrors the exact SiT backbone."""
        return self._predict_transport_output(x, t, y, r=r)
