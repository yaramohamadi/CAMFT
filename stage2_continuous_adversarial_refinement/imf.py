import flax.linen as nn
import jax
import jax.numpy as jnp

from models import imfDiT
from models import jit as imfJiT
from utils.dit_diffusion import get_named_beta_schedule


def _resolve_backbone_ctor(model_str):
    """Resolve a backbone constructor by name.

    Additive dispatch: pixel-space JiT backbones (names starting with
    ``imfJiT``) live in ``models/jit.py``; everything else (DiT/SiT) resolves
    from ``models.imfDiT`` exactly as before.
    """
    if model_str.startswith("imfJiT"):
        return getattr(imfJiT, model_str)
    return getattr(imfDiT, model_str)
from utils.imf_param_util import is_v_only_param_tree
from utils.sit_transport_jax import create_transport


def generate(variable, model, rng, n_sample, config, 
             num_steps, omega, t_min, t_max, sample_idx=None):
    """
    Generate samples from the model
    
    Args:
        variable: Model parameters.
        model: iMeanFlow model.
        rng: JAX random key.
        n_sample: Number of samples to generate.
        config: Configuration object.
        num_steps: Number of sampling steps.
        omega: CFG scale.
        t_min, t_max: Guidance interval.
        sample_idx: Optional index for class-conditional sampling.

    Returns:
        images: Generated images.
    """
    num_classes = config.dataset.num_classes
    img_size, img_channels = config.dataset.image_size, config.dataset.image_channels

    x_shape = (n_sample, img_size, img_size, img_channels)
    rng, rng_xt, rng_sample = jax.random.split(rng, 3)

    z_t = jax.random.normal(rng_xt, x_shape, dtype=model.dtype)

    if sample_idx is not None:
        all_y = jnp.arange(n_sample, dtype=jnp.int32)
        y = all_y + sample_idx * n_sample
        y = y % num_classes
    else:
        y = jax.random.randint(rng_sample, (n_sample,), 0, num_classes)

    meanflow_reverse_time = bool(config.sampling.get("meanflow_reverse_time", False))
    target_transport_eps = 0.0
    if (
        model._uses_dmf_single_head_backbone()
        and model.target_output_prediction_space == "noise"
        and model.target_velocity_map_mode == "transport"
    ):
        default_eps = max(float(model.target_wrapper_eps), 1e-3)
        target_transport_eps = float(
            config.sampling.get(
                "target_transport_velocity_eps",
                config.sampling.get("transport_velocity_eps", default_eps),
            )
        )
    if model._uses_auxiliary_v_head() or meanflow_reverse_time:
        t_steps = jnp.linspace(1.0, target_transport_eps, num_steps + 1)
    else:
        t_steps = jnp.linspace(target_transport_eps, 1.0, num_steps + 1)

    def step_fn(i, x_i):
        return model.apply(variable, x_i, y, i, t_steps,
            omega, t_min, t_max, method=model.sample_one_step)

    images = jax.lax.fori_loop(0, num_steps, step_fn, z_t)

    return images


class iMeanFlow(nn.Module):
    """improved MeanFlow"""

    # Model and dataset
    model_str: str
    dtype = jnp.float32
    num_classes: int = 1000

    # Noise distribution
    P_mean: float = -0.4
    P_std: float = 1.0

    # Loss
    data_proportion: float = 0.5
    cfg_beta: float = 1.0
    class_dropout_prob: float = 0.1
    training_mode: str = "imf_jvp"
    use_dogfit: bool = False
    target_use_null_class: bool = True
    source_prediction_space: str = "v"
    source_model_str: str = ""
    source_num_classes: int = 1000
    source_path_type: str = "Linear"
    source_velocity_map_mode: str = "transport"
    source_native_velocity_derivative_mode: str = "finite_difference"
    source_wrapper_eps: float = 1e-6
    source_model_time_scale: float = 1.0
    source_model_time_flip: bool = False
    source_native_diffusion_steps: int = 1000
    source_native_beta_schedule: str = "linear"
    target_output_prediction_space: str = "velocity"
    target_velocity_map_mode: str = "transport"
    target_input_alignment_mode: str = "none"
    target_native_velocity_derivative_mode: str = "finite_difference"
    target_native_diffusion_steps: int = 1000
    target_native_beta_schedule: str = "linear"
    target_wrapper_eps: float = 1e-6
    target_model_time_scale: float = 1.0
    target_model_time_flip: bool = False
    use_auxiliary_v_head: bool = True
    use_context_guidance_conditioning: bool = False
    use_adaln_guidance_scale_conditioning: bool = False
    adaln_guidance_scale_init: str = "timestep"
    use_adaln_condition_mixing: bool = False
    decoder_only_guidance_conditioning: bool = False
    time_conditioning_mode: str = "split"
    use_ema_vc: bool = False
    use_v_only_teacher_source_copies: bool = False
    use_training_guidance: bool = True
    training_guidance_interval_strategy: str = "sampled"
    training_guidance_t_min: float = 0.0
    training_guidance_t_max: float = 1.0
    training_guidance_start_step: int = 0
    guidance_scale_strategy: str = "sampled"
    max_sampled_guidance_scale: float = 8.0
    fixed_guidance_scale: float = 7.5
    baked_guidance_blend: float = 0.5
    use_positive_sit_dmf_mf_target: bool = False
    split_consistency_midpoint_strategy: str = "uniform"
    split_consistency_midpoint_eps: float = 1e-3
    split_consistency_source_first_prob: float = 0.0
    split_consistency_source_second_prob: float = 0.0
    split_consistency_boundary_mode: str = "exact"
    split_consistency_boundary_epsilon_distribution: str = "half_normal"
    split_consistency_boundary_epsilon: float = 1e-3
    split_consistency_boundary_epsilon_min: float = 1e-6

    # Training dynamics
    norm_p: float = 1.0
    norm_eps: float = 0.01

    # Evaluation mode
    eval: bool = False

    def setup(self):
        """
        Setup improved MeanFlow model.
        """
        self._validate_target_wrapper_configuration()
        net_fn = _resolve_backbone_ctor(self.model_str)
        net_kwargs = dict(
            name="net",
            num_classes=self.num_classes,
            use_null_class=self.target_use_null_class,
            eval=self.eval,
        )
        if self._uses_plain_imf_dit_backbone():
            net_kwargs["use_auxiliary_v_head"] = self.use_auxiliary_v_head
        if self._uses_dmf_single_head_backbone():
            net_kwargs["use_context_guidance_conditioning"] = self.use_context_guidance_conditioning
            net_kwargs["use_adaln_guidance_scale_conditioning"] = (
                self.use_adaln_guidance_scale_conditioning
            )
            net_kwargs["adaln_guidance_scale_init"] = self.adaln_guidance_scale_init
            net_kwargs["use_adaln_condition_mixing"] = self.use_adaln_condition_mixing
            net_kwargs["decoder_only_guidance_conditioning"] = (
                self.decoder_only_guidance_conditioning
            )
            net_kwargs["time_conditioning_mode"] = self.time_conditioning_mode
        self.net: imfDiT.imfDiT = net_fn(**net_kwargs)
        if self._needs_target_native_diffusion_schedule():
            native_betas = get_named_beta_schedule(
                self.target_native_beta_schedule,
                self.target_native_diffusion_steps,
            )
            native_alphas = 1.0 - native_betas
            native_alphas_cumprod = jnp.asarray(
                native_alphas.cumprod(), dtype=jnp.float32
            )
            self.target_native_alpha = jnp.sqrt(native_alphas_cumprod)
            self.target_native_sigma = jnp.sqrt(
                jnp.maximum(1.0 - native_alphas_cumprod, 1e-12)
            )
            self.target_native_time_scale = jnp.asarray(
                max(self.target_native_diffusion_steps - 1, 1), dtype=jnp.float32
            )
            self.target_native_tau = (
                jnp.arange(self.target_native_diffusion_steps, dtype=jnp.float32)
                / self.target_native_time_scale
            )
            diff2flow_t_fm = self.target_native_alpha / jnp.maximum(
                self.target_native_alpha + self.target_native_sigma,
                self.target_wrapper_eps,
            )
            self.target_diff2flow_t_fm_asc = diff2flow_t_fm[::-1]
            self.target_native_indices_asc = jnp.arange(
                self.target_native_diffusion_steps, dtype=jnp.float32
            )[::-1]
            self.target_native_alpha_asc = self.target_native_alpha[::-1]
            self.target_native_sigma_asc = self.target_native_sigma[::-1]
        if (
            self.use_dogfit
            or self._uses_src_reg_training_mode()
            or self._uses_split_consistency_source_ablation()
        ):
            source_num_classes = (
                self.source_num_classes
                if self.use_dogfit
                else self.num_classes
            )
            source_use_null_class = True if self.use_dogfit else self.target_use_null_class
            source_model_str = (
                self.source_model_str if self.source_model_str else self.model_str
            )
            source_net_fn = _resolve_backbone_ctor(source_model_str)
            source_net_kwargs = dict(
                name="source_net",
                num_classes=source_num_classes,
                use_null_class=source_use_null_class,
                eval=False,
            )
            if self._uses_plain_exact_source_backbone(source_model_str):
                source_net_kwargs["learn_sigma"] = True
                source_net_kwargs["return_learned_sigma"] = False
                source_net_kwargs["use_r_conditioning"] = False
            elif self._uses_plain_imf_dit_backbone(source_model_str):
                source_net_kwargs["use_auxiliary_v_head"] = self.use_auxiliary_v_head
            elif self._uses_dmf_single_head_backbone(source_model_str):
                source_net_kwargs["use_context_guidance_conditioning"] = (
                    self.use_context_guidance_conditioning
                )
                source_net_kwargs["use_adaln_guidance_scale_conditioning"] = (
                    self.use_adaln_guidance_scale_conditioning
                )
                source_net_kwargs["adaln_guidance_scale_init"] = (
                    self.adaln_guidance_scale_init
                )
                source_net_kwargs["use_adaln_condition_mixing"] = (
                    self.use_adaln_condition_mixing
                )
                source_net_kwargs["decoder_only_guidance_conditioning"] = (
                    self.decoder_only_guidance_conditioning
                )
                source_net_kwargs["time_conditioning_mode"] = (
                    self.time_conditioning_mode
                )
            self.source_net = source_net_fn(**source_net_kwargs)
            if self.source_prediction_space == "noise":
                self.source_transport = create_transport(
                    path_type=self.source_path_type,
                    prediction="noise",
                )
                native_betas = get_named_beta_schedule(
                    self.source_native_beta_schedule,
                    self.source_native_diffusion_steps,
                )
                native_alphas = 1.0 - native_betas
                native_alphas_cumprod = jnp.asarray(
                    native_alphas.cumprod(), dtype=jnp.float32
                )
                self.source_native_alpha = jnp.sqrt(native_alphas_cumprod)
                self.source_native_sigma = jnp.sqrt(
                    jnp.maximum(1.0 - native_alphas_cumprod, 1e-12)
                )
                self.source_native_time_scale = jnp.asarray(
                    max(self.source_native_diffusion_steps - 1, 1), dtype=jnp.float32
                )
                self.source_native_tau = (
                    jnp.arange(self.source_native_diffusion_steps, dtype=jnp.float32)
                    / self.source_native_time_scale
                )

    def _uses_src_reg_training_mode(self):
        return self.training_mode == "imf_jvp_free_src_reg"

    def _uses_split_consistency_training_mode(self):
        return self.training_mode == "imf_split_consistency"

    def _uses_split_consistency_source_ablation(self):
        return self._uses_split_consistency_training_mode() and (
            self.split_consistency_source_first_prob > 0.0
            or self.split_consistency_source_second_prob > 0.0
        )

    def _uses_auxiliary_v_head(self):
        return self.use_auxiliary_v_head

    def _uses_dmf_single_head_backbone(self, model_str=None):
        model_str = self.model_str if model_str is None else model_str
        return (not self._uses_auxiliary_v_head()) and ("_DMF" in model_str)

    def _uses_sit_dmf_time_convention(self):
        return self._uses_dmf_single_head_backbone()

    def _uses_sit_cfg_channel_rule(self):
        return self._uses_dmf_single_head_backbone() or ("SiT" in self.model_str)

    def _uses_plain_imf_dit_backbone(self, model_str=None):
        model_str = self.model_str if model_str is None else model_str
        return ("DiT" in model_str) and ("SiT" not in model_str) and ("_DMF" not in model_str)

    def _uses_imf_dit_backbone(self):
        return self._uses_plain_imf_dit_backbone()

    def _uses_plain_exact_source_backbone(self, model_str=None):
        model_str = self.model_str if model_str is None else model_str
        return model_str.startswith("flaxDiT") or model_str.startswith("flaxSiT")

    def _uses_sit_guidance_context_conditioning(self):
        return (
            self._uses_dmf_single_head_backbone()
            and self.use_context_guidance_conditioning
        )

    def _uses_sit_adaln_guidance_scale_conditioning(self):
        return (
            self._uses_dmf_single_head_backbone()
            and self.use_adaln_guidance_scale_conditioning
        )

    def _uses_baked_fixed_guidance_sampling(self):
        return (
            (not self._uses_auxiliary_v_head())
            and self.use_training_guidance
            and self.guidance_scale_strategy == "fixed"
            and not self._uses_sit_guidance_context_conditioning()
            and not self._uses_sit_adaln_guidance_scale_conditioning()
        )

    def _uses_target_side_dit_adaptation(self):
        return (
            self.target_output_prediction_space != "velocity"
            or self.target_input_alignment_mode != "none"
            or self.target_model_time_flip
            or float(self.target_model_time_scale) != 1.0
        )

    def _needs_target_native_diffusion_schedule(self):
        return self.target_output_prediction_space == "noise" and (
            self.target_velocity_map_mode == "dit_native"
            or self.target_input_alignment_mode == "diff2flow"
        )

    def _validate_target_wrapper_configuration(self):
        if self.target_output_prediction_space not in {"velocity", "data", "noise"}:
            raise ValueError(
                "iMeanFlow target_output_prediction_space must be one of "
                "['velocity', 'data', 'noise'], got "
                f"{self.target_output_prediction_space!r}."
            )
        if self.target_velocity_map_mode not in {"transport", "dit_native"}:
            raise ValueError(
                "iMeanFlow target_velocity_map_mode must be one of "
                "['transport', 'dit_native'], got "
                f"{self.target_velocity_map_mode!r}."
            )
        if self.target_input_alignment_mode not in {"none", "diff2flow"}:
            raise ValueError(
                "iMeanFlow target_input_alignment_mode must be one of "
                "['none', 'diff2flow'], got "
                f"{self.target_input_alignment_mode!r}."
            )
        if self.target_native_velocity_derivative_mode not in {
            "finite_difference",
            "analytic",
        }:
            raise ValueError(
                "iMeanFlow target_native_velocity_derivative_mode must be one of "
                "['finite_difference', 'analytic'], got "
                f"{self.target_native_velocity_derivative_mode!r}."
            )
        if self._uses_target_side_dit_adaptation() and not self._uses_dmf_single_head_backbone():
            raise ValueError(
                "Target-side DiT adaptation is currently only supported for "
                f"single-head DMF backbones, got model_str={self.model_str!r}."
            )
        if self.target_output_prediction_space == "velocity":
            if self.target_velocity_map_mode != "transport":
                raise ValueError(
                    "Wrapped target_velocity_map_mode only applies to native "
                    "noise/data outputs, got "
                    f"target_velocity_map_mode={self.target_velocity_map_mode!r}."
                )
            if self.target_input_alignment_mode != "none":
                raise ValueError(
                    "Target-side input alignment requires a native noise wrapper, got "
                    f"target_output_prediction_space={self.target_output_prediction_space!r}."
                )
        if (
            self.target_output_prediction_space == "data"
            and self.target_velocity_map_mode != "transport"
        ):
            raise ValueError(
                "Target-side data outputs only support "
                "target_velocity_map_mode='transport', got "
                f"{self.target_velocity_map_mode!r}."
            )
        if self.target_input_alignment_mode == "diff2flow":
            if self.target_output_prediction_space != "noise":
                raise ValueError(
                    "Target-side Diff2Flow alignment currently requires "
                    "target_output_prediction_space='noise', got "
                    f"{self.target_output_prediction_space!r}."
                )
            if self.target_velocity_map_mode != "transport":
                raise ValueError(
                    "Target-side Diff2Flow alignment currently requires "
                    "target_velocity_map_mode='transport', got "
                    f"{self.target_velocity_map_mode!r}."
                )
            if self.target_model_time_flip or float(self.target_model_time_scale) != 1.0:
                raise ValueError(
                    "Target-side Diff2Flow alignment already defines the DiT time map; "
                    "leave target_model_time_flip=False and target_model_time_scale=1.0."
                )

    def _resolve_source_params(self, source_params):
        if source_params is None:
            raise ValueError("source_params must be provided for source-model calls.")

        if not isinstance(source_params, dict):
            return source_params

        if "source_net" in source_params:
            return source_params["source_net"]
        if "net" in source_params:
            return source_params["net"]
        return source_params

    def _batch_scalar(self, value, bz, dtype=None):
        """Collapse `[B]` or broadcasted `[B, 1, ...]` scalars to `[B]`."""
        value = jnp.asarray(value, dtype=dtype)
        if value.ndim == 1:
            return value.reshape((bz,))
        return value.reshape((bz, -1))[:, 0]

    def _broadcast_scalar(self, value, ref, dtype=None):
        value = jnp.asarray(value, dtype=dtype)
        return value.reshape((value.shape[0],) + (1,) * (ref.ndim - 1))

    def _map_target_model_time(self, t):
        t_model = t.astype(self.dtype)
        if self.target_model_time_flip:
            t_model = 1.0 - t_model
        return t_model * jnp.asarray(self.target_model_time_scale, dtype=self.dtype)

    def _target_diff2flow_schedule_values(self, t):
        t_query = jnp.clip(
            t.astype(jnp.float32),
            self.target_diff2flow_t_fm_asc[0],
            self.target_diff2flow_t_fm_asc[-1],
        )
        tau_model = jnp.interp(
            t_query,
            self.target_diff2flow_t_fm_asc,
            self.target_native_indices_asc,
        )
        alpha_tau = jnp.interp(
            t_query,
            self.target_diff2flow_t_fm_asc,
            self.target_native_alpha_asc,
        )
        sigma_tau = jnp.interp(
            t_query,
            self.target_diff2flow_t_fm_asc,
            self.target_native_sigma_asc,
        )
        return t_query, tau_model, alpha_tau, sigma_tau

    def _prepare_target_prediction_context(self, x, t, r):
        bz = x.shape[0]
        x = x.astype(self.dtype)
        t_scalar = self._batch_scalar(t, bz, dtype=self.dtype)
        r_scalar = self._batch_scalar(r, bz, dtype=self.dtype)
        context = {
            "model_x": x,
            "model_t": self._map_target_model_time(t_scalar),
            "model_r": self._map_target_model_time(r_scalar),
            "velocity_x": x,
            "velocity_alpha": None,
            "velocity_sigma": None,
        }
        if self.target_input_alignment_mode != "diff2flow":
            return context

        _, t_tau_model, alpha_tau, sigma_tau = self._target_diff2flow_schedule_values(
            t_scalar
        )
        _, r_tau_model, _, _ = self._target_diff2flow_schedule_values(r_scalar)
        velocity_x = self._broadcast_scalar(alpha_tau + sigma_tau, x, dtype=jnp.float32) * x.astype(
            jnp.float32
        )
        context["model_x"] = velocity_x.astype(self.dtype)
        context["model_t"] = t_tau_model.astype(self.dtype)
        context["model_r"] = r_tau_model.astype(self.dtype)
        context["velocity_x"] = velocity_x.astype(self.dtype)
        context["velocity_alpha"] = alpha_tau.astype(jnp.float32)
        context["velocity_sigma"] = sigma_tau.astype(jnp.float32)
        return context

    def _run_target_backbone(self, x, t, r, y, omega, t_min, t_max):
        bz = x.shape[0]
        if self._uses_sit_guidance_context_conditioning():
            return self.net(
                x.astype(self.dtype),
                t.reshape(bz).astype(self.dtype),
                r.reshape(bz).astype(self.dtype),
                y,
                self._batch_scalar(omega, bz, dtype=self.dtype),
                self._batch_scalar(t_min, bz, dtype=self.dtype),
                self._batch_scalar(t_max, bz, dtype=self.dtype),
            )
        if self._uses_sit_adaln_guidance_scale_conditioning():
            return self.net(
                x.astype(self.dtype),
                t.reshape(bz).astype(self.dtype),
                r.reshape(bz).astype(self.dtype),
                y,
                self._batch_scalar(omega, bz, dtype=self.dtype),
            )
        return self.net(
            x.astype(self.dtype),
            t.reshape(bz).astype(self.dtype),
            r.reshape(bz).astype(self.dtype),
            y,
        )

    def _target_noise_to_data(self, raw_output, x_t, alpha_t, sigma_t):
        alpha_b = self._broadcast_scalar(alpha_t, x_t, dtype=self.dtype)
        sigma_b = self._broadcast_scalar(sigma_t, x_t, dtype=self.dtype)
        return (x_t - sigma_b * raw_output) / jnp.maximum(alpha_b, self.target_wrapper_eps)

    def _target_data_to_transport_velocity(self, raw_output, x_t, t):
        t_scalar = self._batch_scalar(t, x_t.shape[0], dtype=jnp.float32)
        alpha_t = self._broadcast_scalar(t_scalar, x_t, dtype=self.dtype)
        sigma_t = self._broadcast_scalar(1.0 - t_scalar, x_t, dtype=self.dtype)
        sigma_safe = jnp.where(
            jnp.abs(sigma_t) > self.target_wrapper_eps,
            sigma_t,
            jnp.where(sigma_t >= 0.0, self.target_wrapper_eps, -self.target_wrapper_eps),
        )
        x0_hat = (x_t - alpha_t * raw_output) / sigma_safe
        return raw_output - x0_hat

    def _target_noise_to_transport_velocity(
        self,
        raw_output,
        x_t,
        t,
        *,
        alpha_override=None,
        sigma_override=None,
    ):
        if alpha_override is not None and sigma_override is not None:
            x1_hat = self._target_noise_to_data(
                raw_output,
                x_t,
                alpha_override,
                sigma_override,
            )
            return x1_hat - raw_output

        t_scalar = self._batch_scalar(t, x_t.shape[0], dtype=jnp.float32)
        x1_hat = self._target_noise_to_data(
            raw_output,
            x_t,
            t_scalar,
            1.0 - t_scalar,
        )
        return x1_hat - raw_output

    def _target_noise_to_dit_native_velocity(self, raw_output, x_t, t):
        diffusion_steps = max(int(self.target_native_diffusion_steps), 1)
        time_scale = self.target_native_time_scale
        model_time_scale = jnp.asarray(self.target_model_time_scale, dtype=jnp.float32)
        tau = jnp.asarray(t, dtype=jnp.float32) * (model_time_scale / time_scale)
        dtau_dt = model_time_scale / time_scale
        if self.target_model_time_flip:
            tau = 1.0 - tau
            dtau_dt = -dtau_dt
        tau = jnp.clip(tau, 0.0, 1.0)

        schedule_pos = jnp.clip(
            jnp.rint(tau * time_scale).astype(jnp.int32),
            0,
            diffusion_steps - 1,
        )
        alpha_scalar = self.target_native_alpha[schedule_pos]
        sigma_scalar = self.target_native_sigma[schedule_pos]

        if self.target_native_velocity_derivative_mode == "analytic":
            if self.target_native_beta_schedule != "linear":
                raise ValueError(
                    "iMeanFlow analytic target native velocity derivatives currently "
                    "require target_native_beta_schedule='linear', got "
                    f"{self.target_native_beta_schedule!r}."
                )
            beta_start = jnp.asarray(1e-4, dtype=jnp.float32)
            beta_end = jnp.asarray(2e-2, dtype=jnp.float32)
            beta_tau = time_scale * (beta_start + (beta_end - beta_start) * tau)
            sigma_safe = jnp.maximum(sigma_scalar, self.target_wrapper_eps)
            alpha_dot_tau = -0.5 * beta_tau * alpha_scalar
            sigma_dot_tau = 0.5 * beta_tau * (alpha_scalar ** 2) / sigma_safe
        else:
            next_pos = jnp.where(schedule_pos > 0, schedule_pos - 1, schedule_pos + 1)
            alpha_next = self.target_native_alpha[next_pos]
            sigma_next = self.target_native_sigma[next_pos]
            tau_next = self.target_native_tau[next_pos]
            tau_cur = self.target_native_tau[schedule_pos]
            dt = tau_next - tau_cur
            dt = jnp.where(
                jnp.abs(dt) > self.target_wrapper_eps,
                dt,
                jnp.where(dt >= 0.0, self.target_wrapper_eps, -self.target_wrapper_eps),
            )
            alpha_dot_tau = (alpha_next - alpha_scalar) / dt
            sigma_dot_tau = (sigma_next - sigma_scalar) / dt

        alpha_b = self._broadcast_scalar(alpha_scalar, x_t, dtype=self.dtype)
        sigma_b = self._broadcast_scalar(sigma_scalar, x_t, dtype=self.dtype)
        alpha_dot_b = self._broadcast_scalar(alpha_dot_tau * dtau_dt, x_t, dtype=self.dtype)
        sigma_dot_b = self._broadcast_scalar(sigma_dot_tau * dtau_dt, x_t, dtype=self.dtype)
        x1_hat = (x_t - sigma_b * raw_output) / jnp.maximum(alpha_b, self.target_wrapper_eps)
        return alpha_dot_b * x1_hat + sigma_dot_b * raw_output

    def _compute_target_wrapped_velocity(self, raw_output, x_t, t, context=None):
        if self.target_output_prediction_space == "velocity":
            return raw_output
        if self.target_output_prediction_space == "data":
            return self._target_data_to_transport_velocity(raw_output, x_t, t)
        if self.target_velocity_map_mode == "dit_native":
            return self._target_noise_to_dit_native_velocity(raw_output, x_t, t)
        if context is not None and context["velocity_alpha"] is not None:
            return self._target_noise_to_transport_velocity(
                raw_output,
                context["velocity_x"].astype(self.dtype),
                t,
                alpha_override=context["velocity_alpha"],
                sigma_override=context["velocity_sigma"],
            )
        return self._target_noise_to_transport_velocity(raw_output, x_t, t)

    def _predict_target_velocity(self, x, t, r, omega, t_min, t_max, y):
        context = self._prepare_target_prediction_context(x, t, r)
        raw_output = self._run_target_backbone(
            context["model_x"],
            context["model_t"],
            context["model_r"],
            y,
            omega,
            t_min,
            t_max,
        )
        t_scalar = self._batch_scalar(t, x.shape[0], dtype=self.dtype)
        return self._compute_target_wrapped_velocity(
            raw_output,
            x.astype(self.dtype),
            t_scalar,
            context=context,
        )

    def _source_predict_backbone_output(self, source_params, x, t, y):
        source_param_tree = self._resolve_source_params(source_params)
        bz = x.shape[0]
        t_model = self._batch_scalar(t, bz, dtype=self.dtype)
        if self.source_model_time_flip:
            t_model = 1.0 - t_model
        t_model = t_model * jnp.asarray(self.source_model_time_scale, dtype=self.dtype)
        return self.source_net.apply(
            {"params": source_param_tree},
            x.astype(self.dtype),
            t_model,
            y,
        )

    def _source_noise_to_velocity(self, raw_output, x_t, t):
        if self.source_velocity_map_mode == "transport":
            return self._source_noise_to_velocity_transport(raw_output, x_t, t)
        if self.source_velocity_map_mode in {
            "dit_native",
            "dit_native_analytic",
            "dit_native_finite_difference",
        }:
            derivative_mode = self.source_native_velocity_derivative_mode
            if self.source_velocity_map_mode == "dit_native_analytic":
                derivative_mode = "analytic"
            elif self.source_velocity_map_mode == "dit_native_finite_difference":
                derivative_mode = "finite_difference"
            return self._source_noise_to_velocity_dit_native(
                raw_output,
                x_t,
                t,
                derivative_mode=derivative_mode,
            )
        raise ValueError(
            "source_velocity_map_mode must be 'transport', 'dit_native', "
            "'dit_native_analytic', or 'dit_native_finite_difference', got "
            f"{self.source_velocity_map_mode!r}."
        )

    def _source_noise_to_velocity_transport(self, raw_output, x_t, t):
        t_expanded = t.reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
        alpha_t, d_alpha_t = self.source_transport.path_sampler.compute_alpha_t(
            t_expanded
        )
        sigma_t, d_sigma_t = self.source_transport.path_sampler.compute_sigma_t(
            t_expanded
        )
        # FIX: Convert noise (epsilon) to data (x1) using DDPM formula
        # Previously incorrectly treated raw_output as x0 instead of epsilon
        x1_hat = (x_t - sigma_t * raw_output) / jnp.maximum(
            alpha_t,
            self.source_wrapper_eps,
        )
        # Compute transport velocity: v = d_alpha/dt * x1 + d_sigma/dt * epsilon
        return d_alpha_t * x1_hat + d_sigma_t * raw_output

    def _source_noise_to_velocity_dit_native(
        self, raw_output, x_t, t, *, derivative_mode
    ):
        diffusion_steps = max(int(self.source_native_diffusion_steps), 1)
        time_scale = self.source_native_time_scale
        source_time_scale = jnp.asarray(self.source_model_time_scale, dtype=jnp.float32)
        tau = jnp.asarray(t, dtype=jnp.float32) * (source_time_scale / time_scale)
        dtau_dt = source_time_scale / time_scale
        if self.source_model_time_flip:
            tau = 1.0 - tau
            dtau_dt = -dtau_dt
        tau = jnp.clip(tau, 0.0, 1.0)

        schedule_pos = jnp.clip(
            jnp.rint(tau * time_scale).astype(jnp.int32),
            0,
            diffusion_steps - 1,
        )
        alpha_scalar = self.source_native_alpha[schedule_pos]
        sigma_scalar = self.source_native_sigma[schedule_pos]

        if derivative_mode == "analytic":
            beta_start = jnp.asarray(1e-4, dtype=jnp.float32)
            beta_end = jnp.asarray(2e-2, dtype=jnp.float32)
            beta_tau = time_scale * (beta_start + (beta_end - beta_start) * tau)
            sigma_safe = jnp.maximum(sigma_scalar, self.source_wrapper_eps)
            alpha_dot_tau = -0.5 * beta_tau * alpha_scalar
            sigma_dot_tau = 0.5 * beta_tau * (alpha_scalar ** 2) / sigma_safe
        elif derivative_mode == "finite_difference":
            next_pos = jnp.where(schedule_pos > 0, schedule_pos - 1, schedule_pos + 1)
            alpha_next = self.source_native_alpha[next_pos]
            sigma_next = self.source_native_sigma[next_pos]
            tau_next = self.source_native_tau[next_pos]
            tau_cur = self.source_native_tau[schedule_pos]
            dt = tau_next - tau_cur
            dt = jnp.where(
                jnp.abs(dt) > self.source_wrapper_eps,
                dt,
                jnp.where(dt >= 0.0, self.source_wrapper_eps, -self.source_wrapper_eps),
            )
            alpha_dot_tau = (alpha_next - alpha_scalar) / dt
            sigma_dot_tau = (sigma_next - sigma_scalar) / dt
        else:
            raise ValueError(
                "source_native_velocity_derivative_mode must be "
                f"'finite_difference' or 'analytic', got {derivative_mode!r}."
            )

        expand_dims = (1,) * (x_t.ndim - 1)
        alpha_t = alpha_scalar.reshape((alpha_scalar.shape[0],) + expand_dims)
        sigma_t = sigma_scalar.reshape((sigma_scalar.shape[0],) + expand_dims)
        alpha_dot_t = (alpha_dot_tau * dtau_dt).reshape(
            (alpha_dot_tau.shape[0],) + expand_dims
        )
        sigma_dot_t = (sigma_dot_tau * dtau_dt).reshape(
            (sigma_dot_tau.shape[0],) + expand_dims
        )

        x0_hat = (x_t - sigma_t * raw_output) / jnp.maximum(
            alpha_t, self.source_wrapper_eps
        )
        return alpha_dot_t * x0_hat + sigma_dot_t * raw_output

    def _mf_target_interval_coeff(self, t, r):
        if self._uses_sit_dmf_time_convention() and self.use_positive_sit_dmf_mf_target:
            return r - t
        return t - r

    def _sample_guidance_scale(self, bz):
        if self.guidance_scale_strategy == "fixed":
            return jnp.full((bz, 1, 1, 1), self.fixed_guidance_scale, dtype=jnp.float32)
        if self.guidance_scale_strategy != "sampled":
            raise ValueError(
                f"Unsupported guidance_scale_strategy: {self.guidance_scale_strategy}"
            )
        if self.max_sampled_guidance_scale < 1.0:
            raise ValueError(
                "max_sampled_guidance_scale must be >= 1.0 for sampled guidance."
            )
        return self.sample_cfg_scale(bz, s_max=self.max_sampled_guidance_scale - 1.0)

    def _effective_training_guidance_scale(self, t, w, t_min, t_max, current_step=None):
        w_eff = jnp.where((t >= t_min) & (t <= t_max), w, 1.0)
        if current_step is not None:
            current_step = jnp.asarray(current_step)
            guidance_enabled = current_step >= jnp.asarray(
                self.training_guidance_start_step, dtype=current_step.dtype
            )
            w_eff = jnp.where(guidance_enabled, w_eff, jnp.ones_like(w_eff))
        return w_eff

    def _effective_training_guidance_blend(self, t, w, t_min, t_max, current_step=None):
        if self._uses_baked_fixed_guidance_sampling():
            alpha = jnp.full_like(w, self.baked_guidance_blend, dtype=jnp.float32)
            alpha = jnp.where((t >= t_min) & (t <= t_max), alpha, jnp.zeros_like(alpha))
            if current_step is not None:
                current_step = jnp.asarray(current_step)
                guidance_enabled = current_step >= jnp.asarray(
                    self.training_guidance_start_step, dtype=current_step.dtype
                )
                alpha = jnp.where(guidance_enabled, alpha, jnp.zeros_like(alpha))
            return alpha

        w_eff = self._effective_training_guidance_scale(
            t, w, t_min, t_max, current_step=current_step
        )
        return 1.0 - 1.0 / w_eff

    def guided_u_fn(self, x, t, r, omega, t_min, t_max, y):
        """
        Compute a classifier-free guided average velocity for single-head DMF sampling.

        Guidance is applied on the first three channels to mirror the SiT-compatible
        preview path; the remaining channel is taken from the conditioned branch.
        """
        bz = x.shape[0]
        y_null = jnp.full((bz,), self.num_classes, dtype=jnp.int32)
        x_cat = jnp.concatenate([x, x], axis=0)
        y_cat = jnp.concatenate([y, y_null], axis=0)
        t_cat = jnp.concatenate([t, t], axis=0)
        h = t - r
        h_cat = jnp.concatenate([h, h], axis=0)
        omega_cat = jnp.concatenate([omega, omega], axis=0)
        t_min_cat = jnp.concatenate([t_min, t_min], axis=0)
        t_max_cat = jnp.concatenate([t_max, t_max], axis=0)
        u_cat, _ = self.u_fn(
            x_cat,
            t_cat,
            h_cat,
            omega_cat,
            t_min_cat,
            t_max_cat,
            y_cat,
        )
        u_c, u_u = jnp.split(u_cat, 2, axis=0)
        omega_scale = omega.reshape((bz, 1, 1, 1))
        guided_first_three = u_u[..., :3] + omega_scale * (u_c[..., :3] - u_u[..., :3])
        return jnp.concatenate([guided_first_three, u_c[..., 3:]], axis=-1)

    #######################################################
    #                       Solver                        #
    #######################################################

    def sample_one_step(self, z_t, labels, i, t_steps, omega, t_min, t_max):
        """
        Perform one sampling step given current state z_t at time step i.

        Args:
            z_t: Current noisy image at time step t.
            labels: Class labels for the batch.
            i: Current time step index.
            t_steps: Array of time steps.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
        """
        t = jnp.take(t_steps, i)
        r = jnp.take(t_steps, i + 1)
        bsz = z_t.shape[0]

        t = jnp.broadcast_to(t, (bsz,))
        r = jnp.broadcast_to(r, (bsz,))
        omega = jnp.broadcast_to(omega, (bsz,))
        t_min = jnp.broadcast_to(t_min, (bsz,))
        t_max = jnp.broadcast_to(t_max, (bsz,))

        if self._uses_auxiliary_v_head():
            u = self.u_fn(z_t, t, t - r, omega, t_min, t_max, y=labels)[0]
        elif self._uses_baked_fixed_guidance_sampling():
            # In the baked fixed-guidance regime the model already predicts the
            # desired guided field, so sampling should not re-apply external CFG.
            u = self.u_fn(z_t, t, t - r, jnp.ones_like(omega), t_min, t_max, y=labels)[0]
        else:
            u = self.guided_u_fn(z_t, t, r, omega, t_min, t_max, labels)

        return z_t + jnp.einsum("n,n...->n...", r - t, u)

    #######################################################
    #                       Schedule                      #
    #######################################################

    def logit_normal_dist(self, bz):
        rnd_normal = jax.random.normal(
            self.make_rng("gen"), [bz, 1, 1, 1], dtype=self.dtype
        )
        return nn.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def sample_tr(self, bz):
        """
        Sample t and r from logit-normal distribution.
        """
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        if self._uses_sit_dmf_time_convention():
            t, r = jnp.minimum(t, r), jnp.maximum(t, r)
        else:
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)

        data_size = int(bz * self.data_proportion)
        fm_mask = jnp.arange(bz) < data_size
        fm_mask = fm_mask.reshape(bz, 1, 1, 1)
        r = jnp.where(fm_mask, t, r)

        return t, r, fm_mask

    def sample_split_tr(self, bz):
        """
        Sample strictly off-diagonal intervals for SplitMeanFlow consistency.
        """
        t = self.logit_normal_dist(bz)
        r = self.logit_normal_dist(bz)
        if self._uses_sit_dmf_time_convention():
            t, r = jnp.minimum(t, r), jnp.maximum(t, r)
        else:
            t, r = jnp.maximum(t, r), jnp.minimum(t, r)
        fm_mask = jnp.zeros((bz, 1, 1, 1), dtype=bool)
        return t, r, fm_mask

    def sample_caimf_tr(self, bz, interval_eps=1e-3):
        """Sample a non-degenerate interval for adversarial finite differences."""
        if self._uses_sit_dmf_time_convention():
            raise ValueError(
                "Continuous adversarial iMF currently supports the standard "
                "iMF-DiT time convention only."
            )
        t, r, _ = self.sample_split_tr(bz)
        eps = jnp.asarray(interval_eps, dtype=self.dtype)
        t = jnp.maximum(t, eps)
        r = jnp.minimum(r, t - eps)
        r = jnp.maximum(r, 0.0)
        fm_mask = jnp.zeros((bz, 1, 1, 1), dtype=bool)
        return t, r, fm_mask

    def sample_cfg_scale(self, bz, s_max=7.0):
        """
        Sample CFG scale omega from power distribution.
        """
        ukey = self.make_rng("gen")
        u = jax.random.uniform(
            ukey, (bz, 1, 1, 1), minval=0.0, maxval=1.0, dtype=jnp.float32
        )

        if self.cfg_beta == 1.0:
            s = jnp.exp(u * jnp.log1p(jnp.asarray(s_max, jnp.float32)))
        else:
            smax = jnp.asarray(s_max, jnp.float32)
            b = jnp.asarray(self.cfg_beta, jnp.float32)

            log_base = (1.0 - b) * jnp.log1p(smax)
            log_inner = jnp.log1p(u * jnp.expm1(log_base))

            s = jnp.exp(log_inner / (1.0 - b))

        return jnp.asarray(s, jnp.float32)

    def sample_cfg_interval(self, bz, fm_mask=None):
        """
        Sample CFG interval [t_min, t_max] from uniform distribution.
        """
        if self.training_guidance_interval_strategy == "fixed":
            t_min = jnp.full((bz, 1, 1, 1), self.training_guidance_t_min, dtype=self.dtype)
            t_max = jnp.full((bz, 1, 1, 1), self.training_guidance_t_max, dtype=self.dtype)
            return t_min, t_max
        if self.training_guidance_interval_strategy != "sampled":
            raise ValueError(
                "Unsupported training_guidance_interval_strategy: "
                f"{self.training_guidance_interval_strategy}"
            )

        rng_start, rng_end = jax.random.split(self.make_rng("gen"))

        t_min = jax.random.uniform(
            rng_start, (bz, 1, 1, 1), minval=0.0, maxval=0.5, dtype=self.dtype
        )
        t_max = jax.random.uniform(
            rng_end, (bz, 1, 1, 1), minval=0.5, maxval=1.0, dtype=self.dtype
        )

        t_min = jnp.where(fm_mask, 0.0, t_min)
        t_max = jnp.where(fm_mask, 1.0, t_max)

        return t_min, t_max

    def sample_split_midpoint_ratio(self, bz):
        """
        Sample lambda in m = t + lambda * (r - t), so m lies strictly inside
        the interval between t and r when using the default uniform strategy.
        """
        strategy = self.split_consistency_midpoint_strategy
        if strategy == "midpoint":
            return jnp.full((bz, 1, 1, 1), 0.5, dtype=self.dtype)
        if strategy != "uniform":
            raise ValueError(
                "Unsupported split_consistency_midpoint_strategy: "
                f"{strategy}"
            )

        eps = jnp.asarray(self.split_consistency_midpoint_eps, dtype=self.dtype)
        if not (0.0 <= float(self.split_consistency_midpoint_eps) < 0.5):
            raise ValueError(
                "split_consistency_midpoint_eps must be in [0, 0.5)."
            )
        return jax.random.uniform(
            self.make_rng("gen"),
            (bz, 1, 1, 1),
            minval=eps,
            maxval=1.0 - eps,
            dtype=self.dtype,
        )

    def sample_split_source_mask(self, bz, prob):
        if not (0.0 <= prob <= 1.0):
            raise ValueError("split consistency source probabilities must be in [0, 1].")
        if prob == 0.0:
            return jnp.zeros((bz, 1, 1, 1), dtype=bool)
        if prob == 1.0:
            return jnp.ones((bz, 1, 1, 1), dtype=bool)
        return jax.random.uniform(
            self.make_rng("gen"),
            (bz, 1, 1, 1),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        ) < prob

    def sample_split_boundary_r(self, t):
        mode = self.split_consistency_boundary_mode
        if mode == "exact":
            return t
        if mode != "near_boundary":
            raise ValueError(
                f"Unsupported split_consistency_boundary_mode: {mode}"
            )

        dist = self.split_consistency_boundary_epsilon_distribution
        if dist != "half_normal":
            raise ValueError(
                "Unsupported split_consistency_boundary_epsilon_distribution: "
                f"{dist}"
            )

        eps = jnp.asarray(self.split_consistency_boundary_epsilon, dtype=self.dtype)
        eps_min = jnp.asarray(
            self.split_consistency_boundary_epsilon_min, dtype=self.dtype
        )
        delta = jnp.abs(
            jax.random.normal(self.make_rng("gen"), t.shape, dtype=self.dtype) * eps
        )

        if self._uses_sit_dmf_time_convention():
            max_delta = 1.0 - t
            delta = jnp.minimum(delta, max_delta)
            delta = jnp.where(max_delta > 0.0, jnp.maximum(delta, eps_min), 0.0)
            return t + delta

        max_delta = t
        delta = jnp.minimum(delta, max_delta)
        delta = jnp.where(max_delta > 0.0, jnp.maximum(delta, eps_min), 0.0)
        return t - delta

    def sample_split_consistency_tr(self, bz):
        """
        Sample intervals for SplitMeanFlow. The `data_proportion` subset keeps
        the old boundary-vs-split mixture, but boundary samples can use either
        exact diagonal intervals (`r=t`) or near-boundary intervals according to
        `split_consistency_boundary_mode`.
        """
        t, r, fm_mask = self.sample_tr(bz)
        if self.split_consistency_boundary_mode == "exact":
            return t, r, fm_mask
        r_boundary = self.sample_split_boundary_r(t)
        r = jnp.where(fm_mask, r_boundary, r)
        return t, r, fm_mask

    #######################################################
    #               Training Utils & Guidance             #
    #######################################################

    def u_fn(self, x, t, h, omega, t_min, t_max, y):
        """
        Compute the predicted u component from the model.
        In dual-head mode this returns (u, v_head). In single-head mode it
        returns (u, u_boundary), where u_boundary is u(x_t, t, t, y).

        Args:
            x: Noisy image at time t.
            t: Current time step.
            h: Time difference t - r.
            omega: CFG scale.
            t_min, t_max: Guidance interval.
            y: Class labels.
        Returns: (u, v_boundary)
            u: Predicted u (average velocity field).
            v_boundary: Auxiliary v prediction in dual-head mode, or the
                single-head boundary estimate in single-head mode.
        """
        bz = x.shape[0]
        if self._uses_auxiliary_v_head():
            return self.net(
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )

        if self._uses_imf_dit_backbone():
            u, _ = self.net(
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )
            v_boundary, _ = self.net(
                x,
                t.reshape(bz),
                jnp.zeros_like(t).reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )
            return u, v_boundary

        r = t - h
        u = self._predict_target_velocity(x, t, r, omega, t_min, t_max, y)
        v_boundary = self._predict_target_velocity(x, t, t, omega, t_min, t_max, y)
        return u, v_boundary

    def afm_u_fn(self, x, r, t, omega, t_min, t_max, y):
        """Direct average-velocity prediction for AFM, without JVP or v loss."""
        if not self._uses_plain_imf_dit_backbone():
            raise ValueError("AFM currently supports the plain imfDiT backbone only.")
        return self.net.predict_u_only(x, t, t - r, omega, t_min, t_max, y)

    def v_cond_fn(self, x, t, omega, y):
        """
        Compute the predicted v component conditioned on class labels.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            omega: CFG scale.
            y: Class labels.
        
        Returns:
            v: Predicted v component.
        """

        h = jnp.zeros_like(t)
        t_min = jnp.zeros_like(t)
        t_max = jnp.ones_like(t)
        return self.u_fn(x, t, h, omega, t_min, t_max, y=y)[1]

    def v_fn(self, x, t, y):
        """
        Compute the conditioned and unconditioned v components used in CFG.

        Args:
            x: Noisy image at time t.
            t: Current time step.
            y: Class labels.

        Returns:
            v_c: Predicted conditioned v component evaluated at unit guidance.
            v_u: Predicted unconditioned v component evaluated at unit guidance.
        """
        bz = x.shape[0]

        x = jnp.concatenate([x, x], axis=0)
        y_null = jnp.full((bz,), self.num_classes, dtype=jnp.int32)
        y = jnp.concatenate([y, y_null], axis=0)
        t = jnp.concatenate([t, t], axis=0)
        omega = jnp.ones_like(t)

        out = self.v_cond_fn(x, t, omega, y)
        v_c, v_u = jnp.split(out, 2, axis=0)

        return v_c, v_u

    def source_u_fn(self, source_params, x, t, h, omega, t_min, t_max, y):
        """
        Compute frozen-source average velocity for source regularization.
        """
        if source_params is None:
            raise ValueError(
                "source_params must be provided when training_mode="
                "imf_jvp_free_src_reg."
            )

        source_param_tree = self._resolve_source_params(source_params)
        bz = x.shape[0]
        if self._uses_auxiliary_v_head():
            u, _ = self.source_net.apply(
                {"params": source_param_tree},
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )
            return u

        if self._uses_imf_dit_backbone():
            u, _ = self.source_net.apply(
                {"params": source_param_tree},
                x,
                t.reshape(bz),
                h.reshape(bz),
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
                y,
            )
            return u

        r = t - h
        if self._uses_sit_guidance_context_conditioning():
            return self.source_net.apply(
                {"params": source_param_tree},
                x,
                t.reshape(bz),
                r.reshape(bz),
                y,
                omega.reshape(bz),
                t_min.reshape(bz),
                t_max.reshape(bz),
            )
        if self._uses_sit_adaln_guidance_scale_conditioning():
            return self.source_net.apply(
                {"params": source_param_tree},
                x,
                t.reshape(bz),
                r.reshape(bz),
                y,
                omega.reshape(bz),
            )

        del omega, t_min, t_max
        return self.source_net.apply(
            {"params": source_param_tree},
            x,
            t.reshape(bz),
            r.reshape(bz),
            y,
        )

    def source_v_cond_fn(self, source_params, x, t, omega, y):
        """
        Compute a source-model velocity prediction from a frozen source model.

        The interface is intentionally velocity-based so other source families can
        later be adapted behind the same abstraction.
        """
        if source_params is None:
            raise ValueError("source_params must be provided when use_dogfit=True.")
        if self.source_prediction_space == "noise":
            if not self._uses_plain_exact_source_backbone(self.source_model_str):
                raise NotImplementedError(
                    "source_prediction_space='noise' currently requires a plain "
                    f"exact source backbone, got {self.source_model_str!r}."
                )
            del omega
            raw_output = self._source_predict_backbone_output(source_params, x, t, y)
            return self._source_noise_to_velocity(raw_output, x.astype(self.dtype), t)
        if self.source_prediction_space != "v":
            raise NotImplementedError(
                f"Unsupported source_prediction_space: {self.source_prediction_space}"
            )

        source_param_tree = self._resolve_source_params(source_params)
        bz = x.shape[0]
        t_batch = self._batch_scalar(t, bz, dtype=self.dtype)
        omega_batch = self._batch_scalar(omega, bz, dtype=self.dtype)
        if self._uses_auxiliary_v_head():
            h = jnp.zeros_like(t_batch)
            t_min = jnp.zeros_like(t_batch)
            t_max = jnp.ones_like(t_batch)
            if is_v_only_param_tree(source_param_tree):
                v = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    h,
                    omega_batch,
                    t_min,
                    t_max,
                    y,
                    method=self.source_net.predict_v_only,
                )
            else:
                _, v = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    h,
                    omega_batch,
                    t_min,
                    t_max,
                    y,
                )
        else:
            if self._uses_sit_guidance_context_conditioning():
                t_min = jnp.zeros_like(t_batch)
                t_max = jnp.ones_like(t_batch)
                v = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    t_batch,
                    y,
                    omega_batch,
                    t_min,
                    t_max,
                )
            elif self._uses_sit_adaln_guidance_scale_conditioning():
                v = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    t_batch,
                    y,
                    omega_batch,
                )
            elif self._uses_imf_dit_backbone():
                v, _ = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    jnp.zeros_like(t_batch),
                    omega_batch,
                    jnp.zeros_like(t_batch),
                    jnp.ones_like(t_batch),
                    y,
                )
            else:
                del omega
                v = self.source_net.apply(
                    {"params": source_param_tree},
                    x,
                    t_batch,
                    t_batch,
                    y,
                )
        return v

    def teacher_v_cond_fn(self, teacher_params, x, t, omega, y):
        """
        Compute a conditioned v prediction from an alternate frozen parameter tree.

        This is used for teacher-style targets such as EMA-conditioned v_c.
        """
        if teacher_params is None:
            raise ValueError("teacher_params must be provided when use_ema_vc=True.")

        teacher_param_tree = self._resolve_source_params(teacher_params)
        bz = x.shape[0]
        t_batch = self._batch_scalar(t, bz, dtype=self.dtype)
        omega_batch = self._batch_scalar(omega, bz, dtype=self.dtype)
        if self._uses_auxiliary_v_head():
            h = jnp.zeros_like(t_batch)
            t_min = jnp.zeros_like(t_batch)
            t_max = jnp.ones_like(t_batch)
            if is_v_only_param_tree(teacher_param_tree):
                v = self.net.apply(
                    {"params": teacher_param_tree},
                    x,
                    t_batch,
                    h,
                    omega_batch,
                    t_min,
                    t_max,
                    y,
                    method=self.net.predict_v_only,
                )
            else:
                _, v = self.net.apply(
                    {"params": teacher_param_tree},
                    x,
                    t_batch,
                    h,
                    omega_batch,
                    t_min,
                    t_max,
                    y,
                )
            return v

        if self._uses_sit_guidance_context_conditioning():
            t_min = jnp.zeros_like(t_batch)
            t_max = jnp.ones_like(t_batch)
            return self.net.apply(
                {"params": teacher_param_tree},
                x,
                t_batch,
                t_batch,
                y,
                omega_batch,
                t_min,
                t_max,
            )
        if self._uses_sit_adaln_guidance_scale_conditioning():
            return self.net.apply(
                {"params": teacher_param_tree},
                x,
                t_batch,
                t_batch,
                y,
                omega_batch,
            )
        if self._uses_imf_dit_backbone():
            v, _ = self.net.apply(
                {"params": teacher_param_tree},
                x,
                t_batch,
                jnp.zeros_like(t_batch),
                omega_batch,
                jnp.zeros_like(t_batch),
                jnp.ones_like(t_batch),
                y,
            )
            return v

        del omega
        return self.net.apply(
            {"params": teacher_param_tree},
            x,
            t_batch,
            t_batch,
            y,
        )

    def source_v_uncond_fn(self, source_params, x, t):
        bz = x.shape[0]
        y_null = jnp.full((bz,), self.source_num_classes, dtype=jnp.int32)
        omega = jnp.ones_like(t)
        return self.source_v_cond_fn(source_params, x, t, omega, y_null)

    def cond_drop(self, v_t, v_g, labels):
        """
        Drop class labels with a certain probability for CFG.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            v_g: Guided instantaneous velocity at time t.
            labels: Class labels for the batch.

        Returns:
            labels: Possibly dropped class labels.
            v_g: Modified guided instantaneous velocity at time t. For samples
                 with dropped labels, v_g = v_t.
        """
        if (not self.target_use_null_class) or self.class_dropout_prob <= 0:
            return labels, v_g

        bz = v_t.shape[0]

        rand_mask = (
            jax.random.uniform(self.make_rng("gen"), shape=(bz,))
            < self.class_dropout_prob
        )
        num_drop = jnp.sum(rand_mask).astype(jnp.int32)
        drop_mask = jnp.arange(bz)[:, None, None, None] < num_drop

        labels = jnp.where(
            drop_mask.reshape(bz),
            self.num_classes,
            labels,
        )
        v_g = jnp.where(drop_mask, v_t, v_g)

        return labels, v_g

    def guidance_fn(
        self,
        v_t,
        z_t,
        t,
        r,
        y,
        fm_mask,
        w,
        t_min,
        t_max,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        Compute the guided velocity v_g using classifier-free guidance.

        Args:
            v_t: Unguided instantaneous velocity at time t.
            z_t: Noisy image at time t.
            t, r: Two time steps.
            y: Class labels.
            fm_mask: Mask for t=r samples, i.e., flow matching samples.
            t_min, t_max: Guidance interval.
            w: CFG scale.

        Returns:
            v_g: Guided instantaneous velocity at time t, as target for training.
            v_c: Conditioned instantaneous velocity at time t, for jvp computation.
        """

        del r, fm_mask  # This method variant uses one interval-adjusted v_c everywhere.

        if not self.use_training_guidance:
            if self.use_ema_vc:
                v_c = self.teacher_v_cond_fn(
                    teacher_params,
                    z_t,
                    t,
                    jnp.ones_like(w),
                    y=y,
                )
            else:
                v_c = self.v_cond_fn(z_t, t, jnp.ones_like(w), y=y)
            v_c = jax.lax.stop_gradient(v_c)
            return v_t, v_c

        guidance_blend = self._effective_training_guidance_blend(
            t, w, t_min, t_max, current_step=current_step
        )

        if self.use_dogfit:
            if self.use_ema_vc:
                v_c = self.teacher_v_cond_fn(
                    teacher_params,
                    z_t,
                    t,
                    jnp.ones_like(w),
                    y=y,
                )
            else:
                v_c = self.v_cond_fn(z_t, t, jnp.ones_like(w), y=y)
            v_u = self.source_v_uncond_fn(source_params, z_t, t)
            v_c = jax.lax.stop_gradient(v_c)
            v_u = jax.lax.stop_gradient(v_u)
        else:
            v_c, v_u = self.v_fn(z_t, t, y=y)

        if self._uses_sit_cfg_channel_rule():
            guided_first_three = (
                v_t[..., :3]
                + guidance_blend * (v_c[..., :3] - v_u[..., :3])
            )
            v_g = jnp.concatenate([guided_first_three, v_t[..., 3:]], axis=-1)
        else:
            v_g = v_t + guidance_blend * (v_c - v_u)

        return v_g, v_c

    #######################################################
    #               Forward Pass and Loss                 #
    #######################################################

    def forward(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        Forward process of improved MeanFlow and compute loss.

        Args:
            images: A batch of images, shape (B, H, W, C).
            labels: Corresponding class labels, shape (B,).
        
        Returns:
            loss: Scalar loss value.
            dict_losses: Dictionary of individual loss components.
        """
        if self.training_mode == "imf_jvp":
            return self.forward_imf_jvp(
                images,
                labels,
                source_params=source_params,
                teacher_params=teacher_params,
                current_step=current_step,
            )
        if self.training_mode == "imf_jvp_free_src_reg":
            return self.forward_imf_jvp_free_src_reg(
                images,
                labels,
                source_params=source_params,
                teacher_params=teacher_params,
                current_step=current_step,
            )
        if self.training_mode == "imf_split_consistency":
            return self.forward_imf_split_consistency(
                images,
                labels,
                source_params=source_params,
                teacher_params=teacher_params,
                current_step=current_step,
            )
        raise ValueError(f"Unsupported training_mode: {self.training_mode}")

    def forward_imf_jvp(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        Forward process of improved MeanFlow and compute loss.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        # Instantaneous velocity computation
        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        # Sample CFG scale and interval
        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )

        # Compute guided velocity v_g and conditioned velocity v_c
        v_g, v_c = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            teacher_params=teacher_params,
            current_step=current_step,
        )

        # Cond dropout (dropout class labels)
        labels, _ = self.cond_drop(v_t, v_g, labels)
        v_g = jax.lax.stop_gradient(v_g)

        # Warped u-function for jvp computation
        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, model_omega, t_min, t_max, y=labels)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)

        # Different from original MeanFlow, we use predicted v in the jvp
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)

        # Our compound function V = u + (t - r) * du/dt
        V = u + self._mf_target_interval_coeff(t, r) * jax.lax.stop_gradient(du_dt)

        v_t = jax.lax.stop_gradient(v_t)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        # improved MeanFlow objective is conceptually v-loss
        loss_u = jnp.sum((V - v_g) ** 2, axis=(1, 2, 3))
        loss_u = adp_wt_fn(loss_u)

        # auxiliary v-head loss, or single-head boundary loss
        loss_v = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        loss = loss_u + loss_v
        loss = loss.mean()  # mean over batch

        dict_losses = {
            "loss": loss,
            "loss_u": jnp.mean((V - v_g) ** 2),
            "loss_v": jnp.mean((v - v_g) ** 2),
        }

        return loss, dict_losses

    def forward_caimf_generator_terms(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
        interval_eps=1e-3,
    ):
        """Compute iMF loss and finite-interval samples in one generator pass."""
        if not self._uses_plain_imf_dit_backbone():
            raise ValueError(
                "Continuous adversarial iMF currently requires a plain imfDiT backbone."
            )

        x = images.astype(self.dtype)
        bz = images.shape[0]
        t, r, fm_mask = self.sample_caimf_tr(bz, interval_eps=interval_eps)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        z_t = (1.0 - t) * x + t * e
        z_r = (1.0 - r) * x + r * e
        v_t = e - x

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )

        v_g, v_c = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            teacher_params=teacher_params,
            current_step=current_step,
        )
        labels, _ = self.cond_drop(v_t, v_g, labels)
        v_g = jax.lax.stop_gradient(v_g)

        def u_fn(z_t_value, t_value, r_value):
            return self.u_fn(
                z_t_value,
                t_value,
                t_value - r_value,
                model_omega,
                t_min,
                t_max,
                y=labels,
            )

        u, du_dt, v = jax.jvp(
            u_fn,
            (z_t, t, r),
            (v_c, jnp.ones_like(t), jnp.zeros_like(t)),
            has_aux=True,
        )
        interval = t - r
        compound_velocity = u + interval * jax.lax.stop_gradient(du_dt)

        def adaptive_weight(loss_value):
            weight = (loss_value + self.norm_eps) ** self.norm_p
            return loss_value / jax.lax.stop_gradient(weight)

        loss_u_per_example = jnp.sum(
            (compound_velocity - v_g) ** 2, axis=(1, 2, 3)
        )
        loss_v_per_example = jnp.sum((v - v_g) ** 2, axis=(1, 2, 3))
        loss_imf = jnp.mean(
            adaptive_weight(loss_u_per_example)
            + adaptive_weight(loss_v_per_example)
        )
        xhat_r = z_t - interval * u

        return {
            "loss_imf": loss_imf,
            "loss_u": jnp.mean((compound_velocity - v_g) ** 2),
            "loss_v": jnp.mean((v - v_g) ** 2),
            "u": u,
            "x_t": z_t,
            "x_r": z_r,
            "xhat_r": xhat_r,
            "t": t.reshape(bz),
            "r": r.reshape(bz),
            "labels": labels,
        }

    def forward_caimf_discriminator_samples(
        self,
        images,
        labels,
        current_step=None,
        interval_eps=1e-3,
    ):
        """Generate real and reconstructed interval endpoints for a D update."""
        if not self._uses_plain_imf_dit_backbone():
            raise ValueError(
                "Continuous adversarial iMF currently requires a plain imfDiT backbone."
            )

        x = images.astype(self.dtype)
        bz = images.shape[0]
        t, r, fm_mask = self.sample_caimf_tr(bz, interval_eps=interval_eps)
        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        z_t = (1.0 - t) * x + t * e
        z_r = (1.0 - r) * x + r * e

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )
        # CAFM applies the same class dropout to discriminator and generator
        # batches. This also teaches D the null-class condition used by G.
        labels, _ = self.cond_drop(z_t, z_t, labels)
        u, _ = self.u_fn(
            z_t,
            t,
            t - r,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        xhat_r = z_t - (t - r) * u
        return {
            "x_t": z_t,
            "x_r": z_r,
            "xhat_r": xhat_r,
            "t": t.reshape(bz),
            "r": r.reshape(bz),
            "labels": labels,
        }

    def forward_imf_jvp_free_src_reg(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        JVP-free fine-tuning with diagonal instant-velocity supervision and
        off-diagonal frozen-source regularization.
        """
        x = images.astype(self.dtype)
        del teacher_params
        bz = images.shape[0]

        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )

        v_g, _ = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            current_step=current_step,
        )
        labels, _ = self.cond_drop(v_t, v_g, labels)

        student_u, student_v = self.u_fn(
            z_t,
            t,
            t - r,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        source_u = self.source_u_fn(
            source_params,
            z_t,
            t,
            t - r,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        source_u = jax.lax.stop_gradient(source_u)
        v_t = jax.lax.stop_gradient(v_t)

        diag_mask = fm_mask.astype(self.dtype)
        src_mask = 1.0 - diag_mask

        def masked_mean(values, mask):
            denom = jnp.maximum(jnp.sum(mask), 1.0)
            return jnp.sum(values * mask.reshape((bz,))) / denom

        inst_per_example = jnp.mean((student_v - v_g) ** 2, axis=(1, 2, 3))
        src_per_example = jnp.mean((student_u - source_u) ** 2, axis=(1, 2, 3))

        loss_inst = masked_mean(inst_per_example, diag_mask)
        loss_src = masked_mean(src_per_example, src_mask)
        loss = jnp.mean(
            jnp.where(
                fm_mask.reshape((bz,)),
                inst_per_example,
                src_per_example,
            )
        )

        dict_losses = {
            "loss": loss,
            "loss_inst": loss_inst,
            "loss_src": loss_src,
            "t_mean": jnp.mean(t),
            "r_mean": jnp.mean(r),
        }

        return loss, dict_losses

    def forward_imf_split_consistency(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        JVP-free SplitMeanFlow training: regress the long-interval average
        velocity toward the detached convex combination of the two sub-interval
        average velocities.
        """
        x = images.astype(self.dtype)
        del teacher_params
        bz = images.shape[0]

        t, r, fm_mask = self.sample_split_consistency_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )

        v_g, _ = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            current_step=current_step,
        )
        labels, _ = self.cond_drop(v_t, v_g, labels)

        lam = self.sample_split_midpoint_ratio(bz)
        m = t + lam * (r - t)
        use_source_first = self.sample_split_source_mask(
            bz, self.split_consistency_source_first_prob
        )
        use_source_second = self.sample_split_source_mask(
            bz, self.split_consistency_source_second_prob
        )

        u_long, v = self.u_fn(
            z_t,
            t,
            t - r,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        u_first_student, _ = self.u_fn(
            z_t,
            t,
            t - m,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        if self._uses_split_consistency_source_ablation():
            u_first_source = self.source_u_fn(
                source_params,
                z_t,
                t,
                t - m,
                model_omega,
                t_min,
                t_max,
                y=labels,
            )
            u_first_target = jnp.where(use_source_first, u_first_source, u_first_student)
        else:
            u_first_target = u_first_student

        z_m = z_t + (m - t) * jax.lax.stop_gradient(u_first_target)
        u_second_student, _ = self.u_fn(
            z_m,
            m,
            m - r,
            model_omega,
            t_min,
            t_max,
            y=labels,
        )
        if self._uses_split_consistency_source_ablation():
            u_second_source = self.source_u_fn(
                source_params,
                z_m,
                m,
                m - r,
                model_omega,
                t_min,
                t_max,
                y=labels,
            )
            u_second_target = jnp.where(
                use_source_second,
                u_second_source,
                u_second_student,
            )
        else:
            u_second_target = u_second_student

        split_target = jax.lax.stop_gradient(
            lam * u_first_target + (1.0 - lam) * u_second_target
        )
        v_t = jax.lax.stop_gradient(v_t)

        def adp_wt_fn(loss):
            adp_wt = (loss + self.norm_eps) ** self.norm_p
            return loss / jax.lax.stop_gradient(adp_wt)

        boundary_per_example = jnp.sum((u_long - v_t) ** 2, axis=(1, 2, 3))
        split_per_example = jnp.sum((u_long - split_target) ** 2, axis=(1, 2, 3))
        loss_u = jnp.where(
            fm_mask.reshape((bz,)),
            boundary_per_example,
            split_per_example,
        )
        loss_u = adp_wt_fn(loss_u)

        loss_v = jnp.sum((v - v_t) ** 2, axis=(1, 2, 3))
        loss_v = adp_wt_fn(loss_v)

        loss = (loss_u + loss_v).mean()

        dict_losses = {
            "loss": loss,
            "loss_u": jnp.mean(
                jnp.where(
                    fm_mask,
                    (u_long - v_t) ** 2,
                    (u_long - split_target) ** 2,
                )
            ),
            "loss_u_boundary": jnp.mean((u_long - v_t) ** 2),
            "loss_u_split": jnp.mean((u_long - split_target) ** 2),
            "loss_v": jnp.mean((v - v_t) ** 2),
            "m_mean": jnp.mean(m),
            "boundary_dt_mean": jnp.sum(
                jnp.abs(r - t) * fm_mask.astype(self.dtype)
            ) / jnp.maximum(jnp.sum(fm_mask.astype(self.dtype)), 1.0),
            "split_lambda_mean": jnp.mean(lam),
            "diag_fraction": jnp.mean(fm_mask.astype(self.dtype)),
            "source_first_fraction": jnp.mean(use_source_first.astype(self.dtype)),
            "source_second_fraction": jnp.mean(use_source_second.astype(self.dtype)),
        }

        return loss, dict_losses

    def debug_forward(
        self,
        images,
        labels,
        source_params=None,
        teacher_params=None,
        current_step=None,
    ):
        """
        Forward process with intermediate tensors exposed for debugging.
        """
        x = images.astype(self.dtype)
        bz = images.shape[0]

        t, r, fm_mask = self.sample_tr(bz)

        e = jax.random.normal(self.make_rng("gen"), x.shape, dtype=self.dtype)
        if self._uses_sit_dmf_time_convention():
            z_t = (1 - t) * e + t * x
            v_t = x - e
        else:
            z_t = (1 - t) * x + t * e
            v_t = e - x

        t_min, t_max = self.sample_cfg_interval(bz, fm_mask)
        omega = self._sample_guidance_scale(bz)
        model_omega = (
            self._effective_training_guidance_scale(
                t, omega, t_min, t_max, current_step=current_step
            )
            if self._uses_sit_adaln_guidance_scale_conditioning()
            else omega
        )

        if self.use_dogfit:
            v_u = self.source_v_uncond_fn(source_params, z_t, t)
        else:
            _, v_u = self.v_fn(z_t, t, y=labels)

        v_g, v_c = self.guidance_fn(
            v_t,
            z_t,
            t,
            r,
            labels,
            fm_mask,
            omega,
            t_min,
            t_max,
            source_params=source_params,
            teacher_params=teacher_params,
            current_step=current_step,
        )

        labels_after_drop, v_g = self.cond_drop(v_t, v_g, labels)

        def u_fn(z_t, t, r):
            return self.u_fn(z_t, t, t - r, model_omega, t_min, t_max, y=labels_after_drop)

        dtdt = jnp.ones_like(t)
        dtdr = jnp.zeros_like(t)
        u, du_dt, v = jax.jvp(u_fn, (z_t, t, r), (v_c, dtdt, dtdr), has_aux=True)
        V = u + self._mf_target_interval_coeff(t, r) * jax.lax.stop_gradient(du_dt)

        return {
            "x": x,
            "z_t": z_t,
            "v_t": v_t,
            "v_u": v_u,
            "v_c": v_c,
            "v_g": v_g,
            "v_pred": v,
            "V": V,
            "omega": omega,
            "w_eff_mean": jnp.mean(
                self._effective_training_guidance_scale(
                    t, omega, t_min, t_max, current_step=current_step
                )
            ),
            "guidance_blend_mean": jnp.mean(
                self._effective_training_guidance_blend(
                    t, omega, t_min, t_max, current_step=current_step
                )
            ),
            "t": t,
            "r": r,
            "t_min": t_min,
            "t_max": t_max,
            "fm_mask": fm_mask.astype(self.dtype),
        }

    def __call__(self, x, t, y):
        if self._uses_auxiliary_v_head():
            return self.net(x, t, t, t, t, t, y)  # initialization only
        if self._uses_sit_guidance_context_conditioning():
            ones = jnp.ones_like(t)
            zeros = jnp.zeros_like(t)
            return self.net(x, t, t, y, ones, zeros, ones)  # initialization only
        if self._uses_sit_adaln_guidance_scale_conditioning():
            ones = jnp.ones_like(t)
            return self.net(x, t, t, y, ones)  # initialization only
        if self._uses_imf_dit_backbone():
            ones = jnp.ones_like(t)
            zeros = jnp.zeros_like(t)
            return self.net(x, t, zeros, ones, zeros, ones, y)  # initialization only
        return self.net(x, t, t, y)  # initialization only

    def init_source(self, x, t, y):
        if not (
            self.use_dogfit
            or self._uses_src_reg_training_mode()
            or self._uses_split_consistency_source_ablation()
        ):
            return y

        if self._uses_plain_exact_source_backbone(self.source_model_str):
            t_model = t.astype(self.dtype)
            if self.source_model_time_flip:
                t_model = 1.0 - t_model
            t_model = t_model * jnp.asarray(
                self.source_model_time_scale,
                dtype=self.dtype,
            )
            return self.source_net(x.astype(self.dtype), t_model, y)

        if self._uses_auxiliary_v_head():
            zeros = jnp.zeros_like(t)
            ones = jnp.ones_like(t)
            return self.source_net(x, t, zeros, ones, zeros, ones, y)
        if self._uses_sit_guidance_context_conditioning():
            ones = jnp.ones_like(t)
            zeros = jnp.zeros_like(t)
            return self.source_net(x, t, t, y, ones, zeros, ones)
        if self._uses_sit_adaln_guidance_scale_conditioning():
            ones = jnp.ones_like(t)
            return self.source_net(x, t, t, y, ones)
        if self._uses_imf_dit_backbone():
            ones = jnp.ones_like(t)
            zeros = jnp.zeros_like(t)
            return self.source_net(x, t, zeros, ones, zeros, ones, y)
        return self.source_net(x, t, t, y)
