import logging as _logging
import time, os
import shutil
from collections import deque

import jax
from absl import logging

import numpy as np
from PIL import Image
import wandb


def log_for_0(*args):
    if jax.process_index() == 0:
        logging.info(*args, stacklevel=2)


class ExcludeInfo(_logging.Filter):
    def __init__(self, exclude_files):
        super().__init__()
        self.exclude_files = exclude_files

    def filter(self, record):
        if any(file_name in record.pathname for file_name in self.exclude_files):
            return record.levelno > _logging.INFO
        return True


exclude_files = [
    "orbax/checkpoint/async_checkpointer.py",
    "orbax/checkpoint/abstract_checkpointer.py",
    "orbax/checkpoint/multihost/utils.py",
    "orbax/checkpoint/future.py",
    "orbax/checkpoint/_src/handlers/base_pytree_checkpoint_handler.py",
    "orbax/checkpoint/type_handlers.py",
    "orbax/checkpoint/metadata/checkpoint.py",
    "orbax/checkpoint/metadata/sharding.py",
] + [
    "orbax/checkpoint/checkpointer.py",
    "flax/training/checkpoints.py",
] * jax.process_index()
file_filter = ExcludeInfo(exclude_files)


def supress_checkpt_info():
    logging.get_absl_handler().addFilter(file_filter)


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f"{self.elapse_with_reset():.2f} s"


class MetricsTracker:
    def __init__(self):
        self._sum = None  # tree of numpy arrays (host)
        self._n = 0  # number of steps accumulated on *this host*

    @staticmethod
    def _mean_over_local_devices(x):
        """
        Bring one leaf to host and average over local device axis if present.
        This avoids keeping per-device values around on host.
        """
        # device_get blocks on the computation that produced x.
        a = np.asarray(jax.device_get(x))
        # Under pmap, metrics often have shape [local_devices, ...].
        # If it's already scalar (0-D), leave unchanged.
        if a.ndim >= 1:  # treat leading axis as local device axis
            a = a.mean(axis=0)
        return a

    def update(self, metrics_step_tree):
        """
        Incorporate one step's metrics (per-replica JAX arrays) into the running sum.
        Call this once per training step.
        """
        local_mean = jax.tree_map(self._mean_over_local_devices, metrics_step_tree)
        if self._sum is None:
            self._sum = local_mean
        else:
            self._sum = jax.tree_map(lambda s, x: s + x, self._sum, local_mean)
        self._n += 1

    def finalize(self):
        """
        Return global mean over steps, devices, and hosts as a tree of Python floats.
        Resets internal state. Safe to call at any logging boundary.
        """
        if self._n == 0:
            return {}

        out = jax.tree_map(
            lambda s: float(np.asarray(s / self._n, dtype=np.float64).mean()),
            self._sum,
        )

        self._sum, self._n = None, 0
        return out


def _derive_wandb_tags(config):
    """Build readable, always-correct wandb tags from the live config.

    Avoids the stale hardcoded YAML tag list (which said "caltech101" for every
    dataset regardless of what was actually trained).
    """
    tags = []

    dataset = str(config.dataset.get("name", "") or "").strip()
    if dataset:
        tags.append(dataset)

    model_str = str(config.model.get("model_str", "") or "").strip()
    is_jit = "jit" in model_str.lower()
    is_imf = "imf" in model_str.lower()

    # SiT uses an explicit velocity-map mode; JiT is transport-v by construction.
    velocity_map = str(config.model.get("sit_velocity_map_mode", "") or "").strip()
    if is_jit:
        tags.append("transport-v")
    elif velocity_map == "dit_native":
        tags.append("ddpm-v")
    elif velocity_map == "transport":
        tags.append("transport-v")

    if model_str:
        tags.append(model_str)

    if is_imf:
        tags.append("plain-imf")
        tags.append("imf-init")
    elif is_jit:
        tags.append("plain-jit")
        tags.append("jit-init")
    else:
        tags.append("plain-sit")
        tags.append("dit-init")

    try:
        nfe = int(config.sampling.num_steps)
        tags.append("nfe%d" % nfe)
    except Exception:
        pass

    configured = list(config.logging.get("wandb_tags", []) or [])
    stale = {"caltech101", "ddpm-v", "transport-v", "plain-sit", "dit-init", "plain-jit", "jit-init", "plain-imf", "imf-init", "finetune"}
    for t in configured:
        if t not in stale:
            tags.append(str(t))

    seen = set()
    out = []
    for t in tags:
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out or None


class Writer:
    def __init__(self, config, workdir):
        if jax.process_index() != 0:
            return
        self.workdir = workdir
        self._wandb_requested = bool(config.logging.use_wandb)
        self.use_wandb = False
        self._wandb_config = config.to_dict()
        self._wandb_init_kwargs = {
            "name": config.logging.wandb_name if config.logging.wandb_name else None,
            "project": config.logging.wandb_project,
            "entity": config.logging.wandb_entity if config.logging.wandb_entity else None,
            "notes": config.logging.wandb_notes if config.logging.wandb_notes else None,
            "group": config.logging.get("wandb_group", "") or None,
            "tags": _derive_wandb_tags(config),
            "dir": "/tmp",  # avoid writing to workdir
            "settings": wandb.Settings(_service_wait=60),
        }
        self._wandb_run_id = None
        self._wandb_retry_count = 0
        self._wandb_max_retries = int(
            config.logging.get("wandb_max_retries", 3)
        )
        self._wandb_retry_cooldown_seconds = float(
            config.logging.get("wandb_retry_cooldown_seconds", 300)
        )
        self._wandb_retry_after = 0.0
        self._pending_eval_wandb_logs = deque(
            maxlen=int(config.logging.get("wandb_eval_replay_buffer_size", 100))
        )

        if self._wandb_requested:
            self._init_wandb(resume=False)
        else:
            log_for_0("Wandb logging is disabled. Images will be saved to disk.")

    def _init_wandb(self, resume):
        kwargs = dict(self._wandb_init_kwargs)
        if resume and self._wandb_run_id:
            kwargs["id"] = self._wandb_run_id
            kwargs["resume"] = "allow"

        try:
            run = wandb.init(**kwargs)
            self._wandb_run_id = getattr(run, "id", None) or getattr(
                wandb.run, "id", None
            )
            wandb.config.update(self._wandb_config, allow_val_change=True)
        except Exception as exc:
            self._pause_wandb_after_error(exc)
            return False

        self.use_wandb = True
        return True

    def _pause_wandb_after_error(self, exc):
        self.use_wandb = False
        self._wandb_retry_count += 1
        self._wandb_retry_after = time.time() + self._wandb_retry_cooldown_seconds
        logging.warning(
            "Pausing W&B logging after error: %s: %s. Retry %d/%d after %.0f seconds.",
            type(exc).__name__,
            exc,
            self._wandb_retry_count,
            self._wandb_max_retries,
            self._wandb_retry_cooldown_seconds,
        )

    def _maybe_resume_wandb(self):
        if self.use_wandb:
            return True
        if not self._wandb_requested:
            return False
        if self._wandb_retry_count >= self._wandb_max_retries:
            return False
        if time.time() < self._wandb_retry_after:
            return False

        logging.info(
            "Attempting to resume W&B logging for run id %s.",
            self._wandb_run_id,
        )
        try:
            wandb.finish()
        except Exception:
            pass
        return self._init_wandb(resume=True)

    def _safe_wandb_log(self, payload, step):
        if not self._maybe_resume_wandb():
            return False
        try:
            wandb.log(payload, step=step)
        except Exception as exc:
            self._pause_wandb_after_error(exc)
            return False
        return True

    @staticmethod
    def _is_eval_metric_payload(payload):
        if not isinstance(payload, dict):
            return False
        eval_key_fragments = (
            "fid",
            "fd_dino",
            "inception",
            "sampling_num_steps",
            "best_is",
        )
        return any(
            any(fragment in str(key).lower() for fragment in eval_key_fragments)
            for key in payload
        )

    def _buffer_eval_wandb_log(self, payload, step):
        if not self._wandb_requested or self._pending_eval_wandb_logs.maxlen == 0:
            return
        self._pending_eval_wandb_logs.append((step, dict(payload)))
        logging.warning(
            "Buffered eval metric payload for W&B replay at step %s. Pending eval logs: %d.",
            step,
            len(self._pending_eval_wandb_logs),
        )

    def _flush_pending_eval_wandb_logs(self):
        if not self._pending_eval_wandb_logs:
            return True
        if not self._maybe_resume_wandb():
            return False

        while self._pending_eval_wandb_logs:
            step, payload = self._pending_eval_wandb_logs.popleft()
            try:
                wandb.log(payload, step=step)
            except Exception as exc:
                self._pending_eval_wandb_logs.appendleft((step, payload))
                self._pause_wandb_after_error(exc)
                return False
        return True

    def _safe_wandb_log_with_replay(self, payload, step):
        buffer_on_failure = self._wandb_requested and self._is_eval_metric_payload(payload)
        if not self._maybe_resume_wandb():
            if buffer_on_failure:
                self._buffer_eval_wandb_log(payload, step)
            return False

        if not self._flush_pending_eval_wandb_logs():
            if buffer_on_failure:
                self._buffer_eval_wandb_log(payload, step)
            return False

        try:
            wandb.log(payload, step=step)
        except Exception as exc:
            self._pause_wandb_after_error(exc)
            if buffer_on_failure:
                self._buffer_eval_wandb_log(payload, step)
            return False
        return True

    @staticmethod
    def _to_pil_image(v):
        if isinstance(v, Image.Image):
            return v
        assert isinstance(v, np.ndarray), "Invalid image type {}".format(type(v))
        assert v.dtype == np.uint8, "Invalid image dtype {}".format(v.dtype)
        assert v.ndim == 3 and 3 in [
            v.shape[0],
            v.shape[2],
        ], "Invalid image shape {}".format(v.shape)
        if v.shape[0] == 3:
            v = v.transpose((1, 2, 0))
        return Image.fromarray(v)

    def write_scalars(self, step, scalar_dict):
        if jax.process_index() != 0:
            return
        log_str = f"[{step}]"
        for k, v in scalar_dict.items():
            log_str += f" {k}={v:.5g}," if isinstance(v, float) else f" {k}={v},"
        log_str = log_str.strip(",")
        logging.info(log_str)
        self._safe_wandb_log_with_replay(scalar_dict, step=step)

    def write_images(self, step, image_dict):
        if jax.process_index() != 0:
            return

        if not os.path.exists(f"{self.workdir}/images/"):
            os.makedirs(f"{self.workdir}/images/")
        pil_images = {k: self._to_pil_image(v) for k, v in image_dict.items()}
        for k, img in pil_images.items():
            img.save(f"{self.workdir}/images/{step}_{k}.png")

        if self._maybe_resume_wandb():
            self._safe_wandb_log(
                {k: wandb.Image(img) for k, img in pil_images.items()},
                step,
            )

    def write_image_grid(self, step, images, grid_size, key="image_grid"):
        if jax.process_index() != 0:
            return

        assert len(images) == grid_size ** 2, "Number of images must match grid size."
        images = [self._to_pil_image(img) for img in images]

        # Create a grid of images
        grid_image = Image.new('RGB', (grid_size * images[0].width, grid_size * images[0].height))
        for i, img in enumerate(images):
            x = (i % grid_size) * images[0].width
            y = (i // grid_size) * images[0].height
            grid_image.paste(img, (x, y))

        if not os.path.exists(f"{self.workdir}/image_grids/"):
            os.makedirs(f"{self.workdir}/image_grids/")
        grid_image.save(f"{self.workdir}/image_grids/{key}_{step}.png")

        if self._maybe_resume_wandb():
            self._safe_wandb_log({key: wandb.Image(grid_image)}, step)

    def close(self):
        if jax.process_index() != 0:
            return
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception as exc:
                logging.warning(
                    "Ignoring W&B finish error: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            finally:
                self.use_wandb = False
        shutil.rmtree("/tmp/wandb", ignore_errors=True)

    def __del__(self):
        return


def close_wandb():
    if jax.process_index() != 0:
        return
    try:
        wandb.finish()
    except Exception as exc:
        logging.warning(
            "Ignoring W&B finish error: %s: %s",
            type(exc).__name__,
            exc,
        )
    shutil.rmtree("/tmp/wandb", ignore_errors=True)
