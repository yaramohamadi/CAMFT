import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import torch
import torch.utils.data
from flax import jax_utils

from utils.logging_util import log_for_0


class LatentDist(
    object
):  # https://github.com/huggingface/diffusers/blob/v0.29.2/src/diffusers/models/vae_flax.py#L689
    """
    Class of Gaussian distribution.

    Method:
        sample: Sample from the distribution.
    """

    def __init__(self, parameters, deterministic=False):
        """
        parameters: concatenated mean and std
        """
        # Last axis to account for channels-last
        self.mean, self.std = jnp.split(parameters, 2, axis=-1)
        self.deterministic = deterministic
        if self.deterministic:
            self.var = self.std = jnp.zeros_like(self.mean)

    def sample(self, key):
        return self.mean + self.std * jax.random.normal(key, self.mean.shape)


class LatentManager:
    def __init__(
        self,
        vae_type,
        decode_batch_size,
        input_size,
        decode_num_local_devices=None,
    ):
        from diffusers.models import FlaxAutoencoderKL

        # init VAE
        vae, vae_params = FlaxAutoencoderKL.from_pretrained(
            f"pcuenq/sd-vae-ft-{vae_type}-flax"
        )
        self.vae = vae
        self.vae_params = vae_params
        self._vae_class = FlaxAutoencoderKL

        self.batch_size = decode_batch_size
        self.latent_size = input_size
        if decode_num_local_devices is None:
            decode_num_local_devices = jax.local_device_count()
        self.decode_num_local_devices = int(decode_num_local_devices)
        if self.decode_num_local_devices < 1:
            raise ValueError("decode_num_local_devices must be >= 1.")
        if self.decode_num_local_devices > jax.local_device_count():
            raise ValueError(
                f"Requested {self.decode_num_local_devices} VAE decode devices, "
                f"but only {jax.local_device_count()} local devices are visible."
            )
        self.decode_compiled_batch_size = int(self.batch_size)
        self.decode_compiled_total = (
            self.decode_compiled_batch_size * self.decode_num_local_devices
        )
        self.decode_devices = jax.local_devices()[: self.decode_num_local_devices]

        # create decode function
        self.decode_fn = self.get_decode_fn()

        self.mean = jnp.array([0.86488, -0.27787343, 0.21616915, 0.3738409]).reshape(
            1, -1, 1, 1
        )
        self.std = jnp.array([4.85503674, 5.31922414, 3.93725398, 3.9870003]).reshape(
            1, -1, 1, 1
        )

    def get_decode_fn(self):

        def dist_prepare_batch_data(batch):
            # reshape (host_batch_size, 3, height, width) to
            # (local_devices, device_batch_size, height, width, 3)
            local_device_count = self.decode_num_local_devices

            return_dict = {}
            for k, v in batch.items():
                v = v.reshape((local_device_count, -1) + v.shape[1:])
                return_dict[k] = v
            return return_dict

        log_for_0("Compiling vae.apply...")
        now = time.time()

        z_dummy = jnp.ones(
            (
                self.decode_num_local_devices,
                self.decode_compiled_batch_size,
                4,
                self.latent_size,
                self.latent_size,
            )
        )

        p_vae_variable = jax_utils.replicate(
            {"params": self.vae_params},
            devices=self.decode_devices,
        )
        p_decode_fn = partial(self.vae.apply, method=self._vae_class.decode)
        p_decode_fn = jax.pmap(
            p_decode_fn,
            axis_name="batch",
            devices=self.decode_devices,
        )

        lowered = p_decode_fn.lower(p_vae_variable, z_dummy)
        compiled_decod_fn = lowered.compile()
        Bflops = (
            compiled_decod_fn.cost_analysis()[0]["flops"]
            / 1e9
            / self.decode_compiled_total
        )
        log_for_0("Compiling VAE decoder done in %.2f seconds." % (time.time() - now))
        log_for_0(f"FLOPs (1e9): {Bflops}")

        def call_p_compiled_model_fn(x, p_func, var):
            if x.shape[0] > self.decode_compiled_total:
                raise ValueError(
                    "VAE decode chunk larger than compiled size: "
                    f"{x.shape[0]} > {self.decode_compiled_total}."
                )
            if x.shape[0] < self.decode_compiled_total:
                pad_shape = (self.decode_compiled_total - x.shape[0],) + x.shape[1:]
                x = jnp.concatenate(
                    [x, jnp.zeros(pad_shape, dtype=x.dtype)],
                    axis=0,
                )
            x = dist_prepare_batch_data(dict(x=x))["x"]
            x = p_func(var, x)
            x = x.sample
            x = x.reshape((-1,) + x.shape[2:])
            return dict(sample=x)

        call_compiled_decod_func = partial(
            call_p_compiled_model_fn, p_func=compiled_decod_fn, var=p_vae_variable
        )

        return call_compiled_decod_func

    def _decode_raw_latents(self, latents):
        num_latents = latents.shape[0]
        decoded_chunks = []

        for start in range(0, num_latents, self.decode_compiled_total):
            chunk = latents[start : start + self.decode_compiled_total]
            chunk_size = chunk.shape[0]
            decoded = self.decode_fn(chunk)["sample"][:chunk_size]
            decoded_chunks.append(decoded)

        if len(decoded_chunks) == 1:
            return decoded_chunks[0]
        return jnp.concatenate(decoded_chunks, axis=0)

    def cached_encode(self, cached_value, rng):
        latent = LatentDist(cached_value).sample(key=rng).transpose((0, 3, 1, 2))
        latent = (latent - self.mean) / self.std
        return latent.transpose((0, 2, 3, 1))

    def decode(self, latents):
        latents = latents * self.std + self.mean
        return self._decode_raw_latents(latents)


class DiTLatentManager(LatentManager):
    """VAE decode helper for original DiT-scaled latents."""

    def __init__(
        self,
        vae_type,
        decode_batch_size,
        input_size,
        latent_scale=0.18215,
        decode_num_local_devices=None,
    ):
        self.latent_scale = latent_scale
        super().__init__(
            vae_type,
            decode_batch_size,
            input_size,
            decode_num_local_devices=decode_num_local_devices,
        )

    def decode(self, latents):
        return self._decode_raw_latents(latents / self.latent_scale)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, root, use_flip=False):
        self.root = root
        self.file_list = [
            file for file in os.listdir(self.root) if file.endswith(".pt")
        ]
        self.use_flip = use_flip

    def __len__(self):
        return len(self.file_list)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body.append(f"Use flip: {self.use_flip}")
        lines = [head] + [" " * 4 + line for line in body]
        return "\n".join(lines)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root, self.file_list[idx])
        data = torch.load(file_path)
        return data["image"], data["label"]
