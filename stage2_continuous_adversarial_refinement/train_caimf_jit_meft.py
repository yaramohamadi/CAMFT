"""Finite-interval CA-iMF adapter for JiT-DMF MeFT checkpoints.

JiT-DMF is PIXEL space (256x256x3), while the shared train_caimf.py is
latent-native (scans .pt latents, uses create_latent_split, decodes with a VAE
LatentManager). This shim monkeypatches the three latent touch points onto the
pixel helpers that already live in train_jit.py, so the shared trainer runs
unchanged for the latent DiT/SiT sweeps. Each run is its own process, so these
in-process rebindings never leak into the concurrent DiT/CP runs.
"""

import jax
import jax.numpy as jnp
from jax import random

import train_caimf as _base

from caimf_sit_meft import finite_fake_logit, finite_interval_logits
from train_jit import infer_num_classes_from_images
from train_jit_meft_common import create_models
from utils.input_pipeline import create_image_split


class _NCHWPixelManager:
    """Decode helper matching train_caimf's NCHW contract for pixel JiT.

    train_caimf._sample_step already transposes generate()'s BHWC pixels to
    NCHW (line `return images.transpose(0, 3, 1, 2)`) before decode is called,
    and both consumers (run_p_sample_step, preview) transpose the decode output
    NCHW->BHWC afterwards. So decode here is a clip-only identity in NCHW --
    NOT the BHWC->NCHW transpose that train_jit.PixelImageManager applies (that
    manager pairs with train_jit's own sample_step, which does not pre-transpose;
    reusing it here would double-transpose into a (3,256,256)-per-image panel).
    """

    def __init__(self, batch_size, decode_num_local_devices=None):
        self.batch_size = int(batch_size)
        self.decode_num_local_devices = (
            jax.local_device_count()
            if decode_num_local_devices is None
            else int(decode_num_local_devices)
        )

    def decode(self, images_nchw):
        return jnp.clip(images_nchw, -1.0, 1.0)


def _create_models(config):
    model, _, discriminator = create_models(
        config, retention_weight=float(config.caimf.lambda_imf)
    )
    return model, discriminator


def _pixel_get_images_and_labels(batch, rng_vae, distributed):
    """Pixel analogue of _base._get_images_and_labels (no VAE reparam).

    The latent path calls _cached_encode, which jnp.split(cached, 2, axis=-1)s
    an SD-VAE (mean,std) tensor. JiT's create_image_split yields 3-channel BHWC
    images already normalized to [-1, 1] (Normalize(0.5,0.5)), so the "encode"
    is the identity -- return the raw images. rng_vae is unused.
    """
    del rng_vae
    images = batch["image"] if distributed else batch["image"][0]
    labels = batch["label"] if distributed else batch["label"][0]
    return images, labels


def _pixel_set_num_classes_from_data(config):
    """Pixel analogue of _base._set_num_classes_from_data (counts class dirs)."""
    if not config.dataset.get("num_classes_from_data", False):
        return
    num_classes = infer_num_classes_from_images(config.dataset.root)
    config.dataset.num_classes = num_classes
    config.model.num_classes = num_classes
    config.sampling.num_classes = num_classes
    _base.log_for_0(
        "Inferred dataset.num_classes=%d from image class folders.", num_classes
    )


def _pixel_manager(vae, batch_size, image_size, decode_num_local_devices=None):
    """Drop-in for LatentManager(vae, bsz, img, ...) -> NCHW clip-only decode.

    train_caimf feeds NCHW pixels to decode (its _sample_step pre-transposes),
    so decode must return NCHW unchanged. The vae / image_size args are ignored.
    """
    return _NCHWPixelManager(
        batch_size, decode_num_local_devices=decode_num_local_devices
    )


def _pixel_create_state(config, model, discriminator, rng):
    """Pixel-channel analogue of _base._create_state.

    Identical to the base builder except every backbone-init dummy uses
    dataset.image_channels (3 for JiT pixels) instead of the hardcoded 4 latent
    channels. With 4, restore_partial_checkpoint silently drops the 3-channel
    x_embedder.proj1 kernel (shape mismatch), leaving a random 4-channel conv
    that then crashes at sample time against 3-channel noise (ScopeParamShapeError).
    """
    ca_cfg = config.caimf
    gen_lr = float(ca_cfg.gen_learning_rate)
    dis_lr = float(ca_cfg.dis_learning_rate)
    beta1 = float(ca_cfg.adam_beta1)
    beta2 = float(ca_cfg.adam_beta2)
    weight_decay = float(ca_cfg.weight_decay)
    image_channels = int(config.dataset.get("image_channels", 3))
    gen_tx = _base.optax.adamw(gen_lr, b1=beta1, b2=beta2, weight_decay=weight_decay)
    dis_tx = _base.optax.adamw(dis_lr, b1=beta1, b2=beta2, weight_decay=weight_decay)

    base_state = _base.create_train_state(
        rng,
        config,
        model,
        int(config.dataset.image_size),
        lambda _: jnp.asarray(gen_lr),
        input_channels=image_channels,
    )
    if not config.load_from:
        raise ValueError("load_from must point to an existing iMF checkpoint.")
    base_state = _base.restore_partial_checkpoint(
        base_state,
        config.load_from,
        prefer_ema=bool(ca_cfg.load_generator_ema),
        target_model_config=config.model,
    )
    params = base_state.params
    ema_params = _base._copy_device_tree(params)

    discriminator_updates = bool(ca_cfg.discriminator_updates)
    if discriminator_updates or float(ca_cfg.lambda_adv) > 0.0:
        rng, rng_dis = random.split(rng)
        batch_size = 1
        image_size = int(config.dataset.image_size)
        dummy_x = jnp.ones(
            (batch_size, image_size, image_size, image_channels), jnp.float32
        )
        dummy_time = jnp.full((batch_size,), 0.5, jnp.float32)
        dummy_r = jnp.full((batch_size,), 0.25, jnp.float32)
        dummy_y = jnp.zeros((batch_size,), jnp.int32)
        dis_params = discriminator.init(
            {"params": rng_dis},
            dummy_x,
            dummy_time,
            dummy_r,
            dummy_time,
            dummy_y,
        )["params"]
        dis_params, loaded = _base._copy_matching_generator_params(
            dis_params, params["net"]
        )
        _base.log_for_0(
            "Initialized discriminator from generator backbone: copied %d tensors; "
            "new scalar head remains freshly initialized.",
            loaded,
        )
        dis_opt_state = dis_tx.init(dis_params)
    else:
        dis_params = None
        dis_opt_state = None

    state = _base.CAIMFTrainState(
        step=jnp.asarray(0, jnp.int32),
        gen_step=jnp.asarray(0, jnp.int32),
        dis_step=jnp.asarray(0, jnp.int32),
        params=params,
        ema_params=ema_params,
        gen_opt_state=gen_tx.init(params),
        dis_params=dis_params,
        dis_opt_state=dis_opt_state,
    )
    return state, gen_tx, dis_tx


def train_and_evaluate(config, workdir):
    # Patch only these isolated entry points; existing main_caimf.py is unchanged.
    _base._create_models = _create_models
    _base.finite_interval_logits = finite_interval_logits
    _base.finite_fake_logit = finite_fake_logit
    # --- pixel-space data path (JiT has no VAE / no .pt latents) ---
    _base._set_num_classes_from_data = _pixel_set_num_classes_from_data
    _base.input_pipeline.create_latent_split = create_image_split
    _base.LatentManager = _pixel_manager
    # --- pixel-channel init (JiT proj1 is 3-channel, not the latent default 4) ---
    _base._create_state = _pixel_create_state
    # --- skip VAE (mean,std) reparam: JiT images are already pixels in [-1,1] ---
    _base._get_images_and_labels = _pixel_get_images_and_labels
    return _base.train_and_evaluate(config, workdir)
