"""Perceptual auxiliary losses for pMF training."""

from typing import Tuple

import jax
import jax.numpy as jnp

from models.convnext import load_convnext_jax_model
from utils.logging_util import log_for_0


def _resample_crop_to_square(
    img: jnp.ndarray,
    top: jnp.ndarray,
    left: jnp.ndarray,
    crop_h: jnp.ndarray,
    crop_w: jnp.ndarray,
    out_size: int,
) -> jnp.ndarray:
    out = jnp.asarray(out_size, jnp.float32)
    crop_h_f = jnp.asarray(crop_h, jnp.float32)
    crop_w_f = jnp.asarray(crop_w, jnp.float32)
    top_f = jnp.asarray(top, jnp.float32)
    left_f = jnp.asarray(left, jnp.float32)

    sy = out / crop_h_f
    sx = out / crop_w_f
    ty = -top_f * sy
    tx = -left_f * sx

    return jax.image.scale_and_translate(
        img,
        shape=(out_size, out_size, img.shape[-1]),
        spatial_dims=(0, 1),
        scale=jnp.stack([sy, sx]),
        translation=jnp.stack([ty, tx]),
        method="cubic",
        antialias=True,
    )


def paired_random_resized_crop(
    rng: jax.Array,
    x1: jnp.ndarray,
    x2: jnp.ndarray,
    out_size: int = 224,
    scale: Tuple[float, float] = (0.08, 1.0),
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
):
    assert x1.shape == x2.shape
    batch_size, height, width, _ = x1.shape
    keys = jax.random.split(rng, batch_size)

    def sample_params(key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        area = jnp.asarray(height * width, jnp.float32)
        target_area = area * jax.random.uniform(
            k1, (), minval=scale[0], maxval=scale[1]
        )
        log_ratio = jnp.log(jnp.asarray(ratio, jnp.float32))
        aspect = jnp.exp(jax.random.uniform(k2, (), minval=log_ratio[0], maxval=log_ratio[1]))
        crop_w = jnp.clip(
            jnp.round(jnp.sqrt(target_area * aspect)).astype(jnp.int32),
            1,
            width,
        )
        crop_h = jnp.clip(
            jnp.round(jnp.sqrt(target_area / aspect)).astype(jnp.int32),
            1,
            height,
        )
        top = jax.random.randint(k3, (), 0, height - crop_h + 1)
        left = jax.random.randint(k4, (), 0, width - crop_w + 1)
        return top, left, crop_h, crop_w

    tops, lefts, crop_hs, crop_ws = jax.vmap(sample_params)(keys)
    crop_fn = lambda img, t, l, h, w: _resample_crop_to_square(
        img, t, l, h, w, out_size
    )
    return (
        jax.vmap(crop_fn)(x1, tops, lefts, crop_hs, crop_ws),
        jax.vmap(crop_fn)(x2, tops, lefts, crop_hs, crop_ws),
    )


def _param_count(params):
    return sum(
        int(jnp.prod(jnp.array(p.shape))) for p in jax.tree_util.tree_leaves(params)
    )


def _load_lpips_model():
    try:
        from lpips_j.lpips import LPIPS
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pMF LPIPS aux loss requires the 'lpips_j' package. "
            "Install it in the active venv before running with "
            "--config.model.lpips=True."
        ) from exc

    lpips_model = LPIPS()
    dummy_input = jnp.zeros((1, 224, 224, 3))
    lpips_params = lpips_model.init(jax.random.PRNGKey(0), dummy_input, dummy_input)
    return lpips_model, lpips_params


def init_auxloss(config):
    lpips_model = lpips_params = None
    convnext_model = convnext_params = None

    if config.model.lpips:
        log_for_0("Loading LPIPS perceptual model...")
        lpips_model, lpips_params = _load_lpips_model()
        log_for_0("LPIPS model loaded with param count %d.", _param_count(lpips_params))

    if config.model.convnext:
        log_for_0("Loading ConvNeXt perceptual model...")
        convnext_model, convnext_params = load_convnext_jax_model()
        log_for_0(
            "ConvNeXt perceptual model loaded with param count %d.",
            _param_count(convnext_params),
        )

    def auxloss_fn(model_images, gt_images, rng=None):
        batch_size = model_images.shape[0]
        if rng is None:
            rng = jax.random.PRNGKey(0)
        model_images, gt_images = paired_random_resized_crop(
            rng, model_images, gt_images, out_size=224
        )

        if config.model.lpips:
            lpips_dist = lpips_model.apply(lpips_params, model_images, gt_images).reshape(-1)
        else:
            lpips_dist = jnp.zeros((batch_size,), dtype=jnp.float32)

        if config.model.convnext:
            convnext_model_images = convnext_model.apply(convnext_params, model_images)
            convnext_gt_images = convnext_model.apply(convnext_params, gt_images)
            class_dist = jnp.sum((convnext_model_images - convnext_gt_images) ** 2, axis=-1)
        else:
            class_dist = jnp.zeros((batch_size,), dtype=jnp.float32)

        return lpips_dist, class_dist

    log_for_0("Auxiliary perceptual loss function initialized.")
    return auxloss_fn
