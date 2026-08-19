"""Finite-interval CA-iMF logits for the forward-time SiT-MeFT convention."""

import jax.numpy as jnp


def finite_interval_logits(discriminator, discriminator_params, samples):
    """Compute real/fake potential quotients over 0 <= t < r <= 1."""
    t, r = samples["t"], samples["r"]
    labels = samples["labels"]
    interval = r - t
    d_t = discriminator.apply(
        {"params": discriminator_params},
        samples["x_t"],
        t,
        t,
        r,
        labels,
    )
    d_r = discriminator.apply(
        {"params": discriminator_params},
        samples["x_r"],
        r,
        t,
        r,
        labels,
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params},
        samples["xhat_r"],
        r,
        t,
        r,
        labels,
    )
    real_logit = (d_r - d_t) / interval
    fake_logit = (d_fake - d_t) / interval
    return real_logit, fake_logit, (d_t, d_r, d_fake)


def finite_fake_logit(discriminator, discriminator_params, samples):
    """Compute only the generator-facing fake quotient."""
    t, r = samples["t"], samples["r"]
    labels = samples["labels"]
    d_t = discriminator.apply(
        {"params": discriminator_params},
        samples["x_t"],
        t,
        t,
        r,
        labels,
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params},
        samples["xhat_r"],
        r,
        t,
        r,
        labels,
    )
    return (d_fake - d_t) / jnp.maximum(r - t, 1e-6)
