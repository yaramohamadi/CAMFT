"""Pure losses and sampling utilities for discrete adversarial flow post-training."""

import math
import jax
import jax.numpy as jnp


def sample_time_pairs(rng, batch_size, min_interval=0.05, p_r_zero=0.25):
    """Sample 0 <= r < t <= 1 with t-r >= min_interval."""
    min_interval = float(min_interval)
    p_r_zero = float(p_r_zero)
    if not 0.0 < min_interval <= 1.0:
        raise ValueError("min_interval must be in (0, 1].")
    if not 0.0 <= p_r_zero <= 1.0:
        raise ValueError("p_r_zero must be in [0, 1].")

    rng_t, rng_r, rng_zero = jax.random.split(rng, 3)
    t = min_interval + (1.0 - min_interval) * jax.random.uniform(
        rng_t, (batch_size,), dtype=jnp.float32
    )
    r_upper = jnp.maximum(t - min_interval, 0.0)
    sampled_r = jax.random.uniform(rng_r, (batch_size,), dtype=jnp.float32) * r_upper
    zero_mask = jax.random.uniform(rng_zero, (batch_size,)) < p_r_zero
    r = jnp.where(zero_mask, 0.0, sampled_r)
    interval = t - r
    return r, t, interval, zero_mask


def linear_path(x0, x1, time):
    time = time.reshape((time.shape[0],) + (1,) * (x0.ndim - 1))
    return (1.0 - time) * x0 + time * x1


def generated_lower_endpoint(x_t, u, r, t):
    interval = (t - r).reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
    x_r_fake = x_t - interval * u
    if x_r_fake.shape != x_t.shape:
        raise ValueError(
            f"Generated endpoint shape {x_r_fake.shape} != source shape {x_t.shape}."
        )
    return x_r_fake


def discriminator_adversarial_loss(d_real, d_fake):
    """Relativistic AFM discriminator loss: softplus(d_fake - d_real)."""
    return jnp.mean(jax.nn.softplus(d_fake - d_real))


def generator_adversarial_loss(d_real, d_fake):
    """Relativistic AFM generator loss: softplus(d_real - d_fake)."""
    return jnp.mean(jax.nn.softplus(d_real - d_fake))


def centering_penalty(d_real, d_fake):
    return jnp.mean((d_real + d_fake) ** 2)


def finite_difference_penalty(
    discriminator,
    discriminator_params,
    x,
    time,
    labels,
    interval,
    rng,
    *,
    fd_epsilon,
    batch_fraction,
):
    """Directional finite-difference approximation to interval-weighted R1/R2."""
    if fd_epsilon <= 0.0:
        raise ValueError("fd_epsilon must be positive.")
    if not 0.0 < batch_fraction <= 1.0:
        raise ValueError("batch_fraction must be in (0, 1].")
    count = max(1, int(math.ceil(x.shape[0] * batch_fraction)))
    x = x[:count]
    time = time[:count]
    labels = labels[:count]
    interval = interval[:count]
    perturbation = fd_epsilon * jax.random.normal(rng, x.shape, dtype=x.dtype)
    d_clean = discriminator.apply(
        {"params": discriminator_params}, x, time, labels
    )
    d_perturbed = discriminator.apply(
        {"params": discriminator_params}, x + perturbation, time, labels
    )
    squared_difference = (d_clean - d_perturbed) ** 2 / (fd_epsilon**2)
    return jnp.mean(interval * squared_difference)


def optimal_transport_loss(x_r_fake, x_t, interval, interval_eps=1e-3):
    """AFM transport cost normalized by latent dimension and interval length."""
    reduce_axes = tuple(range(1, x_t.ndim))
    data_dimension = 1
    for size in x_t.shape[1:]:
        data_dimension *= size
    squared_distance = jnp.sum((x_r_fake - x_t) ** 2, axis=reduce_axes)
    denominator = data_dimension * jnp.maximum(interval, interval_eps)
    return jnp.mean(squared_distance / denominator)


def discriminator_loss(
    d_real,
    d_fake,
    r1,
    r2,
    *,
    lambda_gp,
    lambda_cp,
):
    loss_adv = discriminator_adversarial_loss(d_real, d_fake)
    loss_cp = centering_penalty(d_real, d_fake)
    total = loss_adv + lambda_gp * (r1 + r2) + lambda_cp * loss_cp
    return total, {
        "loss": total,
        "loss_d": total,
        "loss_d_adv": loss_adv,
        "loss_r1": r1,
        "loss_r2": r2,
        "loss_cp": loss_cp,
        "d_real": jnp.mean(d_real),
        "d_fake": jnp.mean(d_fake),
        "d_margin": jnp.mean(d_real - d_fake),
    }


def generator_loss(
    d_real,
    d_fake,
    x_r_fake,
    x_t,
    interval,
    *,
    lambda_adv,
    lambda_ot,
    lambda_imf=0.0,
    loss_imf=0.0,
    lambda_anchor=0.0,
    loss_anchor=0.0,
    interval_eps=1e-3,
):
    loss_adv = generator_adversarial_loss(d_real, d_fake)
    loss_ot = optimal_transport_loss(x_r_fake, x_t, interval, interval_eps)
    total = (
        lambda_adv * loss_adv
        + lambda_ot * loss_ot
        + lambda_imf * loss_imf
        + lambda_anchor * loss_anchor
    )
    return total, {
        "loss": total,
        "loss_g": total,
        "loss_g_adv": loss_adv,
        "loss_ot": loss_ot,
        "loss_imf": jnp.asarray(loss_imf),
        "loss_anchor": jnp.asarray(loss_anchor),
        "d_real": jnp.mean(d_real),
        "d_fake": jnp.mean(d_fake),
        "d_margin": jnp.mean(d_real - d_fake),
    }


def is_discriminator_step(batch_step, warmup_steps, d_steps_per_g_step):
    """Warm up D, then repeat exactly N discriminator batches and one G batch."""
    after_warmup = jnp.maximum(batch_step - warmup_steps, 0)
    in_d_phase = jnp.mod(after_warmup, d_steps_per_g_step + 1) < d_steps_per_g_step
    return jnp.logical_or(batch_step < warmup_steps, in_d_phase)


def augment_latents(x, rng, probability):
    """Semantic-preserving horizontal flip augmentation in latent space."""
    if probability <= 0.0:
        return x
    flip_mask = jax.random.uniform(rng, (x.shape[0],)) < probability
    flipped = jnp.flip(x, axis=2)
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return jnp.where(flip_mask.reshape(mask_shape), flipped, x)


def decayed_anchor_weight(initial_weight, generator_step, decay_steps):
    if initial_weight <= 0.0:
        return jnp.asarray(0.0, jnp.float32)
    if decay_steps <= 0:
        return jnp.asarray(initial_weight, jnp.float32)
    fraction = jnp.maximum(1.0 - generator_step / float(decay_steps), 0.0)
    return jnp.asarray(initial_weight, jnp.float32) * fraction


def cosine_decay_weight(initial_weight, final_weight, step, decay_steps):
    """Cosine-decay a scalar weight, clamping to the final value."""
    if decay_steps <= 0:
        return jnp.asarray(final_weight, jnp.float32)
    progress = jnp.clip(
        jnp.asarray(step, jnp.float32) / float(decay_steps), 0.0, 1.0
    )
    cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.asarray(final_weight, jnp.float32) + (
        jnp.asarray(initial_weight, jnp.float32) - final_weight
    ) * cosine
