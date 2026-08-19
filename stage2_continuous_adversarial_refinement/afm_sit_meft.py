"""Forward-time endpoint utilities for AFM on SiT-DMF MeFT checkpoints."""

import jax
import jax.numpy as jnp


def sample_time_pairs(rng, batch_size, min_interval=0.05, p_r_one=0.25):
    """Sample 0 <= t < r <= 1 with r-t >= min_interval."""
    min_interval = float(min_interval)
    p_r_one = float(p_r_one)
    if not 0.0 < min_interval <= 1.0:
        raise ValueError("min_interval must be in (0, 1].")
    if not 0.0 <= p_r_one <= 1.0:
        raise ValueError("p_r_one must be in [0, 1].")
    rng_t, rng_r, rng_one = jax.random.split(rng, 3)
    t = (1.0 - min_interval) * jax.random.uniform(
        rng_t, (batch_size,), dtype=jnp.float32
    )
    r_lower = t + min_interval
    sampled_r = r_lower + (1.0 - r_lower) * jax.random.uniform(
        rng_r, (batch_size,), dtype=jnp.float32
    )
    one_mask = jax.random.uniform(rng_one, (batch_size,)) < p_r_one
    r = jnp.where(one_mask, 1.0, sampled_r)
    return t, r, r - t, one_mask


def sit_linear_path(data, noise, time):
    """SiT flow path: noise at time zero and data at time one."""
    time = time.reshape((time.shape[0],) + (1,) * (data.ndim - 1))
    return (1.0 - time) * noise + time * data


def generated_future_endpoint(x_t, average_velocity, t, r):
    """Euler-integrate the learned average velocity from t forward to r."""
    interval = (r - t).reshape((t.shape[0],) + (1,) * (x_t.ndim - 1))
    result = x_t + interval * average_velocity
    if result.shape != x_t.shape:
        raise ValueError(f"Endpoint shape {result.shape} != input {x_t.shape}.")
    return result
