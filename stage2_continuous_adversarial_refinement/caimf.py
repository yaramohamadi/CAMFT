"""Core losses and scheduling for continuous adversarial improved MeanFlow."""

import jax.numpy as jnp


def is_discriminator_step(batch_step, warmup_batches=10_000, discriminator_steps=16):
    """Return whether a post-training batch updates D instead of G.

    After the discriminator-only warmup, a cycle consists of N discriminator
    batches followed by one generator batch, matching the CAFM schedule.
    """
    batch_step = jnp.asarray(batch_step)
    after_warmup = jnp.maximum(batch_step - warmup_batches, 0)
    in_discriminator_part = (
        jnp.mod(after_warmup, discriminator_steps + 1) < discriminator_steps
    )
    return jnp.logical_or(batch_step < warmup_batches, in_discriminator_part)


def finite_interval_logits(
    discriminator,
    discriminator_params,
    samples,
):
    """Evaluate real/fake finite differences of the discriminator potential."""
    x_t = samples["x_t"]
    x_r = samples["x_r"]
    xhat_r = samples["xhat_r"]
    t = samples["t"]
    r = samples["r"]
    labels = samples["labels"]
    interval = t - r

    d_t = discriminator.apply(
        {"params": discriminator_params}, x_t, t, r, t, labels
    )
    d_r = discriminator.apply(
        {"params": discriminator_params}, x_r, r, r, t, labels
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params}, xhat_r, r, r, t, labels
    )
    real_logit = (d_t - d_r) / interval
    fake_logit = (d_t - d_fake) / interval
    return real_logit, fake_logit, (d_t, d_r, d_fake)


def finite_fake_logit(discriminator, discriminator_params, samples):
    """Evaluate only the fake finite difference needed by a generator update."""
    t = samples["t"]
    r = samples["r"]
    labels = samples["labels"]
    d_t = discriminator.apply(
        {"params": discriminator_params}, samples["x_t"], t, r, t, labels
    )
    d_fake = discriminator.apply(
        {"params": discriminator_params}, samples["xhat_r"], r, r, t, labels
    )
    return (d_t - d_fake) / (t - r)


# Centering-penalty (CP) variants for the discriminator ablation. Each maps the
# three finite-interval potentials (d_t = D(x_t), d_r = D(x_r), d_fake = D(xhat_r))
# to a scalar centering penalty. "full" is the default (our method).
#   full        : d_t^2 + d_r^2 + d_fake^2         (D_zt^2 + D_zr^2 + D_zhatr^2)
#   none        : 0                                 (no CP)
#   zt          : d_t^2                             (D_zt^2)
#   zt_zr       : d_t^2 + d_r^2                     (D_zt^2 + D_zr^2)
#   zt_zhatr    : d_t^2 + d_fake^2                  (D_zt^2 + D_zhatr^2)
#   afm_sum2    : (d_t + d_fake)^2                  ((D_zt + D_zhatr)^2, AFM-style)
#   full_sum2   : (d_t + d_r + d_fake)^2            ((D_zt + D_zr + D_zhatr)^2)
def compute_centering(d_t, d_r, d_fake, cp_mode="full"):
    """Return the centering-penalty scalar for the requested CP variant."""
    if cp_mode == "none":
        return jnp.zeros((), dtype=d_t.dtype)
    if cp_mode == "full":
        return jnp.mean(d_t**2 + d_r**2 + d_fake**2)
    if cp_mode == "zt":
        return jnp.mean(d_t**2)
    if cp_mode == "zt_zr":
        return jnp.mean(d_t**2 + d_r**2)
    if cp_mode == "zt_zhatr":
        return jnp.mean(d_t**2 + d_fake**2)
    if cp_mode == "afm_sum2":
        return jnp.mean((d_t + d_fake) ** 2)
    if cp_mode == "full_sum2":
        return jnp.mean((d_t + d_r + d_fake) ** 2)
    raise ValueError(f"Unknown cp_mode: {cp_mode!r}")


def discriminator_loss(
    real_logit, fake_logit, potentials, lambda_cp=1e-3, cp_mode="full"
):
    """Least-squares finite-interval discriminator loss with centering."""
    d_t, d_r, d_fake = potentials
    classification = jnp.mean((real_logit - 1.0) ** 2 + (fake_logit + 1.0) ** 2)
    centering = compute_centering(d_t, d_r, d_fake, cp_mode=cp_mode)
    total = classification + lambda_cp * centering
    return total, {
        "loss": total,
        "loss_d": total,
        "loss_d_classification": classification,
        "loss_d_centering": centering,
        "real_logit": jnp.mean(real_logit),
        "fake_logit": jnp.mean(fake_logit),
    }


def generator_adversarial_loss(fake_logit):
    """Non-saturating least-squares generator objective."""
    return jnp.mean((fake_logit - 1.0) ** 2)


def generator_loss(terms, fake_logit, lambda_imf, lambda_adv, lambda_ot):
    """Compose the requested iMF, adversarial, and transport-energy losses."""
    loss_adv = generator_adversarial_loss(fake_logit)
    loss_ot = jnp.mean(terms["u"] ** 2)
    total = (
        lambda_imf * terms["loss_imf"]
        + lambda_adv * loss_adv
        + lambda_ot * loss_ot
    )
    return total, {
        "loss": total,
        "loss_g": total,
        "loss_imf": terms["loss_imf"],
        "loss_u": terms["loss_u"],
        "loss_v": terms["loss_v"],
        "loss_adv": loss_adv,
        "loss_ot": loss_ot,
        "fake_logit": jnp.mean(fake_logit),
    }
