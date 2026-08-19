"""Local Muon optimizer fallback for Optax versions without optax.contrib.muon."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from optax._src import base, numerics, utils


_DEFAULT_NS_COEFFS = (3.4445, -4.7750, 2.0315)


class MuonState(NamedTuple):
    count: jax.Array
    mu: base.Updates
    ns_coeffs: jax.Array


def _bias_correction(moment, beta, count):
    bias_correction = 1.0 - beta**count
    return jax.tree_util.tree_map(lambda x: x / bias_correction.astype(x.dtype), moment)


def _orthogonalize_matrix(x, ns_coeffs, ns_steps, eps):
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    x = x / (jnp.linalg.norm(x, ord="fro") + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)

    def step_fn(_, x_inner):
        a = x_inner @ x_inner.T.conj()
        b = ns_coeffs[1] * a + ns_coeffs[2] * (a @ a)
        return ns_coeffs[0] * x_inner + b @ x_inner

    x = jax.lax.fori_loop(0, ns_steps, step_fn, x)
    if transposed:
        x = x.T
    return x


def _scale_by_shape(update):
    fan_in, fan_out = update.shape
    return jnp.sqrt(jnp.maximum(1.0, fan_out / fan_in)) * update


def scale_by_muon(
    ns_coeffs=_DEFAULT_NS_COEFFS,
    ns_steps=5,
    beta=0.95,
    eps=1e-8,
    mu_dtype=None,
    *,
    nesterov=True,
):
    """Muon transform for 2D parameter leaves.

    This mirrors the Optax contrib behavior used by pMF for matrix parameters.
    Non-2D leaves are handled by the AdamW branch in ``muon`` below.
    """

    mu_dtype = utils.canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = jax.tree_util.tree_map(
            lambda p: jnp.zeros_like(p, dtype=mu_dtype or p.dtype),
            params,
        )
        return MuonState(
            count=jnp.zeros([], jnp.int32),
            mu=mu,
            ns_coeffs=jnp.asarray(ns_coeffs),
        )

    def update_fn(updates, state, params=None):
        del params
        mu = jax.tree_util.tree_map(
            lambda g, m: (1.0 - beta) * g + beta * m,
            updates,
            state.mu,
        )
        count_inc = numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree_util.tree_map(
                lambda m, g: beta * m + (1.0 - beta) * g,
                _bias_correction(mu, beta, numerics.safe_increment(count_inc)),
                _bias_correction(updates, beta, count_inc),
            )
        else:
            mu_hat = _bias_correction(mu, beta, count_inc)

        updates = jax.tree_util.tree_map(
            lambda x: _scale_by_shape(
                _orthogonalize_matrix(x, state.ns_coeffs, ns_steps, eps)
            ),
            mu_hat,
        )
        mu = jax.tree_util.tree_map(
            lambda x: x.astype(mu_dtype) if mu_dtype is not None else x,
            mu,
        )
        return updates, MuonState(count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs)

    return base.GradientTransformation(init_fn, update_fn)


def muon(
    learning_rate,
    *,
    ns_coeffs=_DEFAULT_NS_COEFFS,
    ns_steps=5,
    beta=0.95,
    eps=1e-8,
    weight_decay=0.0,
    mu_dtype=None,
    nesterov=True,
    adam_b1=0.9,
    adam_b2=0.999,
    adam_weight_decay=0.0,
    adam_learning_rate=None,
):
    """Muon optimizer with AdamW fallback for non-matrix parameters."""

    if adam_learning_rate is None:
        adam_learning_rate = learning_rate

    def param_labels(params):
        return jax.tree_util.tree_map(
            lambda p: "muon" if getattr(p, "ndim", 0) == 2 else "adam",
            params,
        )

    return optax.multi_transform(
        {
            "muon": optax.chain(
                scale_by_muon(
                    ns_coeffs=ns_coeffs,
                    ns_steps=ns_steps,
                    beta=beta,
                    eps=eps,
                    mu_dtype=mu_dtype,
                    nesterov=nesterov,
                ),
                optax.add_decayed_weights(weight_decay),
                optax.scale_by_learning_rate(learning_rate),
            ),
            "adam": optax.adamw(
                learning_rate=adam_learning_rate,
                b1=adam_b1,
                b2=adam_b2,
                eps=eps,
                weight_decay=adam_weight_decay,
                mu_dtype=mu_dtype,
                nesterov=nesterov,
            ),
        },
        param_labels,
    )
