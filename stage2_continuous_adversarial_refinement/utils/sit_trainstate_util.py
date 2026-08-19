"""Train-state helpers for the dedicated plain SiT training path."""

from copy import deepcopy
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import struct
from flax.training import train_state
from jax import random

from utils.ema_util import update_ema
from utils.logging_util import log_for_0
from utils.state_util import print_params


def initialized(key, image_size, model, model_label="model"):
    input_shape = (1, image_size, image_size, 4)
    x = jnp.ones(input_shape, dtype=jnp.float32)
    t = jnp.ones((1,), dtype=jnp.float32)
    y = jnp.ones((1,), dtype=jnp.int32)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing %s params...", model_label)
    variables = init({"params": key}, x, t, y)
    log_for_0("Initializing %s params done.", model_label)

    param_count = sum(x.size for x in jax.tree_leaves(variables["params"]))
    log_for_0("Total trainable parameters: %s", param_count)
    return variables, variables["params"]


class TrainState(train_state.TrainState):
    ema_params: Any
    grad_accum: Any
    grad_accum_step: Any


@struct.dataclass
class EvalState:
    step: Any
    params: Any
    ema_params: Any


def create_eval_state(
    rng,
    config: ml_collections.ConfigDict,
    model,
    image_size,
    model_label="model",
):
    del config
    rng, rng_init = random.split(rng)
    _, params = initialized(rng_init, image_size, model, model_label=model_label)
    return EvalState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        ema_params=None,
    )


def create_train_state(
    rng,
    config: ml_collections.ConfigDict,
    model,
    image_size,
    lr_fn,
    model_label="model",
):
    rng, rng_init = random.split(rng)

    _, params = initialized(rng_init, image_size, model, model_label=model_label)
    use_ema = config.training.get("use_ema", True)
    ema_params = deepcopy(params)
    if use_ema:
        ema_params = update_ema(ema_params, params, 0)

    grad_accum = None
    grad_accum_step = jnp.array(0, dtype=jnp.int32)
    if bool(config.training.get("print_model_params", False)):
        print_params(params["net"])

    tx = optax.adamw(
        learning_rate=lr_fn,
        weight_decay=0,
        b2=config.training.adam_b2,
    )
    state = TrainState.create(
        apply_fn=partial(model.apply, method=model.forward),
        params=params,
        ema_params=ema_params,
        grad_accum=grad_accum,
        grad_accum_step=grad_accum_step,
        tx=tx,
    )
    return state
