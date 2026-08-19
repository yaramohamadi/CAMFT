"""Dedicated entrypoint for JAX JiT training."""

import os
import warnings

import jax
from absl import app, flags
from ml_collections import config_flags

import train_jit
from utils import logging_util
from utils.logging_util import log_for_0


def maybe_initialize_jax_distributed():
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coordinator_address:
        jax.distributed.initialize(coordinator_address=coordinator_address)


logging_util.supress_checkpt_info()
warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store model data.")
flags.DEFINE_bool("debug", False, "Debugging mode.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    maybe_initialize_jax_distributed()
    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())
    log_for_0("FLAGS.config: \n%s", FLAGS.config)

    try:
        if FLAGS.config.eval_only:
            train_jit.just_evaluate(FLAGS.config, FLAGS.workdir)
        else:
            train_jit.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    finally:
        logging_util.close_wandb()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
