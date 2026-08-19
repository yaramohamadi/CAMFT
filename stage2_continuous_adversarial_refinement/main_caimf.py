"""Entry point for continuous adversarial improved MeanFlow post-training."""

import os

import jax


def maybe_initialize_jax_distributed():
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if coordinator_address:
        jax.distributed.initialize(coordinator_address=coordinator_address)


from absl import app, flags
from ml_collections import config_flags

import train_caimf
from utils import logging_util
from utils.logging_util import log_for_0

logging_util.supress_checkpt_info()

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "Directory to store CA-iMF checkpoints.")
config_flags.DEFINE_config_file(
    "config",
    None,
    "CA-iMF training configuration.",
    lock_config=True,
)


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")
    maybe_initialize_jax_distributed()
    log_for_0("JAX process: %d / %d", jax.process_index(), jax.process_count())
    log_for_0("JAX local devices: %r", jax.local_devices())
    try:
        train_caimf.train_and_evaluate(FLAGS.config, FLAGS.workdir)
    finally:
        logging_util.close_wandb()


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
