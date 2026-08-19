"""CSV helpers for evaluation metrics."""

import csv
import os

import jax

from utils.logging_util import log_for_0


EVAL_METRIC_FIELDNAMES = (
    "eval_phase",
    "metric_mode",
    "training_step",
    "sampling_num_steps",
    "omega",
    "t_min",
    "t_max",
    "fid",
    "inception_score",
    "fd_dino",
    "is_best_fid",
    "is_best_fd_dino",
    "checkpoint_path",
    "sample_mode",
)


def append_eval_metrics_row(workdir, row, filename="eval_metrics.csv"):
    """Append one evaluation-metrics row under a run workdir."""
    if jax.process_index() != 0:
        return

    os.makedirs(workdir, exist_ok=True)
    csv_path = os.path.join(workdir, filename)
    file_exists = os.path.exists(csv_path)
    ordered_row = {field: row.get(field, "") for field in EVAL_METRIC_FIELDNAMES}

    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EVAL_METRIC_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(ordered_row)

    log_for_0("Appended evaluation metrics row to %s.", csv_path)
