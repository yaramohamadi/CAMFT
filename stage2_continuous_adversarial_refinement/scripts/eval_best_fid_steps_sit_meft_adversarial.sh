#!/usr/bin/env bash
# Final 1/2-NFE evaluation for an AFM or CA-iMF SiT-MeFT run.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: CONFIG_MODE=caltech_sit_meft_afm_posttrain $0 <run_dir> [steps...] [-- config overrides...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ROOT="$1"
shift
STEPS=()
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  if [[ "$1" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
    break
  fi
  STEPS+=("$1")
  shift
done
[[ ${#STEPS[@]} -gt 0 ]] || STEPS=(1 2)

CONFIG_MODE="${CONFIG_MODE:-caltech_sit_meft_afm_posttrain}"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
USE_WANDB="${USE_WANDB:-False}"
BEST_FID_DIR="$RUN_ROOT/best_fid"
FINAL_CSV="$RUN_ROOT/final_eval_metrics.csv"

[[ -d "$BEST_FID_DIR" ]] || { echo "Missing best_fid directory: $BEST_FID_DIR" >&2; exit 3; }
mapfile -t CKPTS < <(find "$BEST_FID_DIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' | sort)
[[ ${#CKPTS[@]} -eq 1 ]] || { echo "Expected exactly one best_fid checkpoint under $BEST_FID_DIR" >&2; exit 4; }
CHECKPOINT_DIR="${CKPTS[0]}"
rm -f "$FINAL_CSV"

for NUM_STEPS in "${STEPS[@]}"; do
  [[ "$NUM_STEPS" =~ ^[0-9]+$ ]] || { echo "Invalid sampling step: $NUM_STEPS" >&2; exit 5; }
  EVAL_DIR="$RUN_ROOT/eval_best_fid_${NUM_STEPS}steps"
  mkdir -p "$EVAL_DIR"
  TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}" \
  XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}" \
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}" \
  "$PYTHON" "$REPO_ROOT/main.py" \
    --workdir="$EVAL_DIR" \
    --config="$REPO_ROOT/configs/load_config.py:$CONFIG_MODE" \
    "${EXTRA_ARGS[@]}" \
    --config.eval_only=True \
    --config.partial_load=False \
    --config.load_from="$CHECKPOINT_DIR" \
    --config.sampling.num_steps="$NUM_STEPS" \
    --config.training.force_metric_num_steps="$NUM_STEPS" \
    --config.logging.use_wandb="$USE_WANDB" \
    2>&1 | tee -a "$EVAL_DIR/output.log"

  if [[ -f "$EVAL_DIR/eval_metrics.csv" ]]; then
    if [[ ! -f "$FINAL_CSV" ]]; then
      cp "$EVAL_DIR/eval_metrics.csv" "$FINAL_CSV"
    else
      tail -n +2 "$EVAL_DIR/eval_metrics.csv" >> "$FINAL_CSV"
    fi
  fi

  # Promote the final-eval grid into the run-level images directory, matching
  # the convention used by the ordinary MeFT runs.
  mkdir -p "$RUN_ROOT/images"
  image_source="$(find "$EVAL_DIR/images" -maxdepth 1 -type f -name "*image_grid_steps_${NUM_STEPS}.png" -print -quit 2>/dev/null || true)"
  if [[ -z "$image_source" ]]; then
    image_source="$(find "$EVAL_DIR/images" -maxdepth 1 -type f -name '*.png' -print -quit 2>/dev/null || true)"
  fi
  if [[ -n "$image_source" ]]; then
    cp "$image_source" "$RUN_ROOT/images/${NUM_STEPS}_image_grid.png"
  fi
done

echo "Final evaluation CSV: $FINAL_CSV"
