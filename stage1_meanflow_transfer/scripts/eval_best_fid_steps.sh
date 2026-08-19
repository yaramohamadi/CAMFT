#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  cat <<'EOF'
Usage: CONFIG_MODE=caltech_eval bash scripts/eval_best_fid_steps.sh <best_fid_dir_or_run_dir> [num_steps...]

Example:
  CONFIG_MODE=caltech_eval bash scripts/eval_best_fid_steps.sh \
    files/logs/finetuning/caltech_dogfit_20260327_010546_kh9yn5 2 4

If the first argument points to the run directory, the script assumes the checkpoint is under <run_dir>/best_fid.
If the first argument points to the best_fid directory itself, it uses that directly.
Set EVAL_PLATFORM=cpu to force CPU evaluation when GPU checkpoint restore runs out of memory.
EOF
  exit 1
fi

BEST_FID_ROOT="$1"
shift
if [[ $# -gt 0 ]]; then
  STEPS=("$@")
else
  STEPS=(2 4)
fi
CONFIG_MODE="${CONFIG_MODE:-caltech_eval}"
DEFAULT_PYTHON="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="python3"
fi
PYTHON="${PYTHON:-${DEFAULT_PYTHON}}"
USE_WANDB="${USE_WANDB:-True}"
MODEL_STR="${MODEL_STR:-}"
MODEL_USE_DOGFIT="${MODEL_USE_DOGFIT:-}"
TARGET_USE_NULL_CLASS="${TARGET_USE_NULL_CLASS:-}"
CLASS_DROPOUT_PROB="${CLASS_DROPOUT_PROB:-}"
TARGET_OUTPUT_PREDICTION_SPACE="${TARGET_OUTPUT_PREDICTION_SPACE:-}"
TARGET_VELOCITY_MAP_MODE="${TARGET_VELOCITY_MAP_MODE:-}"
USE_EMA_VC="${USE_EMA_VC:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
DATASET_NUM_CLASSES="${DATASET_NUM_CLASSES:-}"
FID_CACHE_REF="${FID_CACHE_REF:-}"
FD_DINO_CACHE_REF="${FD_DINO_CACHE_REF:-}"
EVAL_PLATFORM="${EVAL_PLATFORM:-gpu}"
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-}"

case "$EVAL_PLATFORM" in
  gpu|cpu)
    ;;
  *)
    echo "ERROR: EVAL_PLATFORM must be 'gpu' or 'cpu', got: $EVAL_PLATFORM" >&2
    exit 6
    ;;
esac

if [[ -d "$BEST_FID_ROOT" && $(basename "$BEST_FID_ROOT") != "best_fid" ]]; then
  BEST_FID_DIR="$BEST_FID_ROOT/best_fid"
else
  BEST_FID_DIR="$BEST_FID_ROOT"
fi

if [[ ! -d "$BEST_FID_DIR" ]]; then
  echo "ERROR: best_fid directory not found: $BEST_FID_DIR" >&2
  exit 2
fi

mapfile -t CKPTS < <(find "$BEST_FID_DIR" -maxdepth 1 -type d -name 'checkpoint_*' | sort)
if [[ ${#CKPTS[@]} -eq 0 ]]; then
  echo "ERROR: no checkpoint_*/ directory found under $BEST_FID_DIR" >&2
  exit 3
fi
if [[ ${#CKPTS[@]} -gt 1 ]]; then
  echo "ERROR: more than one checkpoint_*/ directory found under $BEST_FID_DIR:" >&2
  printf '  %s\n' "${CKPTS[@]}" >&2
  exit 4
fi

CHECKPOINT_DIR="${CKPTS[0]}"
echo "Found checkpoint: $CHECKPOINT_DIR"

for NUM_STEPS in "${STEPS[@]}"; do
  if ! [[ "$NUM_STEPS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid num_steps value: $NUM_STEPS" >&2
    exit 5
  fi

  EVAL_WORKDIR="$(dirname "$BEST_FID_DIR")/eval_best_fid_${NUM_STEPS}steps"
  mkdir -p "$EVAL_WORKDIR"

  echo "=== Evaluating checkpoint with num_steps=$NUM_STEPS ==="
  echo "workdir: $EVAL_WORKDIR"
  echo "config mode: $CONFIG_MODE"
  echo "platform: $EVAL_PLATFORM"

  EXTRA_CONFIG_ARGS=()
  if [[ -n "$MODEL_STR" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.model_str=${MODEL_STR}")
  fi
  if [[ -n "$MODEL_USE_DOGFIT" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.use_dogfit=${MODEL_USE_DOGFIT}")
  fi
  if [[ -n "$TARGET_USE_NULL_CLASS" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.target_use_null_class=${TARGET_USE_NULL_CLASS}")
  fi
  if [[ -n "$CLASS_DROPOUT_PROB" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.class_dropout_prob=${CLASS_DROPOUT_PROB}")
  fi
  # Prediction-space / velocity-map / ema-vc must match TRAINING exactly, else the
  # model output is decoded through the wrong convention and FID collapses to noise.
  if [[ -n "$TARGET_OUTPUT_PREDICTION_SPACE" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.target_output_prediction_space=${TARGET_OUTPUT_PREDICTION_SPACE}")
  fi
  if [[ -n "$TARGET_VELOCITY_MAP_MODE" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.target_velocity_map_mode=${TARGET_VELOCITY_MAP_MODE}")
  fi
  if [[ -n "$USE_EMA_VC" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.model.use_ema_vc=${USE_EMA_VC}")
  fi
  if [[ -n "$DATASET_ROOT" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.dataset.root=${DATASET_ROOT}")
  fi
  if [[ -n "$DATASET_NUM_CLASSES" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.dataset.num_classes=${DATASET_NUM_CLASSES}")
    EXTRA_CONFIG_ARGS+=("--config.model.num_classes=${DATASET_NUM_CLASSES}")
    EXTRA_CONFIG_ARGS+=("--config.sampling.num_classes=${DATASET_NUM_CLASSES}")
  fi
  if [[ -n "$FID_CACHE_REF" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.fid.cache_ref=${FID_CACHE_REF}")
  fi
  if [[ -n "$FD_DINO_CACHE_REF" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.fd_dino.cache_ref=${FD_DINO_CACHE_REF}")
  fi
  if [[ -n "$FID_NUM_SAMPLES" ]]; then
    EXTRA_CONFIG_ARGS+=("--config.fid.num_samples=${FID_NUM_SAMPLES}")
  fi

  if [[ "$EVAL_PLATFORM" == "cpu" ]]; then
    JAX_PLATFORM_NAME=cpu \
      JAX_NUM_DEVICES=1 \
      TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
      PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
      $PYTHON \
        "${REPO_ROOT}/main.py" \
        --workdir="$EVAL_WORKDIR" \
        --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
        --config.eval_only=True \
        --config.training.use_ema=False \
        --config.load_from="$CHECKPOINT_DIR" \
        --config.sampling.num_steps=${NUM_STEPS} \
        --config.logging.use_wandb=${USE_WANDB} \
        "${EXTRA_CONFIG_ARGS[@]}" \
        2>&1 | tee -a "$EVAL_WORKDIR/output.log"
  else
    TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3} \
      XLA_FLAGS=${XLA_FLAGS:---xla_gpu_strict_conv_algorithm_picker=false} \
      XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false} \
      PYTHONWARNINGS=${PYTHONWARNINGS:-ignore} \
      $PYTHON \
        "${REPO_ROOT}/main.py" \
        --workdir="$EVAL_WORKDIR" \
        --config="${REPO_ROOT}/configs/load_config.py:${CONFIG_MODE}" \
        --config.eval_only=True \
        --config.training.use_ema=False \
        --config.load_from="$CHECKPOINT_DIR" \
        --config.sampling.num_steps=${NUM_STEPS} \
        --config.logging.use_wandb=${USE_WANDB} \
        "${EXTRA_CONFIG_ARGS[@]}" \
        2>&1 | tee -a "$EVAL_WORKDIR/output.log"
  fi

  echo "=== finished num_steps=$NUM_STEPS ==="
  echo
done
