#!/usr/bin/env bash
#
# Stage 1: MeanFlow-Transfer (MF-T) on any of the five target domains.
#
# Adapts a pretrained ImageNet teacher into a few-step MeanFlow student. Pick a
# family (imf, sit, dit, jit) and a target dataset. The recipe for each family is
# exactly the one that produced the paper numbers: the entry point, config, base
# weights, and the few overrides that matter are all set below so a single
# command reproduces a run end to end.
#
# Usage:
#   bash train.sh <family> [dataset] [extra main.py args]
#   family  = imf | sit | dit | jit
#   dataset = artbench-10 | caltech-101 | cub-200-2011 | food-101 | stanford-cars
#             (defaults to cub-200-2011)
#
# Before running, download the base teacher weights (see WEIGHTS.md) and point
# WEIGHTS_DIR at the folder that holds them. The FID/FD-DINO reference
# statistics for all five datasets ship with this release under ../stats; the
# image and latent data does not (see the top-level README for how to build it).

set -euo pipefail

FAMILY="${1:-}"
if [[ -z "$FAMILY" ]]; then
  echo "Usage: bash train.sh <imf|sit|dit|jit> [dataset] [extra args]" >&2
  exit 1
fi
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$HERE/.." && pwd)"
# shellcheck source=../datasets.sh
source "$RELEASE_ROOT/datasets.sh"

# The dataset is optional and positional: treat the next argument as the dataset
# only when it does not look like a flag for the training script.
DATASET="cub-200-2011"
if [[ $# -gt 0 && "$1" != -* ]]; then
  DATASET="$1"
  shift
fi
EXTRA_ARGS=("$@")

ds_resolve "$DATASET" "$RELEASE_ROOT"

PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-False}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$RELEASE_ROOT/weights}"

# num_samples matches the paper (10k) for every metric. The class count is read
# straight from the data (num_classes_from_data is set in every config), so each
# dataset resolves to its own class count without an explicit override.
FID_NUM_SAMPLES="${FID_NUM_SAMPLES:-10000}"

# Per-family entry point, config mode, base weights, and sampling guidance. The
# DiT and SiT MeanFlow-Transfer runs share the direct-DMF-target recipe; the DiT
# paper run additionally clips gradients at norm 1.0. DogFit is off by default in
# all four configs. iMF and JiT keep their own plain fine-tuning configs.
#
# Latent families (imf/sit/dit) read the precomputed VAE latents; JiT works in
# pixel space and reads the raw images.
case "$FAMILY" in
  imf)
    ENTRY=main.py
    CONFIG_MODE=plain_imf_finetune
    BASE_WEIGHTS="$WEIGHTS_DIR/iMF-XL-2-full"
    OMEGA=7.5
    DATA_ROOT="$DS_LATENT_ROOT"
    EXTRA_ARGS+=(--config.sampling.t_min=0.4 --config.sampling.t_max=0.65)
    ;;
  sit)
    ENTRY=main.py
    CONFIG_MODE=caltech_sit_dmf_finetune
    BASE_WEIGHTS="$WEIGHTS_DIR/SiT-XL-2-256.pt"
    OMEGA=1.5
    DATA_ROOT="$DS_LATENT_ROOT"
    ;;
  dit)
    ENTRY=main.py
    CONFIG_MODE=caltech_dit_dmf_ddpmv
    BASE_WEIGHTS="$WEIGHTS_DIR/DiT-XL-2-256x256.pt"
    OMEGA=1.5
    DATA_ROOT="$DS_LATENT_ROOT"
    EXTRA_ARGS+=(--config.training.grad_clip_norm=1.0)
    ;;
  jit)
    ENTRY=main_imf_jit.py
    CONFIG_MODE=caltech_jit_dmf_meft
    BASE_WEIGHTS="$WEIGHTS_DIR/JiT-H-16-256.pth"
    OMEGA=2.2
    DATA_ROOT="$DS_PIXEL_ROOT"
    ;;
  *)
    echo "ERROR: unknown family '$FAMILY'. Use imf, sit, dit, or jit." >&2
    exit 2
    ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="$HERE/runs/${DS_SLUG}_${FAMILY}_${STAMP}"
mkdir -p "$WORKDIR"

echo "MeanFlow-Transfer run"
echo "  family     $FAMILY"
echo "  dataset    $DS_SLUG ($DS_NUM_CLASSES classes)"
echo "  entry      $ENTRY"
echo "  config     $CONFIG_MODE"
echo "  base       $BASE_WEIGHTS"
echo "  data       $DATA_ROOT"
echo "  fid ref    $DS_FID_REF"
echo "  fdd ref    $DS_FDD_REF"
echo "  workdir    $WORKDIR"

cd "$HERE"
TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" "$ENTRY" \
    --workdir="$WORKDIR" \
    --config=configs/load_config.py:"$CONFIG_MODE" \
    --config.logging.use_wandb="$USE_WANDB" \
    --config.load_from="$BASE_WEIGHTS" \
    --config.dataset.root="$DATA_ROOT" \
    --config.fid.cache_ref="$DS_FID_REF" \
    --config.fid.num_samples="$FID_NUM_SAMPLES" \
    --config.fd_dino.cache_ref="$DS_FDD_REF" \
    --config.sampling.omega="$OMEGA" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} \
    2>&1 | tee -a "$WORKDIR/output.log"
