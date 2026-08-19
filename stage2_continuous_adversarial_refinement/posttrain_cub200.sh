#!/usr/bin/env bash
#
# Stage 2: Continuous Adversarial MeanFlow (CAMF) refinement on CUB-200-2011.
#
# Takes a Stage 1 MeanFlow-Transfer student and refines it with pure adversarial
# post-training (lambda_imf = 0). The generator keeps the guidance behaviour it
# learned in Stage 1; the discriminator scores the guided few-step endpoint. Only
# the generator parameters are restored from the Stage 1 checkpoint, so pass the
# best-FID checkpoint directory from the matching Stage 1 run as <mf_t_checkpoint>.
#
# Usage:
#   bash posttrain_cub200.sh <family> <mf_t_checkpoint> [extra args]
#   family = imf | sit | dit | jit
#   mf_t_checkpoint = path to a Stage 1 best_fid/checkpoint_* directory
#
# The CUB latents and FID/FD-DINO reference statistics ship with this release
# under ../data and ../stats. CUB-200 has 200 classes.

set -euo pipefail

FAMILY="${1:-}"
LOAD_FROM="${2:-}"
if [[ -z "$FAMILY" || -z "$LOAD_FROM" ]]; then
  echo "Usage: bash posttrain_cub200.sh <imf|sit|dit|jit> <mf_t_checkpoint> [extra args]" >&2
  exit 1
fi
shift 2 || true
EXTRA_ARGS=("$@")

[[ -d "$LOAD_FROM" ]] || { echo "ERROR: checkpoint directory not found: $LOAD_FROM" >&2; exit 3; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_ROOT="$(cd "$HERE/.." && pwd)"

PYTHON="${PYTHON:-python3}"
USE_WANDB="${USE_WANDB:-False}"

FID_REF="$RELEASE_ROOT/stats/cub-200-2011_processed-fid_stats.npz"
FDD_REF="$RELEASE_ROOT/stats/cub-200-2011-fd_dino-vitb14_stats.npz"
NUM_CLASSES=200

# Latent families train on the precomputed VAE latents; JiT works in pixel space.
LATENT_ROOT="$RELEASE_ROOT/data/cub-200-2011_processed_latents"
PIXEL_ROOT="$RELEASE_ROOT/data/cub-200-2011_images"

# Per-family entry point, config, dataset root, and dataset name tag. DiT reuses
# the SiT MeanFlow post-training path because they share the imfDiT backbone
# class; only the config differs.
case "$FAMILY" in
  imf)
    ENTRY=main_caimf.py
    CONFIG_MODE=caltech_imf_caimf_posttrain
    DATA_ROOT="$LATENT_ROOT"
    DS_NAME=cub200_latent
    ;;
  sit)
    ENTRY=main_caimf_sit_meft.py
    CONFIG_MODE=caltech_sit_meft_caimf_posttrain
    DATA_ROOT="$LATENT_ROOT"
    DS_NAME=cub200_latent
    ;;
  dit)
    ENTRY=main_caimf_sit_meft.py
    CONFIG_MODE=caltech_dit_meft_caimf_posttrain
    DATA_ROOT="$LATENT_ROOT"
    DS_NAME=cub200_latent
    ;;
  jit)
    ENTRY=main_caimf_jit_meft.py
    CONFIG_MODE=caltech_jit_meft_caimf_posttrain
    DATA_ROOT="$PIXEL_ROOT"
    DS_NAME=cub200
    ;;
  *)
    echo "ERROR: unknown family '$FAMILY'. Use imf, sit, dit, or jit." >&2
    exit 2
    ;;
esac

STAMP="$(date '+%Y%m%d_%H%M%S')"
WORKDIR="$HERE/runs/cub200_${FAMILY}_camf_${STAMP}"
if [[ -d "$WORKDIR" ]] && find "$WORKDIR" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' -print -quit | grep -q .; then
  echo "ERROR: workdir already holds a checkpoint: $WORKDIR (use a fresh one)." >&2
  exit 4
fi
mkdir -p "$WORKDIR"

echo "CAMF CUB-200 post-training"
echo "  family     $FAMILY"
echo "  entry      $ENTRY"
echo "  config     $CONFIG_MODE"
echo "  stage-1 ckpt $LOAD_FROM"
echo "  data       $DATA_ROOT"
echo "  workdir    $WORKDIR"

cd "$HERE"
TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false \
  "$PYTHON" "$ENTRY" \
    --config=configs/load_config.py:"$CONFIG_MODE" \
    --config.logging.use_wandb="$USE_WANDB" \
    --config.load_from="$LOAD_FROM" \
    --config.dataset.name="$DS_NAME" \
    --config.dataset.root="$DATA_ROOT" \
    --config.dataset.num_classes="$NUM_CLASSES" \
    --config.model.num_classes="$NUM_CLASSES" \
    --config.sampling.num_classes="$NUM_CLASSES" \
    --config.fid.cache_ref="$FID_REF" \
    --config.fd_dino.cache_ref="$FDD_REF" \
    --workdir="$WORKDIR" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$WORKDIR/output.log"

echo "Post-training finished. Best-FID checkpoint is under $WORKDIR/best_fid."
echo "Run the final NFE 1 and 2 evaluation with the eval scripts under scripts/."
