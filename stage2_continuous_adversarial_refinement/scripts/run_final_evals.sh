#!/usr/bin/env bash
# Retroactive NFE 1&2 final-eval for early-stopped CAMF runs (DiT + JiT) that
# lack final_eval_metrics.csv. Each eval claims the shared GPU lock so the live
# caimf_jit_watcher can't double-claim. Detached screens; laptop can close.
set -uo pipefail
ADV=/opt/dlami/nvme/meanflow/imeanflow_adversarial
MAIN=/opt/dlami/nvme/meanflow/imeanflow
PY=$MAIN/.venv/bin/python
DATA=/opt/dlami/nvme/meanflow/datasets
cd "$ADV"
source "$ADV/scripts/gpu_lock.inc.sh"
EVAL_LATENT=scripts/eval_best_fid_steps_sit_meft_adversarial.sh   # DiT/SiT (latent, main.py+VAE)
EVAL_PIXEL=scripts/eval_best_fid_steps_plain_jit.sh              # JiT (pixel, main_jit.py, no VAE)

dit_case(){ case "$1" in
  caltech101)   LAT=caltech-101_processed_latents; NC=101; FID=caltech-101-fid_stats.npz; FDD=caltech-101-fd_dino-vitb14_stats.npz ;;
  cub200)       LAT=cub-200-2011_processed_latents; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
  stanfordcars) LAT=stanford-cars_processed_latents; NC=196; FID=stanford_cars_processed-fid_stats.npz; FDD=stanford-cars-fd_dino-vitb14_stats.npz ;;
esac; }
jit_case(){ case "$1" in
  artbench10)   IMG=artbench-10_images; NC=10;  FID=artbench-10_processed-fid_stats.npz; FDD=artbench-10-fd_dino-vitb14_stats.npz ;;
  food101)      IMG=food-101_images;    NC=101; FID=food-101_processed-fid_stats.npz;    FDD=food-101-fd_dino-vitb14_stats.npz ;;
  cub200)       IMG=cub-200-2011_images; NC=200; FID=cub-200-2011_processed-fid_stats.npz; FDD=cub-200-2011-fd_dino-vitb14_stats.npz ;;
esac; }

# family:ds:gpu
JOBS=(
  "dit:caltech101:0" "dit:cub200:2" "dit:stanfordcars:3"
  "jit:artbench10:4" "jit:food101:5" "jit:cub200:6"
)
for j in "${JOBS[@]}"; do
  IFS=':' read -r fam ds gpu <<< "$j"
  sess="fineval_${fam}_gpu${gpu}_${ds}"
  [[ "$fam" == "dit" ]] && FAMD=DiT || FAMD=JiT
  wd="$ADV/files/logs/finetuning/${ds}_${FAMD}_MeFT_CAIMF_puresadv_20260726"
  [[ -f "$wd/final_eval_metrics.csv" ]] && { echo "SKIP $sess (final_eval exists)"; continue; }
  gpu_try_claim "$gpu" "$sess" || { echo "SKIP $sess (GPU $gpu locked)"; continue; }
  if [[ "$fam" == "dit" ]]; then
    dit_case "$ds"; ROOT="$DATA/$LAT"; CM=caltech_dit_meft_caimf_posttrain
    EVAL=$EVAL_LATENT
    ARGS=( "--config.dataset.name=${ds}_latent" "--config.dataset.root=$ROOT" "--config.dataset.class_mapping_root=" "--config.dataset.num_classes=$NC" "--config.model.num_classes=$NC" "--config.sampling.num_classes=$NC" "--config.fid.cache_ref=$ADV/files/fid_stats/$FID" "--config.fd_dino.cache_ref=$ADV/files/fdd_stats/$FDD" )
  else
    jit_case "$ds"; ROOT="$DATA/$IMG"; CM=caltech_jit_meft_caimf_posttrain
    # JiT is PIXEL space (no VAE) -> eval through main_jit.py (plain-JiT eval
    # script), which builds imfJiT_DMF_H_16 from model_str. The CAMF config mode
    # already carries the MF-A op-point (omega=2.2, t in [0.1,1]). Disable the
    # image-dir num-classes scan since we pass explicit num_classes.
    EVAL=$EVAL_PIXEL
    ARGS=( "--config.dataset.num_classes_from_data=False" "--config.dataset.root=$ROOT" "--config.dataset.num_classes=$NC" "--config.model.num_classes=$NC" "--config.sampling.num_classes=$NC" "--config.fid.cache_ref=$MAIN/files/fid_stats/$FID" "--config.fd_dino.cache_ref=$MAIN/files/fdd_stats/$FDD" )
  fi
  if [[ "$fam" == "dit" ]]; then
    echo "LAUNCH $sess wd=$wd (latent eval)"
    screen -dmS "$sess" bash -c "
      cd $ADV
      export CUDA_VISIBLE_DEVICES=$gpu
      export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
      export MPLCONFIGDIR=/tmp/mpl-fineval-$ds-$fam
      CONFIG_MODE=$CM PYTHON=$PY USE_WANDB=False \
        bash $EVAL_LATENT '$wd' 1 2 -- ${ARGS[*]} \
        2>&1 | tee -a $ADV/files/logs/fineval_${fam}_${ds}.log
      rm -rf $LOCKDIR/gpu_$gpu
    "
  else
    # JiT-DMF pixel eval: main_imf_jit.py eval_only (config mode caltech_jit_dmf_meft
    # builds imfJiT_DMF, no VAE). force_metric_num_steps runs NFE 1 & 2 in one pass.
    # 10k samples per standing rule (the MF-A watcher used 5k). Rows land in the
    # run's own eval_metrics.csv (eval_phase=eval_only).
    ckpt=$(ls -d "$wd/best_fid/checkpoint_"* 2>/dev/null | sort -t_ -k2 -n | tail -1)
    echo "LAUNCH $sess wd=$wd (pixel main_imf_jit eval) ckpt=$ckpt"
    screen -dmS "$sess" bash -c "
      cd $MAIN
      export CUDA_VISIBLE_DEVICES=$gpu
      export TF_CPP_MIN_LOG_LEVEL=3 PYTHONWARNINGS=ignore XLA_PYTHON_CLIENT_PREALLOCATE=false
      export MPLCONFIGDIR=/tmp/mpl-fineval-$ds-$fam
      $PY main_imf_jit.py \
        --workdir=$wd \
        --config=$MAIN/configs/load_config.py:caltech_jit_dmf_meft \
        --config.eval_only=True \
        --config.load_from=$ckpt \
        --config.dataset.root=$ROOT \
        --config.dataset.num_classes=$NC \
        --config.model.num_classes=$NC \
        --config.fid.cache_ref=$MAIN/files/fid_stats/$FID \
        --config.fd_dino.cache_ref=$MAIN/files/fdd_stats/$FDD \
        --config.logging.use_wandb=False \
        --config.training.force_metric_num_steps=\"1 2\" \
        --config.fid.num_samples=10000 \
        2>&1 | tee -a $ADV/files/logs/fineval_${fam}_${ds}.log
      rm -rf $LOCKDIR/gpu_$gpu
    "
  fi
  sleep 3
done
echo "=== launched; screens: ==="; screen -ls | grep fineval || true
