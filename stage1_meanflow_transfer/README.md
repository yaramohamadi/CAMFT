# Stage 1: MeanFlow-Transfer (MF-T)

Fine-tunes a pretrained ImageNet-256 teacher into a few-step MeanFlow student on a
target domain. The student keeps the teacher's classifier-free guidance behaviour.

## Quick Start

```bash
cd stage1_meanflow_transfer
WEIGHTS_DIR=/path/to/weights bash train.sh <family> [dataset] [extra args]
```

`<family>` is `imf`, `sit`, `dit`, or `jit`. `[dataset]` is one of `artbench-10`,
`caltech-101`, `cub-200-2011`, `food-101`, `stanford-cars` and defaults to
`cub-200-2011`.

```bash
WEIGHTS_DIR=/path/to/weights bash train.sh dit cub-200-2011
WEIGHTS_DIR=/path/to/weights bash train.sh jit food-101 --config.training.num_steps=1000
```

Anything you append is forwarded to the entry point, so any config field can be
overridden ad hoc.

Results land in `runs/<dataset>_<family>_<stamp>/`, best-FID checkpoint in
`best_fid/checkpoint_*`. That directory is what you pass to Stage 2.

The launcher also reads `WEIGHTS_DIR` (default `../weights`), `PYTHON` (`python3`),
`USE_WANDB` (`False`), and `FID_NUM_SAMPLES` (`10000`, the paper setting).

## Families

| family | entry | config mode | base weights | omega | space |
| --- | --- | --- | --- | --- | --- |
| `imf` | `main.py` | `plain_imf_finetune` | `iMF-XL-2-full` | 7.5 | latent |
| `sit` | `main.py` | `caltech_sit_dmf_finetune` | `SiT-XL-2-256.pt` | 1.5 | latent |
| `dit` | `main.py` | `caltech_dit_dmf_ddpmv` | `DiT-XL-2-256x256.pt` | 1.5 | latent |
| `jit` | `main_imf_jit.py` | `caltech_jit_dmf_meft` | `JiT-H-16-256.pth` | 2.2 | pixel |

`train.sh` applies two family-specific overrides: `dit` clips gradients at global
norm 1.0, and `imf` restricts the sampling window to `[0.4, 0.65]`.

`imf`, `sit`, and `dit` read the precomputed VAE latents. `jit` works in pixel space
and reads the raw images. See the top-level README for how to build both.

Class count comes from the data (`num_classes_from_data`), so each domain resolves
on its own: ArtBench-10 10, Caltech-101 101, CUB-200-2011 200, Food-101 101,
Stanford Cars 196.

## Config

Configs resolve as `configs/load_config.py:<mode>`, which loads
`configs/<mode>_config.yml` over `configs/default.py`.

The files are named after the dataset they were first written for (`caltech_*`,
`plain_*`) but are used for every domain — the launcher overrides the dataset root,
reference statistics, and guidance on the command line.

## Evaluating a Checkpoint

FID and FD-DINO run periodically during training, so a normal run needs no separate
eval pass. To re-evaluate a saved checkpoint:

```bash
CONFIG_MODE=caltech_dit_dmf_ddpmv bash scripts/eval_best_fid_steps.sh \
  runs/cub-200-2011_dit_<stamp>/best_fid 1 2 4
```

Trailing numbers are the NFEs, defaulting to `2 4`. The first argument takes either
the run directory or `best_fid` itself. `EVAL_PLATFORM=cpu` forces CPU evaluation if
a GPU restore runs out of memory.

The `eval_best_fid_steps_plain_*.sh` drivers evaluate the plain fine-tuned teacher
baselines. The `plain_jit` one is pixel space, and Stage 2 borrows it for `jit`.

## Structure

```
stage1_meanflow_transfer/
├── train.sh                  # launcher: family + dataset -> a full run
├── main.py                   # entry point, imf / sit / dit
├── main_imf_jit.py           # entry point, jit
├── main_jit.py               # plain jit fine-tuning baseline
├── train.py
├── train_imf_jit.py
├── train_jit.py
├── imf.py                    # per-family velocity mapping, MeanFlow target, sampling
├── sit.py
├── dit.py
├── plain_jit.py
├── models/
│   ├── imfDiT.py             # MeanFlow DiT backbone (imf / sit / dit)
│   ├── jit.py                # JiT pixel-space backbone
│   ├── torch_DiT.py          # teacher defs, only to read the .pt weights on CPU
│   ├── torch_SiT.py
│   ├── torch_models.py
│   ├── embedder.py
│   ├── convnext.py
│   ├── pmfDiT.py             # pMF variant, ablations only
│   ├── pmf_embedder.py
│   └── pmf_torch_models.py
├── configs/
│   ├── load_config.py
│   ├── default.py
│   ├── plain_imf_finetune_config.yml
│   ├── caltech_sit_dmf_finetune_config.yml
│   ├── caltech_dit_dmf_ddpmv_config.yml
│   ├── caltech_jit_dmf_meft_config.yml
│   └── plain_jit_finetune_config.yml
├── utils/
│   ├── input_pipeline.py     # loading + the guided-diffusion center crop
│   ├── data_util.py
│   ├── vae_util.py
│   ├── sample_util.py
│   ├── sit_sample_util.py
│   ├── sit_transport_jax.py
│   ├── sit_official_transport.py
│   ├── dit_sample_util.py
│   ├── dit_diffusion.py
│   ├── imf_param_util.py     # velocity-space mapping between families
│   ├── fid_util.py
│   ├── dino_util.py
│   ├── dinov2_jax.py         # pure-JAX DINOv2-B/14
│   ├── jax_fid/              # InceptionV3 in JAX
│   ├── ckpt_util.py
│   ├── trainstate_util.py
│   ├── sit_trainstate_util.py
│   ├── state_util.py
│   ├── ema_util.py
│   ├── lr_utils.py
│   ├── muon_util.py          # unused in the main results
│   ├── auxloss_util.py
│   ├── logging_util.py
│   ├── eval_csv_util.py
│   └── preview_util.py
└── scripts/
    ├── eval_best_fid_steps.sh
    ├── eval_best_fid_steps_plain_imf.sh
    ├── eval_best_fid_steps_plain_sit.sh
    ├── eval_best_fid_steps_plain_dit.sh
    └── eval_best_fid_steps_plain_jit.sh   # pixel space; Stage 2 jit uses this too
```
