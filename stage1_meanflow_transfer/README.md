# Stage 1: MeanFlow-Transfer (MF-T)

Fine-tunes a pretrained ImageNet-256 teacher into a few-step MeanFlow student on a
target domain. The student learns to jump across the flow in a handful of steps and
keeps the teacher's classifier-free guidance behaviour. Four teacher families are
supported, spanning all four prediction parameterizations.

- [Quick Start](#quick-start)
- [Families](#families)
- [Config](#config)
- [Evaluating a Checkpoint](#evaluating-a-checkpoint)
- [Structure](#structure)

## Quick Start

```bash
cd stage1_meanflow_transfer
WEIGHTS_DIR=/path/to/weights bash train.sh <family> [dataset] [extra args]
```

`<family>` is `imf`, `sit`, `dit`, or `jit`. `[dataset]` is one of `artbench-10`,
`caltech-101`, `cub-200-2011`, `food-101`, `stanford-cars` and defaults to
`cub-200-2011`. For example:

```bash
WEIGHTS_DIR=/path/to/weights bash train.sh dit cub-200-2011
WEIGHTS_DIR=/path/to/weights bash train.sh jit food-101 --config.training.num_steps=1000
```

Anything you append is forwarded to the Python entry point, so any config field can
be overridden ad hoc.

Results land in `runs/<dataset>_<family>_<stamp>/`, with the best-FID checkpoint in
`best_fid/checkpoint_*`. That directory is what you pass to Stage 2.

Environment variables the launcher reads:

| variable          | default          | purpose |
| ----------------- | ---------------- | ------- |
| `WEIGHTS_DIR`     | `../weights`     | folder holding the four teacher checkpoints |
| `PYTHON`          | `python3`        | interpreter to run the entry point with |
| `USE_WANDB`       | `False`          | enable Weights & Biases logging |
| `FID_NUM_SAMPLES` | `10000`          | samples per metric evaluation (the paper uses 10k) |

## Families

`train.sh` picks the entry point, config mode, base weights, and sampling guidance
per family — these are exactly the settings that produced the paper numbers:

| family | entry             | config mode                | base weights          | omega | space  |
| ------ | ----------------- | -------------------------- | --------------------- | ----- | ------ |
| `imf`  | `main.py`         | `plain_imf_finetune`       | `iMF-XL-2-full`       | 7.5   | latent |
| `sit`  | `main.py`         | `caltech_sit_dmf_finetune` | `SiT-XL-2-256.pt`     | 1.5   | latent |
| `dit`  | `main.py`         | `caltech_dit_dmf_ddpmv`    | `DiT-XL-2-256x256.pt` | 1.5   | latent |
| `jit`  | `main_imf_jit.py` | `caltech_jit_dmf_meft`     | `JiT-H-16-256.pth`    | 2.2   | pixel  |

Two family-specific overrides are applied by the launcher, as in the paper runs:
`dit` clips gradients at global norm 1.0, and `imf` restricts the sampling time
window to `[0.4, 0.65]`.

The latent families (`imf`, `sit`, `dit`) read the precomputed VAE latents; `jit`
works in pixel space and reads the raw images. See the top-level README for how to
build both.

The class count is read from the data itself (`num_classes_from_data` is set in
every config), so each domain resolves to its own count with no override:
ArtBench-10 10, Caltech-101 101, CUB-200-2011 200, Food-101 101, Stanford Cars 196.

## Config

Configs resolve as `configs/load_config.py:<mode>`, which loads
`configs/<mode>_config.yml` and merges it over `configs/default.py`.

Config files are named after the dataset they were first written for (`caltech_*`,
`plain_*`) but are used for **every** target domain — the launcher overrides the
dataset root, reference statistics, and sampling guidance on the command line.

## Evaluating a Checkpoint

FID and FD-DINO are evaluated periodically during training, so a normal run needs
no separate eval pass. To re-evaluate a saved checkpoint at specific step counts:

```bash
CONFIG_MODE=caltech_dit_dmf_ddpmv bash scripts/eval_best_fid_steps.sh \
  runs/cub-200-2011_dit_<stamp>/best_fid 1 2 4
```

The trailing numbers are the NFEs to evaluate; they default to `2 4`. The first
argument accepts either the run directory or the `best_fid` directory itself. Set
`EVAL_PLATFORM=cpu` to force CPU evaluation if a GPU restore runs out of memory.

`scripts/eval_best_fid_steps_plain_*.sh` are the per-family drivers for the plain
fine-tuned teacher baselines reported alongside `MF-T`. Note that
`eval_best_fid_steps_plain_jit.sh` is the pixel-space driver, and Stage 2 also uses
it for `jit` — the latent drivers build a VAE and fail on a pixel-space student.

## Structure

```
stage1_meanflow_transfer/
├── train.sh                  # launcher: family + dataset -> a full paper run
├── main.py                   # entry point for imf / sit / dit (latent space)
├── main_imf_jit.py           # entry point for jit MeanFlow-Transfer (pixel space)
├── main_jit.py               # entry point for the plain jit fine-tuning baseline
├── train.py                  # training loop for the latent families
├── train_imf_jit.py          # training loop for jit MeanFlow-Transfer
├── train_jit.py              # training loop for the plain jit baseline
├── imf.py                    # per-family model + objective definitions
├── sit.py                    #   (velocity mapping, MeanFlow target, sampling)
├── dit.py
├── plain_jit.py
├── models/
│   ├── imfDiT.py             # the MeanFlow DiT backbone used by imf / sit / dit
│   ├── jit.py                # the JiT pixel-space backbone
│   ├── torch_DiT.py          # teacher definitions, used to read the .pt weights
│   ├── torch_SiT.py          #   on CPU before conversion to JAX arrays
│   ├── torch_models.py
│   ├── embedder.py           # timestep and class embedders
│   ├── convnext.py
│   ├── pmfDiT.py             # pMF perceptual variant (ablation only, not in the
│   ├── pmf_embedder.py       #   main results)
│   └── pmf_torch_models.py
├── configs/
│   ├── load_config.py        # resolves configs/load_config.py:<mode>
│   ├── default.py            # base config every mode is merged over
│   ├── plain_imf_finetune_config.yml
│   ├── caltech_sit_dmf_finetune_config.yml
│   ├── caltech_dit_dmf_ddpmv_config.yml
│   ├── caltech_jit_dmf_meft_config.yml
│   └── plain_jit_finetune_config.yml      # plain teacher baseline
├── utils/
│   ├── input_pipeline.py     # data loading and the guided-diffusion center crop
│   ├── data_util.py
│   ├── vae_util.py           # Stable Diffusion VAE encode / decode
│   ├── sample_util.py        # MeanFlow sampling
│   ├── sit_sample_util.py    # per-family samplers and transports
│   ├── sit_transport_jax.py
│   ├── sit_official_transport.py
│   ├── dit_sample_util.py
│   ├── dit_diffusion.py
│   ├── imf_param_util.py     # velocity-space parameter mapping between families
│   ├── fid_util.py           # FID against the InceptionV3 references
│   ├── dino_util.py          # FD-DINO against the DINOv2 ViT-B/14 references
│   ├── dinov2_jax.py         # pure-JAX DINOv2-B/14
│   ├── jax_fid/              # InceptionV3 in JAX
│   ├── ckpt_util.py          # Orbax checkpoint save / restore
│   ├── trainstate_util.py    # train state construction
│   ├── sit_trainstate_util.py
│   ├── state_util.py
│   ├── ema_util.py           # EMA parameter tracking
│   ├── lr_utils.py           # learning-rate schedules
│   ├── muon_util.py          # Muon optimizer (not used in the main results)
│   ├── auxloss_util.py
│   ├── logging_util.py       # console and W&B logging
│   ├── eval_csv_util.py      # writes eval_metrics.csv per run
│   └── preview_util.py       # sample grids during training
└── scripts/
    ├── eval_best_fid_steps.sh            # re-evaluate an MF-T checkpoint
    ├── eval_best_fid_steps_plain_imf.sh  # plain fine-tuned teacher baselines
    ├── eval_best_fid_steps_plain_sit.sh
    ├── eval_best_fid_steps_plain_dit.sh
    └── eval_best_fid_steps_plain_jit.sh  # pixel space; also used by Stage 2 jit
```
