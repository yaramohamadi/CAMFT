# Stage 1: MeanFlow-Transfer

Fine-tunes a pretrained ImageNet teacher into a few-step MeanFlow student on a
target domain. The student learns to jump across the flow in a handful of steps
and keeps the teacher's classifier-free guidance behaviour.

## Run it

```bash
WEIGHTS_DIR=/path/to/weights bash train.sh <family> [dataset]
```

`<family>` is one of `imf`, `sit`, `dit`, `jit`. `[dataset]` is one of
`artbench-10`, `caltech-101`, `cub-200-2011`, `food-101`, `stanford-cars` and
defaults to `cub-200-2011`. The launcher sets the entry point, config, base
weights, sampling guidance, and the FID / FD-DINO reference statistics for that
combination. Results land under `runs/<dataset>_<family>_<stamp>/`, with the
best-FID checkpoint in `best_fid/`. Pass that checkpoint to Stage 2.

Extra flags are forwarded to the training script:

```bash
bash train.sh dit food-101 --config.training.num_steps=1000
```

## What each family uses

| family | entry             | config                     | base weights          | omega |
| ------ | ----------------- | -------------------------- | --------------------- | ----- |
| `imf`  | `main.py`         | `plain_imf_finetune`       | `iMF-XL-2-full`       | 7.5   |
| `sit`  | `main.py`         | `caltech_sit_dmf_finetune` | `SiT-XL-2-256.pt`     | 1.5   |
| `dit`  | `main.py`         | `caltech_dit_dmf_ddpmv`    | `DiT-XL-2-256x256.pt` | 1.5   |
| `jit`  | `main_imf_jit.py` | `caltech_jit_dmf_meft`     | `JiT-H-16-256.pth`    | 2.2   |

The DiT run additionally clips gradients at global norm 1.0. The iMF evaluation
restricts the EMA time window to `[0.4, 0.65]`. Class count is read from the data
itself (`num_classes_from_data`), so each dataset resolves to its own count
automatically: ArtBench-10 10, Caltech-101 101, CUB-200 200, Food-101 101,
Stanford Cars 196.

The `imf`, `sit`, and `dit` families read the precomputed VAE latents; `jit`
works in pixel space and reads the raw images.

## Layout

```
main.py, main_jit.py    entry points
train.py, train_jit.py  training loops
imf.py, sit.py, dit.py, plain_jit.py   model definitions
models/                 network modules
utils/                  sampling, FID, FD-DINO, checkpoint, and data helpers
configs/                config loader, defaults, and the family config files
scripts/                best-FID evaluation drivers
```

The config files are named after the dataset they were first written for
(`caltech_*`, `plain_*`); they are used for every target domain, with the
dataset-specific paths and reference statistics overridden on the command line
by the launcher.

`scripts/` holds the best-checkpoint evaluation drivers used for the paper
tables. They were written for a specific multi-GPU host and use absolute paths
and `screen`; treat them as a record of what was run rather than portable entry
points.




## Stage 1: MeanFlow-Transfer (MF-T)

Fine-tunes a pretrained teacher into a few-step MeanFlow student on a target
domain.


### Config

Configs are resolved as `configs/load_config.py:<mode>`, which loads
`configs/<mode>_config.yml` and merges it over `configs/default.py`. `train.sh`
picks the mode, the base weights, and the sampling guidance per family:

| family | entry             | config mode                | base weights          | omega |
| ------ | ----------------- | -------------------------- | --------------------- | ----- |
| `imf`  | `main.py`         | `plain_imf_finetune`       | `iMF-XL-2-full`       | 7.5   |
| `sit`  | `main.py`         | `caltech_sit_dmf_finetune` | `SiT-XL-2-256.pt`     | 1.5   |
| `dit`  | `main.py`         | `caltech_dit_dmf_ddpmv`    | `DiT-XL-2-256x256.pt` | 1.5   |
| `jit`  | `main_imf_jit.py` | `caltech_jit_dmf_meft`     | `JiT-H-16-256.pth`    | 2.2   |

Two family-specific overrides are applied by the launcher, exactly as in the
paper runs: `dit` clips gradients at global norm 1.0, and `imf` restricts the
sampling time window to `[0.4, 0.65]`.

Config files are named after the dataset they were first written for
(`caltech_*`, `plain_*`) but are used for **every** target domain — the launcher
overrides the dataset paths, class count, and reference statistics on the command
line. Anything you append to the launcher is forwarded to the Python entry point,
so any config field can be overridden ad hoc.

### 1) Train

```bash
cd stage1_meanflow_transfer
WEIGHTS_DIR=/path/to/weights bash train.sh <family> [dataset] [extra args]
```

`<family>` is `imf`, `sit`, `dit`, or `jit`. `[dataset]` is one of the five slugs
and defaults to `cub-200-2011`. For example:

```bash
WEIGHTS_DIR=/path/to/weights bash train.sh dit cub-200-2011
WEIGHTS_DIR=/path/to/weights bash train.sh jit food-101 --config.training.num_steps=1000
```

Results land in `runs/<dataset>_<family>_<stamp>/`, with the best-FID checkpoint
in `best_fid/checkpoint_*`. That directory is what you pass to Stage 2.

The class count is read from the data itself (`num_classes_from_data`), so each
domain resolves to the count in the
[target-domain table](#method) without an override.

### 2) Evaluate a Checkpoint

FID and FD-DINO are evaluated periodically during training, so a normal run needs
no separate eval pass. To re-evaluate a saved checkpoint at specific step counts:

```bash
CONFIG_MODE=caltech_dit_dmf_ddpmv bash scripts/eval_best_fid_steps.sh \
  runs/cub-200-2011_dit_<stamp>/best_fid 1 2 4
```

The trailing numbers are the NFEs to evaluate. Every metric uses 10,000 samples.
