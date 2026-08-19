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
