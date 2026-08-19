# Stage 1: MeanFlow-Transfer

Fine-tunes a pretrained ImageNet teacher into a few-step MeanFlow student on
CUB-200-2011. The student learns to jump across the flow in a handful of steps
and keeps the teacher's classifier-free guidance behaviour.

## Run it

```bash
WEIGHTS_DIR=/path/to/weights bash train_cub200.sh <family>
```

`<family>` is one of `imf`, `sit`, `dit`, `jit`. The launcher sets the entry
point, config, base weights, and sampling guidance for that family. Results land
under `runs/cub200_<family>_<stamp>/`, with the best-FID checkpoint in
`best_fid/`. Pass that checkpoint to Stage 2.

## What each family uses

| family | entry         | config                     | base weights           | omega |
| ------ | ------------- | -------------------------- | ---------------------- | ----- |
| `imf`  | `main.py`     | `plain_imf_finetune`       | `iMF-XL-2-full`        | 7.5   |
| `sit`  | `main.py`     | `caltech_sit_dmf_finetune` | `SiT-XL-2-256.pt`      | 7.5   |
| `dit`  | `main.py`     | `caltech_dit_dmf_ddpmv`    | `DiT-XL-2-256x256.pt`  | 1.5   |
| `jit`  | `main_jit.py` | `plain_jit_finetune`       | `JiT-H-16-256.pth`     | 2.2   |

The DiT run additionally clips gradients at global norm 1.0. The iMF evaluation
restricts the EMA time window to `[0.4, 0.65]`. Class count is read from the
latent data, so CUB-200 resolves to 200 classes automatically.

## Layout

```
main.py, main_jit.py    entry points
train.py, train_jit.py  training loops
imf.py, sit.py, dit.py, plain_jit.py   model definitions
models/                 network modules
utils/                  sampling, FID, FD-DINO, checkpoint, and data helpers
configs/                config loader, defaults, and the four CUB config files
scripts/                best-FID evaluation drivers
```
