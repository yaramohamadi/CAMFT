# Stage 2: Continuous Adversarial MeanFlow

Refines a Stage 1 MeanFlow-Transfer student with pure adversarial post-training
(the MeanFlow regression weight is set to zero). The generator keeps the
guidance behaviour it learned in Stage 1, and a discriminator scores the guided
few-step endpoint. This sharpens the one and two step samples.

## Run it

```bash
bash posttrain.sh <family> <mf_t_checkpoint> [dataset]
```

`<family>` is one of `imf`, `sit`, `dit`, `jit`. `<mf_t_checkpoint>` is the
`best_fid/checkpoint_*` directory from the matching Stage 1 run. `[dataset]` is
one of `artbench-10`, `caltech-101`, `cub-200-2011`, `food-101`,
`stanford-cars` and defaults to `cub-200-2011`; use the same one the Stage 1
checkpoint was trained on, since a class-count mismatch will not load.

Only the generator parameters are restored; the optimizer and discriminator
start fresh. Results land under `runs/<dataset>_<family>_camf_<stamp>/`, with
the best-FID checkpoint in `best_fid/`. Evaluate the final model at NFE 1 and 2
with the drivers under `scripts/`.

## What each family uses

| family | entry                     | config                             | space  |
| ------ | ------------------------- | ---------------------------------- | ------ |
| `imf`  | `main_caimf.py`           | `caltech_imf_caimf_posttrain`      | latent |
| `sit`  | `main_caimf_sit_meft.py`  | `caltech_sit_meft_caimf_posttrain` | latent |
| `dit`  | `main_caimf_sit_meft.py`  | `caltech_dit_meft_caimf_posttrain` | latent |
| `jit`  | `main_caimf_jit_meft.py`  | `caltech_jit_meft_caimf_posttrain` | pixel  |

DiT reuses the SiT post-training entry point because the two share the imfDiT
backbone class; only the config differs. The class count is resolved from the
dataset by the launcher and passed explicitly on the command line.

## Layout

```
main_caimf*.py          entry points (one per family group)
caimf*.py, afm*.py      adversarial training definitions
train_caimf*.py         training loops
imf.py, sit.py, dit.py, plain_jit.py   model definitions
models/                 networks, including the discriminators
utils/                  sampling, FID, FD-DINO, checkpoint, and data helpers
configs/                config loader, defaults, and the four family config files
scripts/                best-FID evaluation drivers
```
