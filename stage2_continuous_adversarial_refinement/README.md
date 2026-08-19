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




## Stage 2: Continuous Adversarial MeanFlow (CAMF)

Refines a Stage 1 student with pure adversarial post-training (`lambda_imf = 0`).
Only the generator parameters are restored from Stage 1; the optimizer and the
discriminator start fresh.

### Config

| family | entry                    | config mode                        | space  |
| ------ | ------------------------ | ---------------------------------- | ------ |
| `imf`  | `main_caimf.py`          | `caltech_imf_caimf_posttrain`      | latent |
| `sit`  | `main_caimf_sit_meft.py` | `caltech_sit_meft_caimf_posttrain` | latent |
| `dit`  | `main_caimf_sit_meft.py` | `caltech_dit_meft_caimf_posttrain` | latent |
| `jit`  | `main_caimf_jit_meft.py` | `caltech_jit_meft_caimf_posttrain` | pixel  |

`dit` reuses the `sit` entry point because the two share the `imfDiT` backbone
class; only the config differs. The knobs that define the paper recipe live under
the `caimf:` block of each config:

```yaml
caimf:
    lambda_imf: 0.0                    # pure adversarial: no MeanFlow regression term
    lambda_adv: 1.0
    lambda_cp: 0.001                   # centering penalty on the learned potential
    interval_eps: 0.001                # finite-interval width floor
    gen_learning_rate: 0.00001
    dis_learning_rate: 0.00001
    discriminator_warmup_batches: 5000
    discriminator_steps_per_cycle: 4   # D steps per G step
    max_posttrain_batches: 150000
```

Unlike Stage 1, the class count is resolved by the launcher and passed
explicitly, so it must match the Stage 1 checkpoint.

### 1) Post-train

```bash
cd stage2_continuous_adversarial_refinement
bash posttrain.sh <family> <mf_t_checkpoint> [dataset] [extra args]
```

`<mf_t_checkpoint>` is the `best_fid/checkpoint_*` directory from the matching
Stage 1 run:

```bash
bash posttrain.sh dit \
  ../stage1_meanflow_transfer/runs/cub-200-2011_dit_<stamp>/best_fid/checkpoint_27500 \
  cub-200-2011
```

Use the **same dataset** the Stage 1 checkpoint was trained on: the class count is
passed to the model and a mismatch will not load. Results land in
`runs/<dataset>_<family>_camf_<stamp>/`, best-FID checkpoint in `best_fid/`.

### 2) Final NFE 1 and 2 Evaluation

Post-training evaluates at NFE 4 on a schedule. The headline Stage 2 numbers are
at NFE 1 and 2, produced by a separate pass:

```bash
CONFIG_MODE=caltech_dit_meft_caimf_posttrain \
  bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh \
  runs/cub-200-2011_dit_camf_<stamp>/best_fid 1 2
```

Use the matching driver for the family: `..._sit_meft_adversarial.sh` for `sit`
and `dit`, `..._plain_imf.sh` for `imf`, and for `jit` the pixel-space driver in
Stage 1 (`stage1_meanflow_transfer/scripts/eval_best_fid_steps_plain_jit.sh`) —
the latent driver builds a VAE and will fail on a pixel-space student.
`scripts/run_final_evals.sh` shows the exact per-family invocations used for the
paper tables.

