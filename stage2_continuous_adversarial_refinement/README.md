# Stage 2: Continuous Adversarial MeanFlow (CAMF)

Refines a Stage 1 MeanFlow-Transfer student with pure adversarial post-training
(`lambda_imf = 0`, so the MeanFlow regression term is switched off). The generator
keeps the guidance behaviour it learned in Stage 1, and a discriminator scores the
guided finite-interval endpoint. This sharpens the one- and two-step samples.

Only the generator parameters are restored from Stage 1; the optimizer and the
discriminator start fresh. Stage 2 accepts any MeanFlow model, not only students
produced by Stage 1.

- [Quick Start](#quick-start)
- [Families](#families)
- [Config](#config)
- [Final NFE 1 and 2 Evaluation](#final-nfe-1-and-2-evaluation)
- [Structure](#structure)

## Quick Start

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

Use the **same dataset** the Stage 1 checkpoint was trained on. Unlike Stage 1, the
class count is resolved by the launcher and passed explicitly to the model, so a
mismatch will not load.

Results land in `runs/<dataset>_<family>_camf_<stamp>/`, with the best-FID
checkpoint in `best_fid/`. The launcher refuses to reuse a workdir that already
holds a checkpoint.

## Families

| family | entry                    | config mode                        | space  |
| ------ | ------------------------ | ---------------------------------- | ------ |
| `imf`  | `main_caimf.py`          | `caltech_imf_caimf_posttrain`      | latent |
| `sit`  | `main_caimf_sit_meft.py` | `caltech_sit_meft_caimf_posttrain` | latent |
| `dit`  | `main_caimf_sit_meft.py` | `caltech_dit_meft_caimf_posttrain` | latent |
| `jit`  | `main_caimf_jit_meft.py` | `caltech_jit_meft_caimf_posttrain` | pixel  |

`dit` reuses the `sit` entry point because the two share the `imfDiT` backbone
class; only the config differs.

Each family has its own discriminator: `caimf_discriminator.py` for `imf`,
`sit_meft_discriminator.py` for `sit` and `dit`, and `jit_meft_discriminator.py`
for `jit`.

## Config

Configs resolve as `configs/load_config.py:<mode>`, the same mechanism as Stage 1.
The knobs that define the paper recipe live under the `caimf:` block:

```yaml
caimf:
    lambda_imf: 0.0                    # pure adversarial: no MeanFlow regression term
    lambda_adv: 1.0
    lambda_cp: 0.001                   # centering penalty on the learned potential
    interval_eps: 0.001                # finite-interval width floor
    gen_learning_rate: 0.00001
    dis_learning_rate: 0.00001
    adam_beta1: 0.0
    adam_beta2: 0.95
    discriminator_warmup_batches: 5000
    discriminator_steps_per_cycle: 4   # D steps per G step
    max_posttrain_batches: 150000
```

Post-training uses an EMA of the generator (`training.use_ema: True`,
`ema_val: 0.9999`), and the released Stage 2 checkpoints are that EMA tree.

## Final NFE 1 and 2 Evaluation

Post-training evaluates at NFE 4 on a schedule. The headline Stage 2 numbers are at
NFE 1 and 2, produced by a separate pass:

```bash
CONFIG_MODE=caltech_dit_meft_caimf_posttrain \
  bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh \
  runs/cub-200-2011_dit_camf_<stamp>/best_fid 1 2
```

Use the driver that matches the family:

| family      | driver |
| ----------- | ------ |
| `sit`, `dit` | `scripts/eval_best_fid_steps_sit_meft_adversarial.sh` |
| `imf`       | `scripts/eval_best_fid_steps_plain_imf.sh` |
| `jit`       | `../stage1_meanflow_transfer/scripts/eval_best_fid_steps_plain_jit.sh` |

`jit` must use the Stage 1 pixel-space driver: the latent drivers build a VAE and
will fail on a pixel-space student. `scripts/run_final_evals.sh` records the exact
per-family invocations used for the paper tables.

## Structure

```
stage2_continuous_adversarial_refinement/
├── posttrain.sh              # launcher: family + stage-1 checkpoint -> a CAMF run
├── main_caimf.py             # entry point for imf
├── main_caimf_sit_meft.py    # entry point for sit and dit (shared backbone)
├── main_caimf_jit_meft.py    # entry point for jit (pixel space)
├── main_afm.py               # AFM endpoint-only adversary (ablation only)
├── main_afm_sit_meft.py
├── caimf.py                  # CAMF objective: finite-interval adversarial loss,
├── caimf_sit_meft.py         #   centering penalty, guided endpoint scoring
├── afm.py                    # AFM variant (ablation only)
├── afm_sit_meft.py
├── train_caimf.py            # adversarial training loops (G/D alternation)
├── train_caimf_sit_meft.py
├── train_caimf_jit_meft.py
├── train_sit_meft_common.py  # shared post-training helpers
├── train_jit_meft_common.py
├── train_afm.py              # AFM training loops (ablation only)
├── train_afm_sit_meft.py
├── train_jit.py
├── imf.py                    # model definitions, shared with Stage 1
├── sit.py
├── dit.py
├── plain_jit.py
├── models/
│   ├── caimf_discriminator.py       # CAMF discriminator for imf
│   ├── sit_meft_discriminator.py    #   for sit and dit
│   ├── jit_meft_discriminator.py    #   for jit (pixel space)
│   ├── afm_discriminator.py         # AFM endpoint adversary (ablation only)
│   ├── cafm_imf_discriminator.py    # original instantaneous CAFM (ablation only)
│   ├── imfDiT.py                    # MeanFlow DiT backbone
│   ├── jit.py                       # JiT pixel-space backbone
│   ├── torch_DiT.py                 # teacher definitions for weight conversion
│   ├── torch_SiT.py
│   ├── torch_models.py
│   ├── embedder.py
│   ├── convnext.py
│   ├── pmfDiT.py                    # pMF perceptual variant (ablation only)
│   ├── pmf_embedder.py
│   └── pmf_torch_models.py
├── configs/
│   ├── load_config.py
│   ├── default.py
│   ├── caltech_imf_caimf_posttrain_config.yml
│   ├── caltech_sit_meft_caimf_posttrain_config.yml
│   ├── caltech_dit_meft_caimf_posttrain_config.yml
│   └── caltech_jit_meft_caimf_posttrain_config.yml
├── utils/                    # same helpers as Stage 1 (sampling, FID, FD-DINO,
│                             #   checkpointing, data, logging)
└── scripts/
    ├── run_final_evals.sh                      # per-family paper-table invocations
    ├── eval_best_fid_steps_sit_meft_adversarial.sh   # sit and dit
    ├── eval_best_fid_steps_plain_imf.sh              # imf
    └── eval_best_fid_steps.sh
```
