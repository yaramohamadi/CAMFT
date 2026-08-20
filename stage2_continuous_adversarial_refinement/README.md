# Stage 2: Continuous Adversarial MeanFlow (CAMF)

Pure adversarial post-training of a Stage 1 student: `lambda_imf = 0`, so the
MeanFlow regression term is off and a discriminator scores the guided
finite-interval endpoint. This is what sharpens the 1- and 2-step samples.

Only the generator is restored from Stage 1; the optimizer and discriminator start
fresh. Any MeanFlow model works here, not just Stage 1 output.

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

Use the same dataset the Stage 1 checkpoint was trained on. Unlike Stage 1, the
class count is passed explicitly by the launcher, so a mismatch will not load.

Results land in `runs/<dataset>_<family>_camf_<stamp>/`, best-FID checkpoint in
`best_fid/`. The launcher refuses a workdir that already holds a checkpoint.

## Families

| family | entry | config mode | space |
| --- | --- | --- | --- |
| `imf` | `main_caimf.py` | `caltech_imf_caimf_posttrain` | latent |
| `sit` | `main_caimf_sit_meft.py` | `caltech_sit_meft_caimf_posttrain` | latent |
| `dit` | `main_caimf_sit_meft.py` | `caltech_dit_meft_caimf_posttrain` | latent |
| `jit` | `main_caimf_jit_meft.py` | `caltech_jit_meft_caimf_posttrain` | pixel |

`dit` reuses the `sit` entry point since both share the `imfDiT` backbone class; only
the config differs. Each family has its own discriminator — `caimf_discriminator.py`
for `imf`, `sit_meft_discriminator.py` for `sit` and `dit`,
`jit_meft_discriminator.py` for `jit`.

## Config

Same mechanism as Stage 1. The recipe lives under the `caimf:` block:

```yaml
caimf:
    lambda_imf: 0.0                    # pure adversarial: no regression term
    lambda_adv: 1.0
    lambda_cp: 0.001                   # centering penalty on the potential
    interval_eps: 0.001                # finite-interval width floor
    gen_learning_rate: 0.00001
    dis_learning_rate: 0.00001
    adam_beta1: 0.0
    adam_beta2: 0.95
    discriminator_warmup_batches: 5000
    discriminator_steps_per_cycle: 4   # D steps per G step
    max_posttrain_batches: 150000
```

Post-training tracks an EMA of the generator (`use_ema: True`, `ema_val: 0.9999`),
and the released Stage 2 checkpoints are that EMA tree.

## Final NFE 1 and 2 Evaluation

Post-training evaluates at NFE 4 on a schedule. The headline numbers are at NFE 1
and 2, from a separate pass:

```bash
CONFIG_MODE=caltech_dit_meft_caimf_posttrain \
  bash scripts/eval_best_fid_steps_sit_meft_adversarial.sh \
  runs/cub-200-2011_dit_camf_<stamp>/best_fid 1 2
```

Drivers by family:

| family | driver |
| --- | --- |
| `sit`, `dit` | `scripts/eval_best_fid_steps_sit_meft_adversarial.sh` |
| `imf` | `scripts/eval_best_fid_steps_plain_imf.sh` |
| `jit` | `../stage1_meanflow_transfer/scripts/eval_best_fid_steps_plain_jit.sh` |

`jit` has to use the Stage 1 pixel-space driver; the latent ones build a VAE and
fail on a pixel-space student. `scripts/run_final_evals.sh` records the exact
invocations behind the paper tables.

## Structure

Shares `imf.py`, `sit.py`, `dit.py`, `plain_jit.py`, `models/`, and `utils/` with
Stage 1. What is specific to this stage:

```
stage2_continuous_adversarial_refinement/
├── posttrain.sh              # launcher: family + stage-1 checkpoint -> a CAMF run
├── main_caimf.py             # entry point, imf
├── main_caimf_sit_meft.py    # entry point, sit and dit
├── main_caimf_jit_meft.py    # entry point, jit
├── caimf.py                  # the CAMF objective
├── caimf_sit_meft.py
├── train_caimf.py            # G/D alternation
├── train_caimf_sit_meft.py
├── train_caimf_jit_meft.py
├── train_sit_meft_common.py
├── train_jit_meft_common.py
├── train_jit.py
├── main_afm.py               # AFM endpoint-only adversary, ablations only
├── main_afm_sit_meft.py
├── afm.py
├── afm_sit_meft.py
├── train_afm.py
├── train_afm_sit_meft.py
├── models/
│   ├── caimf_discriminator.py
│   ├── sit_meft_discriminator.py
│   ├── jit_meft_discriminator.py
│   ├── afm_discriminator.py         # ablations only
│   └── cafm_imf_discriminator.py    # original instantaneous CAFM, ablations only
├── configs/
│   ├── caltech_imf_caimf_posttrain_config.yml
│   ├── caltech_sit_meft_caimf_posttrain_config.yml
│   ├── caltech_dit_meft_caimf_posttrain_config.yml
│   └── caltech_jit_meft_caimf_posttrain_config.yml
└── scripts/
    ├── run_final_evals.sh
    ├── eval_best_fid_steps_sit_meft_adversarial.sh
    ├── eval_best_fid_steps_plain_imf.sh
    └── eval_best_fid_steps.sh
```
