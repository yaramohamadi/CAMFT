# Continuous Adversarial MeanFlow Transfer
Official code for *Continuous Adversarial MeanFlow Transfer*.

## Method

The pipeline has two stages, run in order:

1. **[Stage 1: MeanFlow-Transfer](#stage-1-meanflow-transfer-mf-t).** Fine-tune a
   pretrained ImageNet teacher into a MeanFlow student on the target domain. The
   student samples in a handful of steps and inherits the teacher's
   classifier-free guidance behaviour.
2. **[Stage 2: Continuous Adversarial MeanFlow](#stage-2-continuous-adversarial-meanflow-camf).**
   Refine the Stage 1 student with pure adversarial post-training. The generator
   keeps its guidance behaviour; a discriminator scores the guided few-step
   endpoint, sharpening the one and two step samples.

First stage supports four teacher families, spanning all four prediction
parameterizations:

| family | teacher                | predicts        | space  |
| ------ | ---------------------- | --------------- | ------ |
| `imf`  | improved MeanFlow XL/2 | mean velocity u | latent |
| `sit`  | SiT-XL/2               | velocity v      | latent |
| `dit`  | DiT-XL/2               | noise eps       | latent |
| `jit`  | JiT-H/16               | input x         | pixel  |

Second stage supports MeanFlow family models, which can be derived from the first stage. 
The two stages are kept as separate packages on purpose. They share most of their
code, but a few modules (`imf.py`, `configs/default.py`, `utils/sample_util.py`,
`models/imfDiT.py`) carry stage-specific changes that cannot be collapsed into
one shared copy without breaking one of the stages. Each package is therefore
self-contained and runs on its own.

## Setup

### Environment

Python 3.12, JAX on CUDA 12. All versions are pinned in `requirements.txt`.

```bash
git lfs install                                    # stats/ is stored with Git LFS
git clone https://github.com/yaramohamadi/CAMFT.git
cd CAMFT

python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

The pinned set installs `torch` as a **CPU** build on purpose. Torch is only used
to read the released PyTorch teacher checkpoints on the host before their tensors
are converted to JAX arrays; all training and sampling runs in JAX.

Every run in the paper used a single 80 GB H100.

### Teacher Weights

First stage starts from publicly released ImageNet-256 teacher checkpoints. Download each from its original source, put all four in
one folder, and point `WEIGHTS_DIR` at it. Expected filenames:

| family | filename              |
| ------ | --------------------- |
| `dit`  | `DiT-XL-2-256x256.pt` |
| `sit`  | `SiT-XL-2-256.pt`     |
| `jit`  | `JiT-H-16-256.pth`    |
| `imf`  | `iMF-XL-2-full`       |

See [`WEIGHTS.md`](WEIGHTS.md) for sources and details.

### Data

Five target domains are tested:

| slug           | dataset       | classes | source |
| -------------- | ------------- | ------- | ------ |
| `artbench-10`  | ArtBench-10   | 10      | [liaopeiyuan/artbench](https://github.com/liaopeiyuan/artbench) |
| `caltech-101`  | Caltech-101   | 101     | [CaltechDATA](https://data.caltech.edu/records/mzrjq-6wc02) |
| `cub-200-2011` | CUB-200-2011  | 200     | [Caltech Vision](https://www.vision.caltech.edu/datasets/cub_200_2011/) &middot; [CaltechDATA](https://data.caltech.edu/records/65de6-vp158) |
| `food-101`     | Food-101      | 101     | [ETH Zurich](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) &middot; [HF](https://huggingface.co/datasets/ethz/food101) |
| `stanford-cars`| Stanford Cars | 196     | [HF mirror](https://huggingface.co/datasets/Donghyun99/Stanford-Cars) &middot; [Kaggle mirror](https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder) |

Build a `data/` directory with one subfolder per domain you want to run:

```
data/
├── <dataset>_processed_latents/   # VAE latents  -> used by imf / sit / dit
└── <dataset>_images/              # raw images   -> used by jit (pixel space)
```

Precompute the VAE latents with the same Stable Diffusion VAE used by the
teachers — [`pcuenq/sd-vae-ft-mse-flax`](https://huggingface.co/pcuenq/sd-vae-ft-mse-flax),
selected by `dataset.vae: mse` in the configs and loaded by `utils/vae_util.py`.
The reference statistics in `stats/` were computed on these same processed sets,
so the images must be preprocessed the same way — the guided-diffusion style
center crop to 256x256 in `utils/input_pipeline.py` — for FID and FD-DINO to be
comparable to the paper.

### Reference Statistics

All ten reference statistic files ship with this release:

```
stats/
├── artbench-10_processed-fid_stats.npz        # InceptionV3 FID reference (5 files, ~32 MB each)
├── caltech-101-fid_stats.npz
├── cub-200-2011_processed-fid_stats.npz
├── food-101_processed-fid_stats.npz
├── stanford_cars_processed-fid_stats.npz
├── artbench-10-fd_dino-vitb14_stats.npz       # DINOv2 ViT-B/14 FDD reference (5 files, ~4.5 MB each)
├── caltech-101-fd_dino-vitb14_stats.npz
├── cub-200-2011-fd_dino-vitb14_stats.npz
├── food-101-fd_dino-vitb14_stats.npz
└── stanford-cars-fd_dino-vitb14_stats.npz
```

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

## Released Checkpoints

The four Stage 1 and four Stage 2 generators for **CUB-200-2011** are released
separately (fp32 generator weights only — optimizer and discriminator state are
stripped):

| stage | family | params | fp32 size |
| ----- | ------ | ------ | --------- |
| 1     | `dit`  | 674 M  | 2.70 GB   |
| 1     | `sit`  | 674 M  | 2.70 GB   |
| 1     | `jit`  | 952 M  | 3.81 GB   |
| 1     | `imf`  | 710 M  | 2.84 GB   |
| 2     | `dit`  | 674 M  | 2.70 GB   |
| 2     | `sit`  | 674 M  | 2.70 GB   |
| 2     | `jit`  | 952 M  | 3.81 GB   |
| 2     | `imf`  | 710 M  | 2.84 GB   |

<!-- TODO: add the Hugging Face download link once the export is uploaded. -->

Each is an Orbax checkpoint directory holding the EMA parameters, and restores
with `--config.load_from=/path/to/checkpoint_dir`. One exception: the Stage 1
`jit` run was trained without EMA, so its generator is the raw `params` tree.

## Reproducing the Paper

Every metric uses 10,000 samples. FID uses the InceptionV3 references in `stats/`
and FD-DINO uses the DINOv2 ViT-B/14 references in the same folder.

```bash
# Stage 1, all four families on CUB-200-2011
cd stage1_meanflow_transfer
for fam in imf sit dit jit; do
  WEIGHTS_DIR=/path/to/weights bash train.sh "$fam" cub-200-2011
done

# Stage 2, seeded from each Stage 1 best-FID checkpoint
cd ../stage2_continuous_adversarial_refinement
bash posttrain.sh dit /path/to/stage1/best_fid/checkpoint_XXXX cub-200-2011
```

Repeat with any of the other four domains. The teacher baselines reported
alongside `MF-T` come from the plain fine-tuning configs
(`plain_jit_finetune`, and the `sit.py` / `dit.py` wrappers) with the
`scripts/eval_best_fid_steps_plain_*.sh` drivers.

## Notes

- `scripts/` in both stages holds the best-checkpoint evaluation drivers used for
  the paper tables. They were written for a specific multi-GPU host and use
  absolute paths and `screen`; treat them as a record of what was run rather than
  portable entry points. The two launchers (`train.sh`, `posttrain.sh`) are the
  portable path.
- Both stages ship code for variants that are **not** part of the main results and
  are kept for the ablations and comparisons: the AFM endpoint-only adversary
  (`afm*.py`), the original instantaneous CAFM discriminator
  (`cafm_imf_discriminator.py`), the pMF perceptual variant (`pmf*.py`), and
  DogFit, which is off in every config.
- Trained weights other than the CUB-200 release above are not distributed. Only
  the pretrained teacher weights are needed to reproduce a run from scratch, and
  those come from their original public sources.

## Contact

For questions about the code, please open an
[issue](https://github.com/yaramohamadi/CAMFT/issues).

## Citation

The paper is currently under review. Please check back for the final reference.

```bibtex
@article{camft2027,
  title  = {Adaptation and Acceleration of Diffusion and Flow Models via
            MeanFlow Transfer and Continuous Adversarial Refinement},
  author = {Anonymous},
  note   = {Under review},
  year   = {2026}
}
```

## Acknowledgments

This work builds directly on several open-source releases, and this repository
adapts code from them:

- [DiT](https://github.com/facebookresearch/DiT) and
  [SiT](https://github.com/willisma/SiT) — the latent-space transformer backbone
  and two of the ImageNet-256 teacher checkpoints.
- [JiT](https://github.com/LTH14/JiT) — the pixel-space transformer, ported to
  Flax here as `models/jit.py`.
- MeanFlow and improved MeanFlow — the average-velocity objective that `MF-T`
  targets and the fourth teacher.
- [FLIP](https://github.com/facebookresearch/flip),
  [MAE](https://github.com/facebookresearch/mae),
  [t5x](https://github.com/google-research/t5x), and
  [Flax](https://github.com/google/flax) — JAX/Flax training-state, sharding,
  position-embedding, and layer utilities.
- [guided-diffusion](https://github.com/openai/guided-diffusion) and
  [glide-text2im](https://github.com/openai/glide-text2im) — the DDPM schedules
  and sampler, reimplemented in JAX as `utils/dit_diffusion.py`.
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) and
  [torchvision](https://github.com/pytorch/vision) — the InceptionV3 definition
  behind `utils/jax_fid/`.
- [diffusers](https://github.com/huggingface/diffusers) — the Flax VAE used to
  decode latents for previews.
- [DINOv2](https://github.com/facebookresearch/dinov2) — the FD-DINO feature
  extractor, reimplemented in pure JAX here as `utils/dinov2_jax.py`.
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — perceptual features
  for the auxiliary loss.

The teacher checkpoints and the datasets carry their own licenses; please follow
the terms of each original source.
