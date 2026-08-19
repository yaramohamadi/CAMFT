# Continuous Adversarial MeanFlow Transfer
Official code for *Continuous Adversarial MeanFlow Transfer*.

## Method

The pipeline has two stages, run in order:

1. **[Stage 1: MeanFlow-Transfer](#stage-1-meanflow-transfer-mf-t).** Fine-tune a
   pretrained ImageNet teacher into a MeanFlow student on the target domain. The
   student samples in a handful of step. First stage supports four teacher families, spanning all four prediction
parameterizations (mean velocity u, velocity v, noise eps, data x). 
2. **[Stage 2: Continuous Adversarial MeanFlow](#stage-2-continuous-adversarial-meanflow-camf).**
   Refine the trained stage 1 student with pure adversarial post-training. Stage 2 supports MeanFlow models even if they are not derived from stage 1.

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

  First stage starts from publicly released ImageNet-256 teacher checkpoints. Download
  each from its original source, put all four in one folder, and point `WEIGHTS_DIR` at
  it. Expected filenames:

  | family | filename              | download | size |
  | ------ | --------------------- | -------- | ---- |
  | `dit`  | `DiT-XL-2-256x256.pt` | [direct
  link](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) ([DiT
  repo](https://github.com/facebookresearch/DiT)) | 2.7 GB |
  | `sit`  | `SiT-XL-2-256.pt`     | [Dropbox](https://www.dl.dropboxusercontent.com/scl
  /fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) ([SiT
  repo](https://github.com/willisma/SiT)) | 2.7 GB |
  | `jit`  | `JiT-H-16-256.pth`    | [Dropbox folder](https://www.dropbox.com/scl/fo/3ke
  n1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=0) —
  take the `JiT-H/16` 256x256 checkpoint ([JiT repo](https://github.com/LTH14/JiT)) | —
  |
  | `imf`  | `iMF-XL-2-full`       |
  [`iMF-XL-2-full.zip`](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-XL-2-full.zip)
  ([iMF repo](https://github.com/Lyy-iiis/imeanflow)) | 10.5 GB |

# Pretrained teacher weights

The two stages start from publicly released ImageNet-256 teacher checkpoints.
This release does not bundle them. Download each one from its original source
and place it in a `weights/` directory (or point `WEIGHTS_DIR` at wherever you
keep them). The launchers expect these filenames:

| family | filename                  | source |
| ------ | ------------------------- | ------ |
| `dit`  | `DiT-XL-2-256x256.pt`     | DiT (Peebles and Xie), the official `DiT-XL/2` 256x256 checkpoint |
| `sit`  | `SiT-XL-2-256.pt`         | SiT (Ma et al.), the official `SiT-XL/2` 256x256 checkpoint |
| `jit`  | `JiT-H-16-256.pth`        | JiT (the "just image transformers" pixel-space model), `JiT-H/16` at 256 |
| `imf`  | `iMF-XL-2-full`           | improved MeanFlow `XL/2`, saved as a checkpoint directory |

Notes:

- `DiT-XL-2-256x256.pt` and `SiT-XL-2-256.pt` are single PyTorch state-dict
  files. The training code loads them on CPU with torch and converts the tensors
  to JAX arrays, so a CPU torch build is enough (see `requirements.txt`).
- `iMF-XL-2-full` is a directory rather than a single file. Point `--config.load_from`
  at the directory itself.
- Place all four under one folder and pass it once, for example:

  ```bash
  WEIGHTS_DIR=/data/meanflow_teachers bash train.sh dit cub-200-2011
  ```

If you keep the weights elsewhere per family, override `--config.load_from`
directly on the launcher command line.



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
