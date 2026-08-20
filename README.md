# Continuous Adversarial MeanFlow Transfer
This is the official JAX implementation for the paper *Continuous Adversarial MeanFlow Transfer*. This code is written and tested on GPUs.

**[Project page](https://yasaman-dt.github.io/CAMFT/)**

The pipeline has two stages, run in order:

1. **[Stage 1: MeanFlow-Transfer](stage1_meanflow_transfer).** Fine-tune a
   pretrained ImageNet teacher into a MeanFlow student on the target domain. The
   student samples in a handful of step. First stage supports four teacher families, spanning all four prediction
parameterizations (mean velocity u, velocity v, noise eps, data x). 
2. **[Stage 2: Continuous Adversarial MeanFlow](stage2_continuous_adversarial_refinement).**
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
  | `dit`  | `DiT-XL-2-256x256.pt` | [direct link](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt) ([DiT repo](https://github.com/facebookresearch/DiT)) | 2.7 GB |
  | `sit`  | `SiT-XL-2-256.pt`     | [Dropbox](https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0) ([SiT repo](https://github.com/willisma/SiT)) | 2.7 GB |
  | `jit`  | `JiT-H-16-256.pth`    | [Dropbox folder](https://www.dropbox.com/scl/fo/3ken1avtsd81ip67b9qpi/AK218ZNvXKSv74igVvht4PQ?rlkey=14gjrblmljewpl6ygxzlr3njm&dl=0) ([JiT repo](https://github.com/LTH14/JiT)) | 11.4 GB |
  | `imf`  | `iMF-XL-2-full`  (directory)     | [HuggingFace](https://huggingface.co/Lyy0725/iMF/blob/main/iMF-XL-2-full.zip) ([iMF repo](https://github.com/Lyy-iiis/imeanflow)) | 10.5 GB |

Notes:

- `DiT-XL-2-256x256.pt` and `SiT-XL-2-256.pt` are single PyTorch state-dict
  files. The training code loads them on CPU with torch and converts the tensors
  to JAX arrays.
- `iMF-XL-2-full` is a directory rather than a single file. 


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

| stage | family | checkpoint | params  | step    | fp32 size |
| ----- | ------ | ---------- | ------- | ------- | --------- |
| 1     | `dit`  | [`stage1_dit`][s1-dit] | 674.2 M | 27,500  | 2.51 GB |
| 1     | `sit`  | [`stage1_sit`][s1-sit] | 674.2 M | 20,000  | 2.51 GB |
| 1     | `jit`  | [`stage1_jit`][s1-jit] | 951.8 M | 27,500  | 3.53 GB |
| 1     | `imf`  | [`stage1_imf`][s1-imf] | 710.3 M | 35,000  | 2.64 GB |
| 2     | `dit`  | [`stage2_dit`][s2-dit] | 674.2 M | 115,000 | 2.51 GB |
| 2     | `sit`  | [`stage2_sit`][s2-sit] | 674.2 M | 70,000  | 2.51 GB |
| 2     | `jit`  | [`stage2_jit`][s2-jit] | 951.8 M | 60,000  | 3.53 GB |
| 2     | `imf`  | [`stage2_imf`][s2-imf] | 710.3 M | 65,000  | 2.64 GB |

[s1-dit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage1_dit
[s1-sit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage1_sit
[s1-jit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage1_jit
[s1-imf]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage1_imf
[s2-dit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage2_dit
[s2-sit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage2_sit
[s2-jit]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage2_jit
[s2-imf]: https://huggingface.co/elbahramino/CAMFT-CUB200/tree/main/stage2_imf

<!-- TODO: add the Hugging Face download link once the export is uploaded. -->

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

## Contact

For questions about the code, please open an
[issue](https://github.com/yaramohamadi/CAMFT/issues).

## Citation

The paper is currently under review. Stay tuned for updated citation.

```bibtex
@article{camft2027,
  title  = {Adaptation and Acceleration of Diffusion and Flow Models via
            MeanFlow Transfer and Continuous Adversarial Refinement},
  author = {Yara Bahram, Zahra Dehghani, Mélodie Desbos, Eric Granger, Pablo Piantanida, Mohammadhadi Shateri},
  note   = {Under review},
  year   = {2027}
}
```

## Acknowledgments

This work builds directly on open-source releases of DiT, SiT, JiT, and iMF, and Adversarial-Flow-Models. We thank the authors of these projects for their codes.
