# MeanFlow-Transfer and Continuous Adversarial MeanFlow

Code for *Adaptation and Acceleration of Diffusion and Flow Models via MeanFlow
Transfer and Continuous Adversarial Refinement*.

The method turns a pretrained ImageNet diffusion or flow model into a few-step
generator on a new target domain, in two stages:

1. **MeanFlow-Transfer (Stage 1).** Fine-tune the pretrained teacher into a
   MeanFlow student that produces samples in a handful of steps and inherits the
   teacher's classifier-free guidance behaviour.
2. **Continuous Adversarial MeanFlow (Stage 2).** Refine the Stage 1 student
   with pure adversarial post-training. The generator keeps its guidance
   behaviour; a discriminator scores the guided few-step endpoint, sharpening the
   one and two step samples.

Both stages support four teacher families, spanning all four prediction
parameterizations:

| family | teacher                | predicts        | Stage 1 space | Stage 2 space |
| ------ | ---------------------- | --------------- | ------------- | ------------- |
| `imf`  | improved MeanFlow XL/2 | mean velocity u | latent        | latent        |
| `sit`  | SiT-XL/2               | velocity v      | latent        | latent        |
| `dit`  | DiT-XL/2               | noise eps       | latent        | latent        |
| `jit`  | JiT-H/16               | input x         | pixel         | pixel         |

Five target domains are supported: ArtBench-10, Caltech-101, CUB-200-2011,
Food-101, and Stanford Cars.

## Layout

```
stage1_meanflow_transfer/       Stage 1 code, configs, and launcher
stage2_continuous_adversarial_refinement/
                                Stage 2 code, configs, and launcher
stats/                          FID and FD-DINO reference statistics (all 5 domains)
datasets.sh                     Target-domain table shared by both launchers
requirements.txt                Pinned Python environment (JAX CUDA 12)
WEIGHTS.md                      Where to download the pretrained teacher weights
```

The two stages are kept as separate packages on purpose. They share most of
their code, but a few modules (`imf.py`, `configs/default.py`,
`utils/sample_util.py`, `models/imfDiT.py`) carry stage-specific changes that
cannot be collapsed into one shared copy without breaking one of the stages.
Each package is therefore self-contained and runs on its own.

## Setup

The reference statistics in `stats/` are stored with [Git LFS](https://git-lfs.com).
Install it before cloning, or run `git lfs pull` in an existing clone:

```bash
git lfs install
git clone https://github.com/yaramohamadi/CAMFT.git
```

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Download the pretrained teacher weights following `WEIGHTS.md` and place them in
a `weights/` directory (or point `WEIGHTS_DIR` at wherever you keep them).

### Data

The image and latent data is not bundled (size). Build a `data/` directory with
one subfolder per domain you want to run:

```
data/<dataset>_processed_latents/   VAE latents (latent families: imf/sit/dit)
data/<dataset>_images/             raw images (jit, pixel space)
```

where `<dataset>` is one of `artbench-10`, `caltech-101`, `cub-200-2011`,
`food-101`, `stanford-cars`. Download each dataset from its official source for
the pixel images, and precompute the VAE latents with the same Stable Diffusion
VAE used by the teachers. The reference statistics in `stats/` were computed on
these same processed sets, so the images must be preprocessed the same way for
FID and FD-DINO to be comparable to the paper.

## Running

Stage 1, then Stage 2. The examples use `dit` on `cub-200-2011`; swap in `imf`,
`sit`, or `jit` and any of the five domains.

```bash
# Stage 1: MeanFlow-Transfer
cd stage1_meanflow_transfer
WEIGHTS_DIR=/path/to/weights bash train.sh dit cub-200-2011
# best-FID checkpoint lands under runs/cub-200-2011_dit_<stamp>/best_fid

# Stage 2: adversarial refinement, seeded from the Stage 1 checkpoint
cd ../stage2_continuous_adversarial_refinement
bash posttrain.sh dit /path/to/stage1/best_fid/checkpoint_XXXX cub-200-2011
```

Both launchers default to `cub-200-2011` when the dataset is omitted. Extra
flags are forwarded to the training script:

```bash
bash train.sh dit food-101 --config.training.num_steps=1000
```

Use the same dataset for Stage 2 that the Stage 1 checkpoint was trained on: the
class count is passed to the model and a mismatch will not load.

## Evaluation

Every metric uses 10,000 samples. FID uses the InceptionV3 reference statistics
in `stats/` and FD-DINO uses the DINOv2 ViT-B/14 reference in the same folder.
Class counts are read from the data itself (`num_classes_from_data`), so each
domain resolves to its own count: ArtBench-10 10, Caltech-101 101, CUB-200 200,
Food-101 101, Stanford Cars 196.

Sampling guidance (`omega`) is set per family inside each launcher: DiT 1.5,
SiT 1.5, JiT 2.2, iMF 7.5. The iMF evaluation additionally restricts the EMA
time window to `[0.4, 0.65]`, and the DiT Stage 1 run clips gradients at global
norm 1.0.

The launchers evaluate at the sampling-step counts reported in the paper: Stage 1
reports few-step and many-step FID, Stage 2 reports NFE 1 and 2.

## Notes

- `scripts/` in each stage holds the best-checkpoint evaluation drivers used for
  the paper tables. They were written for a specific multi-GPU host and use
  absolute paths and `screen`; treat them as a record of what was run rather
  than portable entry points.
- The config files are named after the dataset they were first written for
  (`caltech_*`, `plain_*`). They are used for every target domain, with the
  dataset-specific paths and reference statistics overridden on the command line
  by the launchers.
- Trained model weights are not in this repository. Only the pretrained teacher
  weights are needed to reproduce a run, and those come from their original
  public sources (see `WEIGHTS.md`).
