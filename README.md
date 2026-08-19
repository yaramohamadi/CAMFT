# MeanFlow-Transfer

Code and CUB-200-2011 data for *Adaptation and Acceleration of Diffusion and
Flow Models via MeanFlow Transfer and Continuous Adversarial Refinement*.

The method turns a pretrained ImageNet diffusion or flow model into a few-step
generator in two stages:

1. **MeanFlow-Transfer (Stage 1).** Fine-tune the pretrained teacher into a
   MeanFlow student that produces samples in a handful of steps and inherits the
   teacher's classifier-free guidance behaviour.
2. **Continuous Adversarial MeanFlow (Stage 2).** Refine the Stage 1 student
   with pure adversarial post-training. The generator keeps its guidance
   behaviour; a discriminator scores the guided few-step endpoint, sharpening the
   one and two step samples.

Both stages support four backbone families:

| family | teacher                | Stage 1 space | Stage 2 space |
| ------ | ---------------------- | ------------- | ------------- |
| `imf`  | improved MeanFlow XL/2 | latent        | latent        |
| `sit`  | SiT-XL/2               | latent        | latent        |
| `dit`  | DiT-XL/2               | latent        | latent        |
| `jit`  | JiT-H/16               | pixel         | pixel         |

## Layout

```
stage1_meanflow_transfer/    Stage 1 code, configs, and the CUB launcher
stage2_adversarial_refinement/  Stage 2 code, configs, and the CUB launcher
stats/                       FID and FD-DINO reference statistics for CUB-200
requirements.txt             Pinned Python environment (JAX CUDA 12)
WEIGHTS.md                   Where to download the pretrained teacher weights
```

The CUB-200-2011 data is not bundled in this archive (submission size limit).
Prepare a `data/` directory yourself as described in Setup below.

The two stages are kept as separate packages on purpose. They share most of
their code, but a few modules (`imf.py`, `configs/default.py`,
`utils/sample_util.py`, `models/imfDiT.py`) carry stage-specific changes that
cannot be collapsed into one shared copy without breaking one of the stages.
Each package is therefore self-contained and runs on its own.

## Setup

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Prepare the CUB-200 data once. This archive omits it to stay under the
submission size limit, so build a `data/` directory with two subfolders:

```
data/cub-200-2011_processed_latents/   VAE latents (latent families: imf/sit/dit)
data/cub-200-2011_images/              raw images (JiT, pixel space)
```

Download CUB-200-2011 from its official source
(https://www.vision.caltech.edu/datasets/cub_200_2011/) for the pixel images,
and precompute the VAE latents with the same Stable Diffusion VAE used by the
teachers. The FID and FD-DINO reference statistics in `stats/` were computed on
this same processed set.

Download the pretrained teacher weights following `WEIGHTS.md` and place them in
a `weights/` directory (or point `WEIGHTS_DIR` at wherever you keep them).

## Reproducing CUB-200

Stage 1, then Stage 2, for any family. The examples use `dit`; swap in `imf`,
`sit`, or `jit` to run the others.

```bash
# Stage 1: MeanFlow-Transfer
cd stage1_meanflow_transfer
WEIGHTS_DIR=/path/to/weights bash train_cub200.sh dit
# best-FID checkpoint lands under runs/cub200_dit_<stamp>/best_fid

# Stage 2: adversarial refinement, seeded from the Stage 1 checkpoint
cd ../stage2_adversarial_refinement
bash posttrain_cub200.sh dit /path/to/stage1/best_fid/checkpoint_XXXX
```

Every metric uses 10,000 samples. FID uses the reference statistics in `stats/`
and FD-DINO uses the DINOv2 ViT-B/14 reference in the same folder. Sampling
guidance (`omega`) is set per family inside each launcher: DiT 1.5, SiT 7.5,
JiT 2.2, iMF 7.5 (the iMF evaluation additionally restricts the EMA time window
to `[0.4, 0.65]`).

## Notes

- The launchers evaluate at the sampling-step counts reported in the paper.
  Stage 1 reports few-step and many-step FID; Stage 2 reports NFE 1 and 2.
- Trained model weights are not included. Only the pretrained teacher weights are
  needed to reproduce a run, and those come from their original public sources
  (see `WEIGHTS.md`).
