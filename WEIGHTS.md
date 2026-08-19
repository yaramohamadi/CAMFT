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
  WEIGHTS_DIR=/data/meanflow_teachers bash train_cub200.sh dit
  ```

If you keep the weights elsewhere per family, override `--config.load_from`
directly on the launcher command line.
