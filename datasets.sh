#!/usr/bin/env bash
#
# Target-domain dataset table, shared by the Stage 1 and Stage 2 launchers.
#
# One entry per target domain reported in the paper. Each entry names the data
# directories and the FID / FD-DINO reference statistics that ship in stats/.
# The class counts below were read from the labels stored in the latent files
# themselves; both stages also set num_classes_from_data, so they re-derive the
# same number at run time and the value here only documents it (and is passed
# explicitly in Stage 2).
#
# Source this file, then call: ds_resolve <dataset>
# It sets DS_SLUG, DS_NUM_CLASSES, DS_LATENT_ROOT, DS_PIXEL_ROOT,
# DS_FID_REF, DS_FDD_REF.
#
# The FID and FD-DINO reference statistics for all five datasets ship in
# stats/. The image / latent data does not (size); see the top-level README.

ds_list() {
  echo "artbench-10 caltech-101 cub-200-2011 food-101 stanford-cars"
}

ds_resolve() {
  local ds="${1:-}"
  local release_root="${2:-}"

  if [[ -z "$ds" || -z "$release_root" ]]; then
    echo "ds_resolve: usage: ds_resolve <dataset> <release_root>" >&2
    return 2
  fi

  # Accept a few convenient aliases for the longer directory names.
  case "$ds" in
    artbench|artbench10|artbench-10)      ds=artbench-10 ;;
    caltech|caltech101|caltech-101)       ds=caltech-101 ;;
    cub|cub200|cub-200|cub-200-2011)      ds=cub-200-2011 ;;
    food|food101|food-101)                ds=food-101 ;;
    cars|stanford|stanford-cars|stanford_cars) ds=stanford-cars ;;
  esac

  # FID and FD-DINO stat filenames are not uniformly named: they were produced
  # by different runs and the names are kept exactly as the released files, so
  # each dataset spells out both explicitly rather than deriving them.
  case "$ds" in
    artbench-10)
      DS_NUM_CLASSES=10
      DS_FID_REF="$release_root/stats/artbench-10_processed-fid_stats.npz"
      DS_FDD_REF="$release_root/stats/artbench-10-fd_dino-vitb14_stats.npz"
      ;;
    caltech-101)
      DS_NUM_CLASSES=101
      DS_FID_REF="$release_root/stats/caltech-101-fid_stats.npz"
      DS_FDD_REF="$release_root/stats/caltech-101-fd_dino-vitb14_stats.npz"
      ;;
    cub-200-2011)
      DS_NUM_CLASSES=200
      DS_FID_REF="$release_root/stats/cub-200-2011_processed-fid_stats.npz"
      DS_FDD_REF="$release_root/stats/cub-200-2011-fd_dino-vitb14_stats.npz"
      ;;
    food-101)
      DS_NUM_CLASSES=101
      DS_FID_REF="$release_root/stats/food-101_processed-fid_stats.npz"
      DS_FDD_REF="$release_root/stats/food-101-fd_dino-vitb14_stats.npz"
      ;;
    stanford-cars)
      DS_NUM_CLASSES=196
      DS_FID_REF="$release_root/stats/stanford_cars_processed-fid_stats.npz"
      DS_FDD_REF="$release_root/stats/stanford-cars-fd_dino-vitb14_stats.npz"
      ;;
    *)
      echo "ERROR: unknown dataset '$1'. Known: $(ds_list)" >&2
      return 2
      ;;
  esac

  DS_SLUG="$ds"
  DS_LATENT_ROOT="$release_root/data/${ds}_processed_latents"
  DS_PIXEL_ROOT="$release_root/data/${ds}_images"

  for ref in "$DS_FID_REF" "$DS_FDD_REF"; do
    [[ -f "$ref" ]] || {
      echo "ERROR: reference statistics not found: $ref" >&2
      return 3
    }
  done

  export DS_SLUG DS_NUM_CLASSES DS_LATENT_ROOT DS_PIXEL_ROOT DS_FID_REF DS_FDD_REF
}
