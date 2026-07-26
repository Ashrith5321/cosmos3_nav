#!/usr/bin/env bash
# Download HM3D train scene meshes + semantic annotations.
#
# Needs Matterport credentials from the HM3D EULA form:
#   https://matterport.com/habitat-matterport-3d-research-dataset
#
# Usage (credentials via environment, so they stay out of your shell history):
#
#   read -rp  'Matterport username: ' MATTERPORT_USERNAME
#   read -rsp 'Matterport password: ' MATTERPORT_PASSWORD; echo
#   export MATTERPORT_USERNAME MATTERPORT_PASSWORD
#   bash scripts/download_hm3d_train.sh
#
# Roughly 50 GB for train (~800 scenes), extrapolating from val's 6.0 GB for
# 100 scenes. Check free space before starting; the tarball is staged before
# extraction, so peak usage is briefly higher than the final size.

set -euo pipefail

PYTHON="${PYTHON:-/home/ashed/miniconda3/envs/habitat033/bin/python}"
DATA_ROOT="${DATA_ROOT:-/home/ashed/Documents/spatial_training/data}"
TARGET="${DATA_ROOT}/scene_datasets/hm3d_v0.2"

if [[ -z "${MATTERPORT_USERNAME:-}" || -z "${MATTERPORT_PASSWORD:-}" ]]; then
  echo "ERROR: set MATTERPORT_USERNAME and MATTERPORT_PASSWORD first." >&2
  echo "       Credentials come from the HM3D EULA form; see the header." >&2
  exit 1
fi

echo "python    : ${PYTHON}"
echo "data root : ${DATA_ROOT}"
echo "target    : ${TARGET}"
echo "free space: $(df -h "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
echo

# hm3d_train_habitat_v0.2        the .basis.glb meshes and navmeshes
# hm3d_train_semantic_annots_v0.2 the .semantic.glb / .semantic.txt annotations
# hm3d_train_configs_v0.2        the scene_dataset_config for the split
for UID in hm3d_train_habitat_v0.2 hm3d_train_semantic_annots_v0.2 hm3d_train_configs_v0.2; do
  echo "=== ${UID} ==="
  "${PYTHON}" -m habitat_sim.utils.datasets_download \
    --username "${MATTERPORT_USERNAME}" \
    --password "${MATTERPORT_PASSWORD}" \
    --uids "${UID}" \
    --data-path "${DATA_ROOT}"
done

# The downloader writes to scene_datasets/hm3d-0.2/hm3d/train and links
# scene_datasets/hm3d. This repo's config points at scene_datasets/hm3d_v0.2,
# whose annotated scene_dataset_config already lists the train scenes -- the
# meshes just have to be reachable under that directory.
DOWNLOADED_TRAIN=""
for CANDIDATE in \
  "${DATA_ROOT}/scene_datasets/hm3d-0.2/hm3d/train" \
  "${DATA_ROOT}/scene_datasets/hm3d/train" \
  "${DATA_ROOT}/scene_datasets/hm3d_v0.2/train"; do
  if [[ -d "${CANDIDATE}" ]]; then
    DOWNLOADED_TRAIN="${CANDIDATE}"
    break
  fi
done

if [[ -z "${DOWNLOADED_TRAIN}" ]]; then
  echo "WARNING: could not find an extracted train/ directory. Inspect" >&2
  echo "         ${DATA_ROOT}/scene_datasets and link it to ${TARGET}/train." >&2
  exit 1
fi

echo
echo "extracted train scenes: ${DOWNLOADED_TRAIN}"
if [[ ! -e "${TARGET}/train" ]]; then
  ln -s "${DOWNLOADED_TRAIN}" "${TARGET}/train"
  echo "linked ${TARGET}/train -> ${DOWNLOADED_TRAIN}"
fi

echo
echo "train scene dirs: $(ls "${TARGET}/train" | wc -l)"
echo "size            : $(du -sh "${DOWNLOADED_TRAIN}" | cut -f1)"
echo
echo "Verify a train scene loads with semantics:"
echo "  python scripts/run_episode.py --override data.split=train --override episode.max_steps=20"
