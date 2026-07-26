#!/usr/bin/env bash
# Download HM3D train scene meshes + semantic annotations.
#
# Credentials are a Matterport API token (Token ID / Token Secret) from
# https://matterport.com/habitat-matterport-3d-research-dataset
#
#   read -rp  'Matterport Token ID: '     MP_TOKEN_ID
#   read -rsp 'Matterport Token Secret: ' MP_TOKEN_SECRET; echo
#   export MP_TOKEN_ID MP_TOKEN_SECRET
#   bash scripts/download_hm3d_train.sh
#
# NOTE: habitat_sim.utils.datasets_download does NOT work with these tokens.
# The Matterport endpoint answers with a 307 to a presigned S3 URL, and the
# habitat downloader writes the resulting "Unauthorized" JSON body into a file
# named *.tar, which then fails to untar with a confusing ReadError. curl with
# -L handles the redirect correctly, so this script fetches directly.
#
# Sizes: meshes 27.2 GB, semantics 8.1 GB. Downloads resume with curl -C -.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/ashed/Documents/spatial_training/data}"
STAGING="${DATA_ROOT}/_hm3d_download"
TARGET="${DATA_ROOT}/scene_datasets/hm3d_v0.2"
BASE="https://api.matterport.com/resources/habitat"

PACKAGES=(
  hm3d-train-habitat-v0.2.tar          # .basis.glb meshes and navmeshes
  hm3d-train-semantic-annots-v0.2.tar  # .semantic.glb / .semantic.txt
  hm3d-train-semantic-configs-v0.2.tar
  hm3d-train-configs.tar
)

if [[ -z "${MP_TOKEN_ID:-}" || -z "${MP_TOKEN_SECRET:-}" ]]; then
  echo "ERROR: set MP_TOKEN_ID and MP_TOKEN_SECRET first (see header)." >&2
  exit 1
fi

mkdir -p "${STAGING}"
echo "staging   : ${STAGING}"
echo "target    : ${TARGET}"
echo "free space: $(df -h "${DATA_ROOT}" | awk 'NR==2 {print $4}')"
echo

for PKG in "${PACKAGES[@]}"; do
  echo "=== ${PKG} ==="
  curl -sSL -u "${MP_TOKEN_ID}:${MP_TOKEN_SECRET}" -C - \
    -o "${STAGING}/${PKG}" \
    -w "  HTTP %{http_code}  %{size_download} bytes  %{time_total}s\n" \
    "${BASE}/${PKG}"

  # An auth failure arrives as a short JSON body with a .tar name, so check the
  # archive is real before trusting it.
  if ! tar tf "${STAGING}/${PKG}" >/dev/null 2>&1; then
    echo "ERROR: ${PKG} is not a valid tar. First bytes:" >&2
    head -c 200 "${STAGING}/${PKG}" >&2
    echo >&2
    exit 1
  fi
done

echo
echo "extracting..."
for PKG in "${PACKAGES[@]}"; do
  tar xf "${STAGING}/${PKG}" -C "${STAGING}"
done

# The archives unpack to hm3d-<version>/hm3d/train. This repo reads
# scene_datasets/hm3d_v0.2, whose annotated scene_dataset_config already lists
# the train scenes -- they just have to be reachable under that directory.
EXTRACTED="$(find "${STAGING}" -maxdepth 4 -type d -name train | head -1)"
if [[ -z "${EXTRACTED}" ]]; then
  echo "ERROR: no extracted train/ directory found under ${STAGING}" >&2
  exit 1
fi

echo "extracted train scenes: ${EXTRACTED}"
if [[ ! -e "${TARGET}/train" ]]; then
  ln -s "${EXTRACTED}" "${TARGET}/train"
  echo "linked ${TARGET}/train -> ${EXTRACTED}"
fi

echo
echo "train scene dirs: $(ls "${TARGET}/train" | wc -l)"
echo "size            : $(du -sh "${EXTRACTED}" | cut -f1)"
echo
echo "Next:"
echo "  python scripts/build_manifests.py --name full --override data.split=train"
echo "  rm -rf ${STAGING}   # once the manifests build cleanly"
