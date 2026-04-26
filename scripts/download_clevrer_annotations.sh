#!/usr/bin/env bash
# Download CLEVRER annotation json files (needed for Experiments 1 and 4).
# These are ~50 MB total and unzip to per-scene json files with 3D trajectories.
set -euo pipefail

ANN_DIR="${ANN_DIR:-/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations}"

mkdir -p "${ANN_DIR}/train"

echo "[info] Downloading CLEVRER train annotations..."
wget -c --show-progress \
  "http://data.csail.mit.edu/clevrer/annotations/annotation_train.zip" \
  -P "${ANN_DIR}"

echo "[info] Extracting..."
unzip -q "${ANN_DIR}/annotation_train.zip" -d "${ANN_DIR}/train"
rm -f "${ANN_DIR}/annotation_train.zip"

echo "[done] Annotations available at ${ANN_DIR}/train"
echo "       $(find "${ANN_DIR}/train" -name "*.json" | wc -l) json files"
