#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="${BASE_DIR:-/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER}"
TRAIN_DIR="${TRAIN_DIR:-${BASE_DIR}/train_video}"
ARCHIVE_URL="http://data.csail.mit.edu/clevrer/videos/train/video_train.zip"
ARCHIVE_PATH="${TRAIN_DIR}/video_train.zip"

if ! command -v wget >/dev/null 2>&1; then
  echo "[error] wget is required to download CLEVRER." >&2
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[error] unzip is required to extract CLEVRER." >&2
  exit 1
fi

echo "[info] Preparing CLEVRER dataset directory at ${TRAIN_DIR}"
mkdir -p "${TRAIN_DIR}"

echo "[info] Downloading CLEVRER training videos from ${ARCHIVE_URL}"
wget -c --show-progress -P "${TRAIN_DIR}" "${ARCHIVE_URL}"

echo "[info] Extracting ${ARCHIVE_PATH}"
unzip -q "${ARCHIVE_PATH}" -d "${TRAIN_DIR}"

echo "[info] Removing ${ARCHIVE_PATH}"
rm -f "${ARCHIVE_PATH}"

echo "[done] CLEVRER training videos are available in ${TRAIN_DIR}"
