#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=wan_cache_W33
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Re-cache Wan latents for v5.1 chunk-wise training.
#
# Difference from the W=12 cache:
#   - window_length=33 pixel frames  (= 9 latent frames; Wan 4K+1 convention)
#   - deterministic non-overlapping starts at [0, 33, 66] per video
#   - 3 windows per video (yields 2 adjacent pairs + 1 distant chunk
#     for InfoNCE positive in v5.1's paired-chunk sampler).
#
# Output: /storage/scratch1/8/lwang831/cache/wan_10000vid_W33
# Expected: ~30000 blobs, T_lat=9 each. Wall ~3-6h on H200.
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
OUT_DIR="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
DATA_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
ANN_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}" "${OUT_DIR}"

cd "${WORKDIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-local}"
echo "OutDir:  ${OUT_DIR}"
echo "Start:   $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

nvidia-smi || true
python -V

python -u scripts/cache_wan_latents.py \
    --data_dir "${DATA_DIR}" \
    --annotation_dir "${ANN_DIR}" \
    --split train \
    --max_videos 10000 \
    --window_length 33 \
    --windows_per_video 3 \
    --frames_per_video 128 \
    --deterministic_starts "0,33,66" \
    --max_objects 8 \
    --image_size 128 \
    --seed 0 \
    --model_id "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
    --dtype float16 \
    --device cuda \
    --out_dir "${OUT_DIR}"

N=$(ls "${OUT_DIR}/latents" 2>/dev/null | wc -l)
echo "Wrote ${N} latent blobs."
touch "${OUT_DIR}/CACHE_DONE.marker"
echo ""; echo "Finished at $(date)"
