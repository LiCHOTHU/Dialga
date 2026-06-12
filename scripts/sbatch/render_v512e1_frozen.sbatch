#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=render_e1
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Render 5 reconstruction videos from the v512_e1_frozen run (last night's
# ckpt that survived disk-quota crash).

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

CKPT="${WORKDIR}/outputs/v512_e1_frozen_20260522_221139/v5_best.pt"
OUT_DIR="${WORKDIR}/outputs/v512e1_recon_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

echo "Ckpt: ${CKPT}"
echo "Out:  ${OUT_DIR}"

python -u scripts/viz/save_v51_overfit_videos.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --n_videos 5 --use_gt_gate

echo "Done at $(date)"
echo "Videos under: ${OUT_DIR}"
