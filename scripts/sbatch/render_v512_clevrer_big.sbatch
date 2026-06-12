#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200,gpu-l40s
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=render_v512cb
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Render 5 GT-vs-model mp4s from the v512_clevrer_big (Option A) checkpoint.
# Frozen Wan VAE => save_v51_overfit_videos.py is the right script.

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

CKPT="${CKPT:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/v512_clevrer_big/last.pt}"
OUT_DIR="${OUT_DIR:-/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/v512_clevrer_big/rollouts_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${OUT_DIR}"

echo "Ckpt: ${CKPT}"
echo "Out:  ${OUT_DIR}"

python -u scripts/viz/save_v51_overfit_videos.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --n_videos 5 --use_gt_gate --dtype bfloat16

echo "Done at $(date)"
echo "Videos under: ${OUT_DIR}"
