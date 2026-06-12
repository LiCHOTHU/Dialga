#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=render_e2
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Faithful reconstruction render for the e2 (encoder-unfrozen) checkpoint:
# loads the fine-tuned Wan VAE from the ckpt and re-encodes raw pixels through it.
# Output: 5 GT-vs-model mp4s to the local workspace outputs/ (5-clip rule).

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
VIDEO_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
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

CKPT="/storage/scratch1/8/lwang831/dialga_outputs/v512_e2_enc_unfrozen_20260524_194228/v5_best.pt"
OUT_DIR="${WORKDIR}/outputs/v512e2_recon_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_DIR}"

echo "Ckpt: ${CKPT}"
echo "Out:  ${OUT_DIR}"

python -u scripts/viz/save_v512_vae_recon.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --video_dir "${VIDEO_DIR}" \
    --out_dir "${OUT_DIR}" \
    --n_videos 5 --use_gt_gate

echo "Done at $(date)"
echo "Videos under: ${OUT_DIR}"
