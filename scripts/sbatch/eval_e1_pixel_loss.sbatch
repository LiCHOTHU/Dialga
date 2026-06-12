#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200,gpu-l40s
#SBATCH --time=00:40:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=e1_pixloss
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Compute the trainer's L_pixel (obs) for the OLD latent-recon e1 checkpoint,
# which was trained WITHOUT pixel supervision (pixel=0.0 logged as placeholder).
# Gives a genuine pixel-loss baseline to compare against e2/e3's reported value.
# No-grad decode through the frozen Wan VAE => fits 16GB V100.

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
VIDEO_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"; export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

CKPT="/storage/scratch1/8/lwang831/dialga_outputs/v512_e1_frozen_20260526_001020/v5_best.pt"
echo "Eval ckpt: ${CKPT}"; nvidia-smi || true

python -u scripts/eval_pixel_loss.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --video_dir "${VIDEO_DIR}" \
    --max_videos 1000 \
    --split val --val_frac 0.2 --seed 42 \
    --n_chunks 200 --batch_size 4 --num_workers 2 \
    --dtype bfloat16

echo "Done at $(date)"
