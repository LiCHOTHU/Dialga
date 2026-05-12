#!/bin/bash
#SBATCH --job-name=dlg_5_overfit_s1
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_overfit_s1_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_overfit_s1_%j.err

# 5-video overfit Stage 1 — small, fast. Stage 2 disabled (we only need encoder
# for Phase-2 event-extractor evaluation).

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_5_overfit_full_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 5-video overfit Stage 1"
echo "=========================================="
echo "  JobID            : ${SLURM_JOB_ID}"
echo "  Output dir       : ${OUTDIR}"
echo "  GPU              : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start            : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.windows_per_video=123 \
    training.batch_size=16 \
    training.num_workers=4 \
    training.stage1_epochs=100 \
    training.stage2_epochs=0 \
    training.ckpt_every=20 \
    training.noise_sigma=0.0 \
    training.lambda_collision=1.0 \
    wandb.enabled=false

echo "[$(date)] Stage 1 done."
echo "  Stage 1 ckpt @ ${OUTDIR}/stage1.pt"
