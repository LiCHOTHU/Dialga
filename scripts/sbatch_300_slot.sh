#!/bin/bash
#SBATCH --job-name=dlg_300_slot
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_slot_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_slot_%j.err

# Stage 1+2 of the slot pipeline on 300 CLEVRER videos.
# Outputs: stage1.pt, stage2.pt, scene videos, loss plots.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

GROUP=${GROUP:-scale_300_$(date +%m%d)}
RUN_NAME=${RUN_NAME:-slot_300_${GROUP}}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " Slot pipeline @ 300 videos"
echo "=========================================="
echo "  JobID            : ${SLURM_JOB_ID}"
echo "  Output dir       : ${OUTDIR}"
echo "  Wandb group      : ${GROUP}"
echo "  Wandb name       : ${RUN_NAME}"
echo "  GPU              : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start            : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=300 \
    training.windows_per_video=8 \
    training.batch_size=16 \
    training.num_workers=4 \
    training.stage1_epochs=40 \
    training.stage2_epochs=20 \
    training.ckpt_every=5 \
    training.noise_sigma=5e-3 \
    training.lambda_collision=1.0 \
    wandb.enabled=true \
    wandb.project=dialga \
    wandb.name="${RUN_NAME}" \
    wandb.group="${GROUP}" \
    wandb.job_type=train_slot \
    wandb.tags='[scale_300,slot,accel,shakedown]'

echo "[$(date)] Slot training done. Mirroring small artifacts to repo outputs/."
LOCAL_MIRROR="${REPO}/outputs/${RUN_NAME}"
mkdir -p "${LOCAL_MIRROR}"
cp -f "${OUTDIR}"/stage1_loss.png       "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/stage2_loss.png       "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/scene_video_*.gif     "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/train_slot.log        "${LOCAL_MIRROR}/" 2>/dev/null || true
echo "  -> ${LOCAL_MIRROR}/"
echo "[$(date)] Stage2 checkpoint at: ${OUTDIR}/stage2.pt"
