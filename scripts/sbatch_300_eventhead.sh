#!/bin/bash
#SBATCH --job-name=dlg_300_eventhead
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eventhead_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eventhead_%j.err

# 300-video Stage 1 + EventHead supervision (v3 design).
#
# Coverage knob: with windows_per_video=8 (production default) only ~32% of
# collision frames are seen during training. We bump to 32 for ~80% catch
# rate so the EventHead has enough positive examples to learn.
#
# Stage 2 disabled — EventHead inference test only needs encoder + head.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

GROUP=${GROUP:-eventhead_300_$(date +%m%d)}
RUN_NAME=${RUN_NAME:-slot_300_eventhead_${GROUP}}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 300-video Stage 1 + EventHead"
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
    training.windows_per_video=32 \
    training.batch_size=16 \
    training.num_workers=4 \
    training.stage1_epochs=40 \
    training.stage2_epochs=0 \
    training.ckpt_every=5 \
    training.noise_sigma=5e-3 \
    training.lambda_collision=1.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=50.0 \
    training.event_label_dilation=3 \
    wandb.enabled=true \
    wandb.project=dialga \
    wandb.name="${RUN_NAME}" \
    wandb.group="${GROUP}" \
    wandb.tags="[scale_300,eventhead,v3]"

echo "[$(date)] Stage 1 + EventHead done."
echo "  Stage 1 ckpt @ ${OUTDIR}/stage1.pt"
