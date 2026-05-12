#!/bin/bash
#SBATCH --job-name=dlg_300_eventhead2
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eventhead2_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eventhead2_%j.err

# v2 — fix encoder convergence on 300 videos.
#
# v1 (job 8320227) finished cleanly but encoder state loss plateaued at 0.16
# (RMS≈1.5 world units). Phase-3 EventHead F1 = 0.054 in-dist / 0.066 held-out
# because the head's input positions were essentially random.
#
# Diagnosis: 300×8 slot-binding search space is much larger than 5×8, but we
# gave it the same 40 epochs and lambda_state=1.0. Below are the changes.
#
# Changes vs v1 (sbatch_300_eventhead.sh):
#   - lambda_state          1.0 -> 5.0    (force encoder to lock positions first)
#   - stage1_epochs         40  -> 60     (more iterations to converge)
#   - windows_per_video     32  -> 48     (~95% per-collision catch + better
#                                          per-frame coverage for encoder)
#   - stage1_lr             5e-4 -> 7e-4  (compensate for stronger lambda_state
#                                          dominating gradient magnitude)
#
# Walltime: estimated 60ep × 48/32 × 7.9 min = ~9.5h. Cutting to 7.5h target by
# raising LR; if it timeouts we still have a 60-ep ckpt to evaluate against
# the previous lambda_state=1 run.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

GROUP=${GROUP:-eventhead_300_v2_$(date +%m%d)}
RUN_NAME=${RUN_NAME:-slot_300_eventhead_${GROUP}}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 300-video Stage 1 + EventHead (v2)"
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
    training.windows_per_video=48 \
    training.batch_size=16 \
    training.num_workers=4 \
    training.stage1_epochs=60 \
    training.stage2_epochs=0 \
    training.stage1_lr=7e-4 \
    training.ckpt_every=5 \
    training.noise_sigma=5e-3 \
    training.lambda_state=5.0 \
    training.lambda_collision=1.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=50.0 \
    training.event_label_dilation=3 \
    wandb.enabled=true \
    wandb.project=dialga \
    wandb.name="${RUN_NAME}" \
    wandb.group="${GROUP}" \
    wandb.tags="[scale_300,eventhead,v3,v2]"

echo "[$(date)] Stage 1 + EventHead (v2) done."
echo "  Stage 1 ckpt @ ${OUTDIR}/stage1.pt"
