#!/bin/bash
#SBATCH --job-name=dlg_20_eh_qva
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_20_eh_qva_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_20_eh_qva_%j.err

# 20-video overfit test of the NEW event-head architecture (event_input_mode=qva).
#
# Goal: prove the architecture can learn collision detection from
# (q, v, a, z_static) when given enough capacity (overfit). If F1 < 0.7 in
# this regime, the head architecture itself is the problem — not the
# encoder convergence or data scale.
#
# Settings:
#   - 20 videos, full coverage (windows_per_video=64 ≈ all frames seen 3×/epoch)
#   - 80 epochs, lambda_state=1.0, lambda_event=1.0
#   - event_input_mode=qva (new): feeds [q, v, a] to the head
#   - noise_sigma=0 so the model can fit exactly (no augmentation)

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_20_overfit_eh_qva_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 20-video overfit Stage 1 + EventHead (qva)"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=20 \
    training.windows_per_video=64 \
    training.batch_size=16 \
    training.num_workers=4 \
    training.stage1_epochs=80 \
    training.stage2_epochs=0 \
    training.ckpt_every=10 \
    training.noise_sigma=0.0 \
    training.lambda_state=1.0 \
    training.lambda_collision=1.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=50.0 \
    training.event_label_dilation=3 \
    model.event_input_mode=qva \
    wandb.enabled=false

echo "[$(date)] Training done."

# Phase-3 in-distribution (the same 20 vids we trained on)
echo "[$(date)] Running Phase-3 inference test (in-distribution)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 20 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -40

echo "[$(date)] Done."
