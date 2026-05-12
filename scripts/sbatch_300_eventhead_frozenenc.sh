#!/bin/bash
#SBATCH --job-name=dlg_eh_frozen2
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_eh_frozen_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_eh_frozen_%j.err

# Train EventHead only, on top of frozen encoder from 8317648 (which converged
# cleanly: state loss 0.21 → 0.006, RMS ≈ 0.22 world).
#
# Companion to v2 joint-retrain (8326612). This isolates the event-head
# question from any encoder-convergence noise: if F1 here is high, the head
# architecture works at scale; if F1 is still low, the head itself is the
# bottleneck regardless of encoder quality.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

SRC_CKPT=${SRC_CKPT:-${OUTBASE}/slot_300_scale_300_0509b/stage2.pt}
RUN_NAME=${RUN_NAME:-eventhead_frozen_300_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " EventHead-only train, frozen encoder"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Src ckpt : ${SRC_CKPT}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_event_head_only.py \
    --src_ckpt "${SRC_CKPT}" \
    --output_dir "${OUTDIR}" \
    --max_videos 300 \
    --windows_per_video 32 \
    --epochs 40 \
    --batch_size 16 \
    --num_workers 4 \
    --lr 1e-3 \
    --pos_weight 50.0 \
    --label_dilation 3 \
    --log_every 100 \
    --ckpt_every 5 \
    --use_diff_features

echo "[$(date)] Training done."
echo "  ckpt @ ${OUTDIR}/event_head_only.pt"

# Phase-3 inference test, in-distribution + held-out
echo "[$(date)] Running Phase-3 inference test (in-distribution)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/event_head_only.pt" \
    --max_videos 30 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -45

echo ""
echo "[$(date)] Running Phase-3 inference test (held-out)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/event_head_only.pt" \
    --max_videos 30 \
    --seed 1 \
    --output "${OUTDIR}/phase3_heldout.json" 2>&1 | tail -45

echo "[$(date)] All done."
