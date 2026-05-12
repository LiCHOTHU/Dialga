#!/bin/bash
#SBATCH --job-name=dlg_5_gl
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_gl_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_gl_%j.err

# 5-video overfit, GENERAL representation + curriculum, LONGER run.
# Same hyperparams as the 3-min quick test (8336035, F1=0.085 in 15 ep)
# but with:
#   - 16 windows/video (up from 8) — more data per epoch
#   - 40 epochs (up from 15)       — encoder has time to converge
#   - 10-ep curriculum anneal      — longer strong-lambda phase
# Target: F1 climbs from 0.085 toward the 0.78–1.00 range we see with
# fully-supervised baselines.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_5_gen_long_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 5-vid overfit (general + curriculum, longer)"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "  Supervision:"
echo "    lambda_state          1.0 → 0.1  (anneal over 10 epochs)"
echo "    lambda_recon_stage1   1.0"
echo "    lambda_contrastive    0.5"
echo "    event_supervision     self"
echo "    event_input_mode      qva"
echo "    windows/video         16"
echo "    epochs                40"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.windows_per_video=16 \
    training.batch_size=8 \
    training.num_workers=2 \
    training.stage1_epochs=40 \
    training.stage2_epochs=0 \
    training.ckpt_every=40 \
    training.noise_sigma=0.0 \
    training.lambda_state=1.0 \
    training.lambda_state_anneal_to=0.1 \
    training.lambda_state_anneal_epochs=10 \
    training.lambda_solver=1.0 \
    training.lambda_static=0.1 \
    training.lambda_collision=0.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=50.0 \
    training.event_label_dilation=3 \
    training.event_supervision=self \
    training.self_event_z_thresh=1.5 \
    training.self_event_sharpness=2.5 \
    training.lambda_recon_stage1=1.0 \
    training.lambda_contrastive=0.5 \
    training.log_interval=2 \
    model.event_input_mode=qva \
    wandb.enabled=false

echo "[$(date)] Training done."

# Phase-3 (in-distribution: same 5 vids)
echo "[$(date)] Running Phase-3 inference test (in-distribution)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 5 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20

echo "[$(date)] Done."
