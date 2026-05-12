#!/bin/bash
#SBATCH --job-name=dlg_20_gen
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_20_gen_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_20_gen_%j.err

# 20-video overfit test of the domain-portable representation:
#   - Tiny lambda_state = 0.05 (weak position anchor; in the truly-no-GT
#     case this would be replaced by centroid-of-motion-mask under static
#     camera. Keeping it small + GT here is just a stand-in.)
#   - Pixel-reconstruction grounding via SlotPixelDecoder (lambda_recon_stage1 = 1)
#     -- decoder NOW uses q-relative coord channels, so it is identifiable
#     in q (no longer collapses to constant-velocity encoder).
#   - Self-supervised event labels from z-scored inertial residual
#     (event_supervision = "self") — NO GT collision_mask
#   - InfoNCE slot-contrastive on z_static (lambda_contrastive = 0.5)
#     -- prevents identity collapse across slots.
#   - Encoder + AccelNet + EventHead + SlotPixelDecoder trained jointly
#
# Success criterion: pixel-recon drops below ~0.005, event head fires at
# real collision frames, Phase-3 F1 vs CLEVRER GT > 0.5.
#
# Side-by-side reference: scripts/sbatch_20_overfit_eventhead.sh (CLEVRER-
# supervised version that previously got F1 = 0.779).

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_20_general_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 20-video overfit, GENERAL representation"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "  Supervision:"
echo "    lambda_state          = 0.05  (weak position anchor)"
echo "    lambda_recon_stage1   = 1.0   (pixel-recon, q-identifiable decoder)"
echo "    lambda_contrastive    = 0.5   (InfoNCE slot-discrimination)"
echo "    event_supervision     = self  (z-score teacher)"
echo "    event_input_mode      = qva"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=20 \
    training.windows_per_video=64 \
    training.batch_size=8 \
    training.num_workers=4 \
    training.stage1_epochs=80 \
    training.stage2_epochs=0 \
    training.ckpt_every=10 \
    training.noise_sigma=0.0 \
    training.lambda_state=0.05 \
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
    model.event_input_mode=qva \
    wandb.enabled=false

echo "[$(date)] Training done."

# In-distribution Phase-3 collision-event test on the 20 training videos.
# In the general regime the EventHead is supervised by self-generated labels;
# Phase-3 still compares its predictions against CLEVRER GT collisions to
# see whether the self-supervised signal recovers the same events.
echo "[$(date)] Running Phase-3 inference test (in-distribution, vs CLEVRER GT)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 20 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -40

echo "[$(date)] Done."
