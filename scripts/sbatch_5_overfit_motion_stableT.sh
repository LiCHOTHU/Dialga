#!/bin/bash
#SBATCH --job-name=dlg_5_st
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_st_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_st_%j.err

# Fix for 8354123 catastrophe (F1=0.0 from unstable head training):
# anchor motion teacher's spatial mask on GT q (frozen across epochs).
# This decouples teacher quality from encoder drift — the previous run
# had labels flipping every epoch as q wobbled.
#
# Soft labels (sharpness=8, not 20). Lower sharpness means a small q error
# doesn't produce a binary label flip — gradient signal stays smooth.
# But now GT-q anchor means even minor sharpness misbehavior is moot.
#
# Critical hyperparams:
#   motion_teacher_gt_q   true   (anchor mask on GT q, decouples from encoder)
#   motion_abs_thresh     0.05   (loosened from 0.08; more positive samples
#                                  available since GT q gives a tight mask)
#   sharpness             8.0
#   hard_binarize         false  (smooth gradient; combined with stable q
#                                  this gives clean BCE)
#   pos_weight            20.0

set -euo pipefail
REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"
RUN_NAME=${RUN_NAME:-slot_5_motion_stableT_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"
cd "${REPO}"
echo "Run dir: ${OUTDIR}"
export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.windows_per_video=8 \
    training.batch_size=8 \
    training.num_workers=2 \
    training.stage1_epochs=80 \
    training.stage2_epochs=0 \
    training.ckpt_every=80 \
    training.noise_sigma=0.0 \
    training.lambda_state=1.0 \
    training.lambda_state_anneal_to=1.0 \
    training.lambda_state_anneal_epochs=0 \
    training.lambda_solver=1.0 \
    training.lambda_static=0.1 \
    training.lambda_collision=0.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=20.0 \
    training.event_label_dilation=3 \
    training.event_supervision=motion \
    training.self_event_z_thresh=2.0 \
    training.self_event_sharpness=8.0 \
    training.pixel_event_attention_sigma=0.20 \
    training.motion_abs_thresh=0.05 \
    training.event_teacher_hard=false \
    training.motion_teacher_gt_q=true \
    training.lambda_recon_stage1=1.0 \
    training.lambda_contrastive=0.5 \
    training.log_interval=10 \
    model.event_input_mode=qva \
    wandb.enabled=false

"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 5 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20
