#!/bin/bash
#SBATCH --job-name=dlg_5_m80
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_m80_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_m80_%j.err

# 5-vid × 80 epochs definitive test of the motion-centroid event teacher,
# with strong lambda_state=1.0 held constant. This matches the training
# budget the supervised baseline needs to reach F1=0.78 on 20 vids.
#
# Outcome interpretation:
#   - F1 > 0.5  → motion-teacher works given proper budget; cogsci pipeline
#                 is operational; only thing left for "domain-portable" claim
#                 is replacing GT-q with a self-derived position signal.
#   - F1 ~ 0   → teacher has a remaining issue we haven't found.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_5_motion_80ep_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 5-vid × 80ep motion teacher (definitive)"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "  lambda_state          1.0 (constant, no curriculum)"
echo "  event_supervision     motion (Δ² saliency centroid)"
echo "  epochs                80"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.windows_per_video=8 \
    training.batch_size=8 \
    training.num_workers=2 \
    training.stage1_epochs=80 \
    training.stage2_epochs=0 \
    training.ckpt_every=20 \
    training.noise_sigma=0.0 \
    training.lambda_state=1.0 \
    training.lambda_state_anneal_to=1.0 \
    training.lambda_state_anneal_epochs=0 \
    training.lambda_solver=1.0 \
    training.lambda_static=0.1 \
    training.lambda_collision=0.0 \
    training.lambda_event=1.0 \
    training.event_pos_weight=50.0 \
    training.event_label_dilation=3 \
    training.event_supervision=motion \
    training.self_event_z_thresh=2.0 \
    training.self_event_sharpness=2.5 \
    training.pixel_event_attention_sigma=0.20 \
    training.lambda_recon_stage1=1.0 \
    training.lambda_contrastive=0.5 \
    training.log_interval=5 \
    model.event_input_mode=qva \
    wandb.enabled=false

echo "[$(date)] Training done."
echo "[$(date)] Running Phase-3 inference test (in-distribution)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 5 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20
echo "[$(date)] Done."
