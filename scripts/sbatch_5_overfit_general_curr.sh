#!/bin/bash
#SBATCH --job-name=dlg_5_gc
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-h100,gpu-l40s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_gc_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_5_gc_%j.err

# 5-VIDEO ULTRA-FAST overfit (~5 min budget) of the domain-portable
# representation with lambda_state CURRICULUM (1.0 → 0.1 over first 6 epochs).
# Goal: validate that with a strong initial GT-q pull, the encoder produces
# 2nd-diff spikes, the self-event teacher gets real signal, and the event
# head learns. If solver loss stays NON-ZERO and event loss falls below
# ~1.0, the architecture is sound and only the teacher generation needs
# replacement to be fully domain-portable.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga
mkdir -p "${OUTBASE}/slurm"

RUN_NAME=${RUN_NAME:-slot_5_gen_curr_$(date +%m%d_%H%M)}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "=========================================="
echo " 5-vid quick overfit (general + curriculum)"
echo "=========================================="
echo "  JobID    : ${SLURM_JOB_ID}"
echo "  Output   : ${OUTDIR}"
echo "  GPU      : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start    : $(date)"
echo "  Supervision:"
echo "    lambda_state          1.0 → 0.1  (anneal over 6 epochs)"
echo "    lambda_recon_stage1   1.0   (pixel-recon, q-identifiable decoder)"
echo "    lambda_contrastive    0.5   (InfoNCE slot-discrimination)"
echo "    event_supervision     self  (z-score teacher)"
echo "    event_input_mode      qva"
echo "=========================================="

export PYTHONUNBUFFERED=1

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=5 \
    training.windows_per_video=8 \
    training.batch_size=8 \
    training.num_workers=2 \
    training.stage1_epochs=15 \
    training.stage2_epochs=0 \
    training.ckpt_every=15 \
    training.noise_sigma=0.0 \
    training.lambda_state=1.0 \
    training.lambda_state_anneal_to=0.1 \
    training.lambda_state_anneal_epochs=6 \
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
    training.log_interval=1 \
    model.event_input_mode=qva \
    wandb.enabled=false

echo "[$(date)] Training done."

# Phase-3 against CLEVRER GT (in-distribution on the same 5 vids)
echo "[$(date)] Running Phase-3 inference test (in-distribution)..."
"${PYTHON}" scripts/test_event_head.py \
    --ckpt "${OUTDIR}/stage1.pt" \
    --max_videos 5 \
    --in_distribution \
    --output "${OUTDIR}/phase3_indist.json" 2>&1 | tail -20

echo "[$(date)] Done."
