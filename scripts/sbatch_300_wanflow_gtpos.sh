#!/bin/bash
#SBATCH --job-name=dlg_300_wf_gtpos
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_wf_gtpos_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_wf_gtpos_%j.err

# Wan-flow decoder, GT q baseline at scale.
# Submit with --dependency=afterok:<slot_job_id>.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

GROUP=${GROUP:-scale_300_$(date +%m%d)}
SLOT_RUN=${SLOT_RUN:-slot_300_${GROUP}}
SLOT_CKPT=${SLOT_CKPT:-${OUTBASE}/${SLOT_RUN}/stage2.pt}
RUN_NAME=${RUN_NAME:-wanflow_300_gtpos_${GROUP}}
OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

if [[ ! -f "${SLOT_CKPT}" ]]; then
    echo "ERROR: slot checkpoint not found: ${SLOT_CKPT}"
    echo "       set SLOT_CKPT or SLOT_RUN env var, or wait for the slot job to finish."
    exit 2
fi

echo "=========================================="
echo " Wan-flow decoder @ 300 videos (GT q)"
echo "=========================================="
echo "  JobID            : ${SLURM_JOB_ID}"
echo "  Slot ckpt        : ${SLOT_CKPT}"
echo "  Output dir       : ${OUTDIR}"
echo "  Wandb group      : ${GROUP}"
echo "  Wandb name       : ${RUN_NAME}"
echo "  GPU              : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start            : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1
export WANDB__SERVICE_WAIT=300

"${PYTHON}" -u scripts/overfit_wan_flow.py \
    --ckpt "${SLOT_CKPT}" \
    --out_dir "${OUTDIR}" \
    --max_videos 300 \
    --cond_source z_static \
    --use_gt_pos \
    --epochs 200 \
    --batch_size 8 \
    --lr 2e-4 \
    --d_model 512 \
    --n_blocks 8 \
    --n_heads 8 \
    --ema_decay 0.999 \
    --time_dist logitnorm \
    --num_sample_steps 8 \
    --ckpt_every 10 \
    --wandb \
    --wandb_project dialga \
    --wandb_name "${RUN_NAME}" \
    --wandb_group "${GROUP}" \
    --wandb_tags "scale_300,wanflow,gt_pos,baseline"

echo "[$(date)] Wan-flow GT-q done."
LOCAL_MIRROR="${REPO}/outputs/${RUN_NAME}"
mkdir -p "${LOCAL_MIRROR}"
cp -f "${OUTDIR}"/video_*_grid.png  "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/decoder.pt         "${LOCAL_MIRROR}/" 2>/dev/null || true
echo "  -> ${LOCAL_MIRROR}/"
