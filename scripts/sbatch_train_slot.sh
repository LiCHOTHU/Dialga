#!/bin/bash
#SBATCH --job-name=dialga_slot
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --partition=gpu-l40s
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/dialga_outputs/slurm/slot_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/dialga_outputs/slurm/slot_%j.err

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/dialga_outputs

mkdir -p "${OUTBASE}/slurm"

# Defaults — override via env vars when sbatch'ing
MAX_VIDEOS=${MAX_VIDEOS:-200}
WINDOWS_PER_VIDEO=${WINDOWS_PER_VIDEO:-8}
BATCH_SIZE=${BATCH_SIZE:-16}
STAGE1_EPOCHS=${STAGE1_EPOCHS:-100}
STAGE2_EPOCHS=${STAGE2_EPOCHS:-50}
RUN_NAME=${RUN_NAME:-option_a_v${MAX_VIDEOS}_$(date +%m%d_%H%M)}

OUTDIR="${OUTBASE}/${RUN_NAME}"
mkdir -p "${OUTDIR}"

cd "${REPO}"

echo "[$(date)] starting run ${RUN_NAME}"
echo "  output dir       : ${OUTDIR}"
echo "  max_videos       : ${MAX_VIDEOS}"
echo "  windows/video    : ${WINDOWS_PER_VIDEO}"
echo "  batch_size       : ${BATCH_SIZE}"
echo "  stage1/2 epochs  : ${STAGE1_EPOCHS}/${STAGE2_EPOCHS}"
echo "  GPU              : $(nvidia-smi -L 2>/dev/null || echo 'no GPU?')"

"${PYTHON}" scripts/train_slot.py \
    hydra.run.dir="${OUTDIR}" \
    training.max_videos=${MAX_VIDEOS} \
    training.windows_per_video=${WINDOWS_PER_VIDEO} \
    training.batch_size=${BATCH_SIZE} \
    training.stage1_epochs=${STAGE1_EPOCHS} \
    training.stage2_epochs=${STAGE2_EPOCHS} \
    training.num_workers=4 \
    wandb.enabled=true \
    wandb.name="${RUN_NAME}" \
    wandb.tags='[option_a,clevrer,slot]'

# Mirror small viewable artifacts (loss plots + scene GIFs + summary) to local
# repo outputs/ so the user can browse them without going through project storage.
LOCAL_MIRROR="${REPO}/outputs/${RUN_NAME}"
mkdir -p "${LOCAL_MIRROR}"
cp -f "${OUTDIR}"/stage1_loss.png       "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/stage2_loss.png       "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/scene_video_*.gif     "${LOCAL_MIRROR}/" 2>/dev/null || true
cp -f "${OUTDIR}"/train_slot.log        "${LOCAL_MIRROR}/" 2>/dev/null || true
echo "[$(date)] mirrored artifacts to ${LOCAL_MIRROR}"
echo "[$(date)] full checkpoints + logs remain at ${OUTDIR}"
