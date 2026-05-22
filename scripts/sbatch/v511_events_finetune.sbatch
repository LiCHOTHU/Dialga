#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=evfine
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# TODO 0.1: frozen-encoder events fine-tune.
# Load Exp 1 v5_best.pt -> freeze enc/dec/fwd/ah -> train only
# EventHead+GEvent+GatePredictor for 10 ep on 10k vids.
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
IN_CKPT="${WORKDIR}/outputs/v511_exp1_attrs_20260521_095305/v5_best.pt"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v511_events_finetune}"
OUT_DIR="${WORKDIR}/outputs/${RUN_TAG}_${STAMP}"
mkdir -p "${OUT_DIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-local}"
echo "OutDir:  ${OUT_DIR}"
echo "InCkpt:  ${IN_CKPT}"
echo "Start:   $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

nvidia-smi || true
python -V

if [ ! -f "${IN_CKPT}" ]; then
  echo "*** in_ckpt not found: ${IN_CKPT} ***" >&2
  exit 2
fi

echo ""; echo "==== TODO 0.1: events fine-tune (frozen encoder) ===="
python -u scripts/train_v5_events.py \
    --cache_dir "${WAN_CACHE}" \
    --in_ckpt   "${IN_CKPT}" \
    --out_dir   "${OUT_DIR}" \
    --epochs 10 --batch_size 16 --num_workers 4 \
    --lr 1e-3 --weight_decay 1e-3 \
    --lambda_event_aux 1.0 --lambda_gate 1.0 \
    --val_frac 0.2 --val_every 1 --log_every 1 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}_${STAMP}" \
    --device cuda

echo ""; echo "Finished at $(date)"
