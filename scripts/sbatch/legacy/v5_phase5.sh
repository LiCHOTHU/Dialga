#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-h200
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=dialga_v5_phase5
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# v5 Phase-5 full-scale training (10k videos, all stages).
#
# Setup against existing Wan cache (T_lat=3 → L_obs=1, L_pred=2).
# This means L_fwd and L_event_aux are dormant (need L_obs >= 2);
# only L_recon, L_pred, L_consist, L_gate fire. That is a known
# limitation of using the W=12 cache; a re-cache to W=24 would
# enable the full loss table. Phase 5 still tests whether the
# factorization mechanism bites at all at 10k scale.
#
# Followed by probe_v5_modal on the produced checkpoint.
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W12"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}"

cd "${WORKDIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${WORKDIR}/outputs/v5_phase5_${STAMP}"
mkdir -p "${OUT_DIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID:-local}"
echo "Node:    ${SLURM_NODELIST:-local}"
echo "OutDir:  ${OUT_DIR}"
echo "Start:   $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

nvidia-smi || true
python -V

N_WAN=$(ls "${WAN_CACHE}/latents" 2>/dev/null | wc -l)
if [ "${N_WAN}" -lt 1 ]; then
  echo "*** Wan cache empty at ${WAN_CACHE}/latents — build it first ***" >&2
  exit 2
fi
echo "Wan cache present: ${N_WAN} latents"

echo ""; echo "==== v5 training (full 10k vids, all 3 stages, batch=16, workers=4) ===="
python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 0 --pairs_per_video 4 --batch_size 16 --num_workers 4 \
    --epochs 24 --stage1_epochs 3 --stage2_epochs 16 \
    --L_obs 1 --L_pred 2 \
    --d_static 32 --d_dyn 16 --d_state 8 --d_event 4 --hidden_ch 64 \
    --lambda_pred 1.0 --lambda_fwd 0.1 --lambda_consist 0.1 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lr 5e-4 --weight_decay 1e-3 \
    --val_frac 0.2 --val_every 2 \
    --log_every 1 --ckpt_every 4 \
    --device cuda

CKPT="${OUT_DIR}/v5.pt"
if [ ! -f "${CKPT}" ]; then
  echo "*** No checkpoint produced at ${CKPT} ***" >&2; exit 3
fi
echo "Checkpoint: ${CKPT}"

echo ""; echo "==== v5 modal-attribute probe ===="
python -u scripts/probes/probe_v5_modal.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0 \
    --batch_size 16 --num_workers 4

echo ""; echo "Finished at $(date)"
