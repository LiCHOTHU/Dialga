#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=e1attr
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Exp 1: v511s + AttrsHead supervised CE (lambda_attrs=0.05).
#
# Single delta vs v511s: add --lambda_attrs 0.05.
# Everything else identical (10k vids, enc 128 / dec 256, InfoNCE,
# split trunks, no stage 3, early-stop patience 15).
#
# Question: does weak attribute supervision push material/shape probes
# above +0.15 without hurting color (+0.224) or val_recon (~0.017)?
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}"

cd "${WORKDIR}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TAG="${RUN_TAG:-v511_exp1_attrs}"
OUT_DIR="${WORKDIR}/outputs/${RUN_TAG}_${STAMP}"
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
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

nvidia-smi || true
python -V

N_BLOBS=$(ls "${WAN_CACHE}/latents" 2>/dev/null | wc -l)
if [ "${N_BLOBS}" -lt 1 ]; then
  echo "*** Wan cache empty at ${WAN_CACHE}/latents ***" >&2
  exit 2
fi
echo "Wan cache present: ${N_BLOBS} latents"

echo ""; echo "==== Exp 1: v511s + AttrsHead (lambda_attrs=0.05) ===="
python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 0 \
    --batch_size 16 --num_workers 4 \
    --epochs 200 --stage1_epochs 10 --stage2_epochs 190 \
    --lr 1e-3 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 64 --d_dyn 64 --d_state 32 \
    --enc_hidden_ch 128 --dec_hidden_ch 256 \
    --lambda_pred 1.0 --lambda_fwd 0.1 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lambda_attrs 0.05 \
    --infonce_temperature 0.1 \
    --early_stop_patience 15 --early_stop_min_delta 5e-5 \
    --val_frac 0.2 --val_every 2 \
    --log_every 1 --ckpt_every 10 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}_${STAMP}" \
    --device cuda

# Prefer v5_best.pt if early stop hit; fall back to v5.pt
CKPT="${OUT_DIR}/v5_best.pt"
if [ ! -f "${CKPT}" ]; then
  CKPT="${OUT_DIR}/v5.pt"
fi
if [ ! -f "${CKPT}" ]; then
  echo "*** No checkpoint at ${OUT_DIR} ***" >&2; exit 3
fi
echo "Probing ckpt: ${CKPT}"

echo ""; echo "==== modal-attribute probe ===="
python -u scripts/probes/probe_v5_modal.py \
    --cache_dir "${WAN_CACHE}" --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0 \
    --batch_size 16 --num_workers 4

echo ""; echo "==== z_dyn diag probe ===="
python -u scripts/probes/probe_v5_zdyn_diag.py \
    --cache_dir "${WAN_CACHE}" --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0 \
    --batch_size 16 --num_workers 4

echo ""; echo "==== render 5 rollout videos ===="
python -u scripts/viz/save_v51_overfit_videos.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --n_videos 5 --use_gt_gate

echo ""; echo "Finished at $(date)"
