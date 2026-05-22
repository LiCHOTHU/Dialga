#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=v511
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# v5.1.1 main 10k-vid run.
#
# Architecture (validated by 20-vid overfit, ep 1500, all losses converged):
#   - chunk-wise paired sampler on the W=33 cache (T_lat=9)
#   - encoder: per-frame z_dyn (B, T, D_d=64); time-pooled z_static (B, D_s=64)
#   - decoder: per-frame (z_static, z_dyn[t]) -> chunk
#   - ForwardDynamics: chunk-to-chunk via chunk_step (one Verlet call,
#     analytical T-frame expansion; constant accel within chunk)
#   - InfoNCE on z_static at temperature 0.1
#   - Stage gating: 3 epochs recon-only / 22 epochs +pred+fwd+consist /
#     5 epochs +event_aux+gate
#
# Followed by probe_v5_modal + probe_v5_zdyn_diag on the ckpt.
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
OUT_DIR="${WORKDIR}/outputs/v511_main_${STAMP}"
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

echo ""; echo "==== v5.1.1 training (10k vids, all 3 stages) ===="
python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 0 \
    --batch_size 16 --num_workers 4 \
    --epochs 120 --stage1_epochs 12 --stage2_epochs 88 \
    --lr 1e-3 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 64 --d_dyn 64 --d_state 32 \
    --enc_hidden_ch 64 --dec_hidden_ch 128 \
    --lambda_pred 1.0 --lambda_fwd 0.1 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --infonce_temperature 0.1 \
    --val_frac 0.2 --val_every 2 \
    --log_every 1 --ckpt_every 5 \
    --device cuda

CKPT="${OUT_DIR}/v5.pt"
if [ ! -f "${CKPT}" ]; then
  echo "*** No checkpoint at ${CKPT} ***" >&2; exit 3
fi

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
