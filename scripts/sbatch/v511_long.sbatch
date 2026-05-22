#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h200,gpu-h100,gpu-a100
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=v511long
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# v5.1.1 long-training run.  Same model + losses as Exp 1 (AttrsHead),
# three deltas:
#   - lr        : 1e-3 -> 5e-4
#   - epochs    : 200  -> 1000
#   - early_stop: patience 15 -> 0 (disabled)
#
# Motivation: at the Exp 1 stop point (ep 156) val_recon had plateaued
# but L_attrs was still dropping 3x and L_fwd / L_event_aux were rising
# (i.e., not converged).  Run long enough to let every loss flatten,
# with halved LR for stability on the rising aux losses.
#
# Two ckpts at the end:
#   v5_best.pt   - best val_recon (probably from early stage 2)
#   v5.pt        - final (ep 1000); probably weaker val_recon but
#                  smaller L_attrs and stronger identity probes.
# Probes run on whichever exists.
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
RUN_TAG="${RUN_TAG:-v511_long}"
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

echo ""; echo "==== v5.1.1 LONG: lr 5e-4, 1000 ep, no early stop ===="
python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 0 \
    --batch_size 16 --num_workers 4 \
    --epochs 1000 --stage1_epochs 10 --stage2_epochs 990 \
    --lr 5e-4 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 64 --d_dyn 64 --d_state 32 \
    --enc_hidden_ch 128 --dec_hidden_ch 256 \
    --lambda_pred 1.0 --lambda_fwd 0.1 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lambda_attrs 0.05 \
    --infonce_temperature 0.1 \
    --early_stop_patience 0 \
    --val_frac 0.2 --val_every 5 \
    --log_every 1 --ckpt_every 50 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}_${STAMP}" \
    --device cuda

# Prefer v5_best.pt (best val_recon); fall back to v5.pt (final)
CKPT="${OUT_DIR}/v5_best.pt"
if [ ! -f "${CKPT}" ]; then
  CKPT="${OUT_DIR}/v5.pt"
fi
if [ ! -f "${CKPT}" ]; then
  echo "*** No checkpoint at ${OUT_DIR} ***" >&2; exit 3
fi
echo "Probing best-ckpt: ${CKPT}"

echo ""; echo "==== modal-attribute probe (best ckpt) ===="
python -u scripts/probes/probe_v5_modal.py \
    --cache_dir "${WAN_CACHE}" --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0 \
    --batch_size 16 --num_workers 4

echo ""; echo "==== z_dyn diag probe (best ckpt) ===="
python -u scripts/probes/probe_v5_zdyn_diag.py \
    --cache_dir "${WAN_CACHE}" --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0 \
    --batch_size 16 --num_workers 4

# Also probe the FINAL ckpt to see what extra training bought us
FINAL_CKPT="${OUT_DIR}/v5.pt"
if [ -f "${FINAL_CKPT}" ] && [ "${FINAL_CKPT}" != "${CKPT}" ]; then
  echo ""; echo "==== modal-attribute probe (FINAL ckpt) ===="
  python -u scripts/probes/probe_v5_modal.py \
      --cache_dir "${WAN_CACHE}" --ckpt "${FINAL_CKPT}" \
      --val_frac 0.2 --seed 42 --probe_split_seed 0 \
      --batch_size 16 --num_workers 4
fi

echo ""; echo "==== render 5 rollout videos (best ckpt) ===="
python -u scripts/viz/save_v51_overfit_videos.py \
    --ckpt "${CKPT}" \
    --cache_dir "${WAN_CACHE}" \
    --n_videos 5 --use_gt_gate

echo ""; echo "Finished at $(date)"
