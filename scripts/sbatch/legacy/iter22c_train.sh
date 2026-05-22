#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=dialga_iter22c
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Iter 22 — Job 3/3: Iter 22c training (--lambda_attrs 1.0) + probes.
#
# INDEPENDENT of the DINO cache — can be submitted alongside the
# cache job. SLURM will schedule them on separate GPUs if available.
#
# After training, runs the linear identity probe and the z_dyn
# diagnostic probe on the produced checkpoint.
#
# Outputs:
#   /storage/home/.../outputs/iter22c_attrs_${STAMP}/trajectory.pt
#   /storage/home/.../outputs/iter22c_attrs_${STAMP}/probe_identity_linear.json
#   /storage/home/.../outputs/iter22c_attrs_${STAMP}/probe_identity_diag.json
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
OUT_DIR="${WORKDIR}/outputs/iter22c_attrs_${STAMP}"
mkdir -p "${OUT_DIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "WanCache:${WAN_CACHE}"
echo "OutDir:  ${OUT_DIR}"
echo "Stamp:   ${STAMP}"
echo "Start:   $(date)"
echo "========================================="

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

nvidia-smi || true
python -V

# Sanity: Wan cache present (DINO cache NOT needed for this job).
N_WAN=$(ls "${WAN_CACHE}/latents" 2>/dev/null | wc -l)
if [ "${N_WAN}" -lt 1 ]; then
  echo "*** Wan cache empty at ${WAN_CACHE}/latents — build it first ***" >&2
  exit 2
fi
echo "Wan cache present: ${N_WAN} latents"

# ---------- Phase 3 Iter 22c training -------------------------------
echo ""; echo "==== Phase 3: Iter 22c training (--lambda_attrs 1.0) ===="
python -u scripts/legacy/train_trajectory.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --epochs 60 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_attrs 1.0 \
    --val_frac 0.2 --val_every 5 \
    --log_every 50 --ckpt_every 5 \
    --device cuda

CKPT="${OUT_DIR}/trajectory.pt"
if [ ! -f "${CKPT}" ]; then
  echo "*** No checkpoint produced at ${CKPT} ***" >&2; exit 3
fi
echo "Iter 22c checkpoint: ${CKPT}"

# ---------- Phase 4 probes on 22c -----------------------------------
echo ""; echo "==== Phase 4c: linear identity probe ===="
python -u scripts/legacy/probes/probe_iter21_identity.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0

echo ""; echo "==== Phase 4c: z_dyn diagnostic probe ===="
python -u scripts/legacy/probes/probe_iter21_zdyn_diag.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0

echo ""; echo "Finished at $(date)"
echo "Results: ${OUT_DIR}/probe_identity_linear.json"
echo "         ${OUT_DIR}/probe_identity_diag.json"
