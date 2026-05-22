#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100
#SBATCH --time=05:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=dialga_iter22_cache
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Iter 22 — Job 1/3: smoke verification + full DINO cache.
#
# Phases:
#   0a Tiny Wan cache (5 vids x 4 windows)
#   0b Tiny DINO cache (downloads dinov2-small if absent)
#   0c Contract check on the DINO blob (cls.shape == (12, 384), float32, finite)
#   0d 1-epoch trainer smoke with --lambda_dino 0.5
#   0e 1-epoch trainer smoke with --lambda_attrs 1.0
#   1  Full DINO cache on the 10k Wan cache (resume-safe)
#
# Outputs:
#   /storage/scratch1/8/lwang831/cache/wan_10000vid_W12/dino/<idx>.pt   x 40000
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
DATA_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
ANNOTATION_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W12"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}"

cd "${WORKDIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "WorkDir: ${WORKDIR}"
echo "WanCache:${WAN_CACHE}"
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
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${HF_HOME}" "${HF_HUB_CACHE}"

nvidia-smi || true
python -V
which python

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SMOKE_DIR="${PROJECT_TMP}/iter22_smoke_${STAMP}"
mkdir -p "${SMOKE_DIR}"

gate() {
  if [ "$1" -ne 0 ]; then echo "*** FAILED at: $2 (exit $1) ***" >&2; exit "$1"; fi
}

# ---------- Phase 0 smoke ------------------------------------------
echo ""; echo "==== Phase 0a: tiny Wan cache (5 vids x 4 windows) ===="
python -u scripts/cache_wan_latents.py \
    --data_dir "${DATA_DIR}" \
    --annotation_dir "${ANNOTATION_DIR}" \
    --max_videos 5 --windows_per_video 4 --window_length 12 \
    --frames_per_video 128 --image_size 128 \
    --out_dir "${SMOKE_DIR}" --device cuda --dtype float16
gate $? "phase0a_wan_cache"

echo ""; echo "==== Phase 0b: tiny DINO cache (may download dinov2-small) ===="
python -u scripts/legacy/cache_dino_features.py \
    --cache_dir "${SMOKE_DIR}" \
    --data_dir "${DATA_DIR}" \
    --annotation_dir "${ANNOTATION_DIR}" \
    --device cuda --dtype float16
gate $? "phase0b_dino_cache"

echo ""; echo "==== Phase 0c: contract check ===="
python -c "
import torch, sys
b = torch.load('${SMOKE_DIR}/dino/000000.pt', map_location='cpu', weights_only=False)
cls = b['cls']
print('cls.shape =', tuple(cls.shape))
print('cls.dtype =', cls.dtype)
print('cls.finite =', bool(torch.isfinite(cls).all()))
assert cls.shape == (12, 384), f'expected (12, 384), got {tuple(cls.shape)}'
assert cls.dtype == torch.float32, f'expected float32 on disk, got {cls.dtype}'
assert torch.isfinite(cls).all(), 'non-finite DINO features'
print('OK contract holds')
"
gate $? "phase0c_contract_check"

echo ""; echo "==== Phase 0d: 1-epoch smoke train with --lambda_dino 0.5 ===="
python -u scripts/legacy/train_trajectory.py \
    --cache_dir "${SMOKE_DIR}" \
    --out_dir "${SMOKE_DIR}/smoke_dino" \
    --epochs 1 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_dino 0.5 --d_dino 384 \
    --val_frac 0.0 --log_every 1 --ckpt_every 0 \
    --device cuda
gate $? "phase0d_smoke_dino_train"

echo ""; echo "==== Phase 0e: 1-epoch smoke train with --lambda_attrs 1.0 ===="
python -u scripts/legacy/train_trajectory.py \
    --cache_dir "${SMOKE_DIR}" \
    --out_dir "${SMOKE_DIR}/smoke_attrs" \
    --epochs 1 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_attrs 1.0 \
    --val_frac 0.0 --log_every 1 --ckpt_every 0 \
    --device cuda
gate $? "phase0e_smoke_attrs_train"

echo ""; echo "Phase 0 PASSED"

# ---------- Phase 1 full DINO cache --------------------------------
echo ""; echo "==== Phase 1: full DINO cache on 10k Wan cache ===="
mkdir -p "${WAN_CACHE}/dino"

python -u scripts/legacy/cache_dino_features.py \
    --cache_dir "${WAN_CACHE}" \
    --data_dir "${DATA_DIR}" \
    --annotation_dir "${ANNOTATION_DIR}" \
    --device cuda --dtype float16
gate $? "phase1_full_dino_cache"

N_DINO=$(ls "${WAN_CACHE}/dino" 2>/dev/null | wc -l)
echo "DINO cache complete: ${N_DINO} blobs"

echo ""; echo "Finished at $(date)"
