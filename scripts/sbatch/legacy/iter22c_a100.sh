#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100
#SBATCH --time=07:45:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=dialga_iter22c_a100
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Iter 22c re-run on A100/H100 — color baseline reference.
#
# Changes vs first attempt (which timed out at ep 5/60 on V100):
#   - drop gpu-v100 partition (was 50 min/ep there)
#   - batch_size 4 -> 16 (A100 80GB has plenty of headroom)
#   - num_workers 0 -> 4 (dataloader was the V100 bottleneck)
#   - wallclock 4h -> 7h45 (embers QoS cap is 8h)
# Same loss recipe as before: lambda_attrs=1.0, 60 epochs, lr=5e-4.
#
# Probes run after training on the produced ckpt.
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
OUT_DIR="${WORKDIR}/outputs/iter22c_attrs_a100_${STAMP}"
mkdir -p "${OUT_DIR}"

echo "================ Job Info ================"
echo "JobID:   ${SLURM_JOB_ID}"
echo "Node:    ${SLURM_NODELIST}"
echo "OutDir:  ${OUT_DIR}"
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

N_WAN=$(ls "${WAN_CACHE}/latents" 2>/dev/null | wc -l)
if [ "${N_WAN}" -lt 1 ]; then
  echo "*** Wan cache empty at ${WAN_CACHE}/latents — build it first ***" >&2
  exit 2
fi
echo "Wan cache present: ${N_WAN} latents"

echo ""; echo "==== Training (--lambda_attrs 1.0, batch=16, workers=4, A100) ===="
python -u scripts/legacy/train_trajectory.py \
    --cache_dir "${WAN_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --epochs 60 --batch_size 16 --num_workers 4 \
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
echo "Checkpoint: ${CKPT}"

echo ""; echo "==== Linear identity probe ===="
python -u scripts/legacy/probes/probe_iter21_identity.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0

echo ""; echo "==== z_dyn diagnostic probe ===="
python -u scripts/legacy/probes/probe_iter21_zdyn_diag.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0

echo ""; echo "==== Per-slot GT-position raw-latent probe ===="
python -u scripts/legacy/probes/probe_wan_perslot_gtpos.py \
    --cache_dir "${WAN_CACHE}" \
    --ckpt "${CKPT}" \
    --val_frac 0.2 --seed 42 --probe_split_seed 0

echo ""; echo "Finished at $(date)"
