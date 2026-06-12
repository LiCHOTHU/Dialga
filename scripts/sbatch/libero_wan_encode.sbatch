#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=06:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=libero_wan_encode
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Cache LIBERO-90 (4500 episodes, ~18k chunks at W=33) into Wan-VAE latents.
# Smoke test showed ~1.6 chunk/s on an L40S → ~3.2 hours full run.

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
LIBERO_ROOT="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/LIBERO-datasets/libero_90_processed"
OUT_DIR="/storage/scratch1/8/lwang831/cache/libero_90_wan_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${OUT_DIR}"

echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

python -u scripts/encode_libero_wan.py \
    --libero_root "${LIBERO_ROOT}" \
    --out_dir "${OUT_DIR}" \
    --dtype float16 \
    --device cuda

echo "Done at $(date)"
