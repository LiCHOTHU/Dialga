#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=00:40:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v58_droid_readout
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.8 DROID stage — STEP 3/3: OFFLINE readout of camera conditioning on the
# held-out episodes. Compares pose-ON vs pose-BLIND on:
#   corr(‖z_dyn‖, ‖cam_vel‖)      (lower = camera divided out)
#   R²  z_dyn -> cam_vel          (lower = camera not recoverable from z_dyn)
#   z_static within-episode ratio (lower = viewpoint-stable identity)
# pose conditioning WORKS iff ON is lower than BLIND on all three.
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
DROID_CACHE="/storage/scratch1/8/lwang831/cache/droid_wrist_W33"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"; export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

echo "############ DROID camera readout: ON vs BLIND ############"
python -u scripts/probes/droid_camera_readout.py \
    --cache_dir "${DROID_CACHE}" --val_frac 0.2 --seed 42 \
    --ckpt on:"${CEDAR}/v58_droid_on/v5.pt" \
    --ckpt blind:"${CEDAR}/v58_droid_blind/v5.pt" \
    --out_json "${CEDAR}/v58_droid_readout.json"

echo "V58_DROID_READOUT_DONE at $(date)"
