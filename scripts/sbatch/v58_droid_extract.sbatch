#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v58_droid_extract
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.8 DROID stage — STEP 1/3: cache DROID wrist-camera episodes as Wan latents
# WITH the known EE (=camera) pose. Small-first: first N_EPISODES episodes,
# near-static cameras dropped (nothing to divide out). Idempotent (skips blobs
# that already exist), so re-runs resume.
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
DROID_CACHE="/storage/scratch1/8/lwang831/cache/droid_wrist_W33"
N_EPISODES="${N_EPISODES:-300}"
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
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${DROID_CACHE}"
nvidia-smi || true

echo "############ DROID extract: first ${N_EPISODES} episodes -> ${DROID_CACHE} ############"
python -u scripts/extract_droid_wrist.py \
    --out_dir "${DROID_CACHE}" \
    --start_episode 0 --end_episode "${N_EPISODES}" \
    --min_ee_span 0.05 --dtype float16

echo "V58_DROID_EXTRACT_DONE at $(date)"
