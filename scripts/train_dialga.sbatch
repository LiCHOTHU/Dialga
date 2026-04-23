#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-h100,gpu-a100,gpu-v100,gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=48G
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --job-name=dialga-train
#SBATCH --output=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/slurm/%x_%j.out
#SBATCH --error=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/slurm/%x_%j.err

set -euo pipefail

source ~/.bashrc
conda activate river

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
SCRIPT="${WORKDIR}/train.py"

cd "${WORKDIR}"
mkdir -p "${WORKDIR}/outputs/slurm"

TRAIN_ARGS=("$@")

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  TRAIN_ARGS=(
    "training.inference_every=10"
    "training.save_inference_images=true"
    "training.save_inference_video=true"
  )
fi

echo "======================================"
echo "JobID:      ${SLURM_JOB_ID}"
echo "JobName:    ${SLURM_JOB_NAME}"
echo "Node:       ${SLURM_NODELIST}"
echo "Workdir:    ${WORKDIR}"
echo "Python:     $(which python)"
echo "Conda env:  ${CONDA_DEFAULT_ENV:-unknown}"
echo "Start:      $(date)"
echo "Overrides:  ${TRAIN_ARGS[*]}"
echo "======================================"

export PYTHONUNBUFFERED=1

python "${SCRIPT}" "${TRAIN_ARGS[@]}"

echo "Finished at $(date)"
