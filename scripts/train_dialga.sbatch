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
#SBATCH --job-name=dlgbase
#SBATCH --output=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/slurm/%x_%j.out
#SBATCH --error=/storage/home/hcoda1/8/lwang831/workspace/Dialga/outputs/slurm/%x_%j.err


source ~/.bashrc
conda activate river

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
SCRIPT="${WORKDIR}/train.py"

cd "${WORKDIR}"
mkdir -p "${WORKDIR}/outputs/slurm"

TRAIN_ARGS=("$@")

if [[ ${#TRAIN_ARGS[@]} -eq 0 ]]; then
  TRAIN_ARGS=(
    "training.epochs=1"
    "training.log_interval=200"
    "training.val_every=1"
    "training.val_every_steps=1000"
    "training.val_max_batches=8"
    "training.checkpoint_every=1"
    "training.checkpoint_every_steps=2000"
    "training.inference_every=1"
    "training.inference_every_steps=1000"
    "training.save_inference_images=true"
    "training.save_inference_video=true"
    "training.save_final_inference=true"
    "training.lambda_solver_mse=0.1"
  )
fi

HAS_WANDB_NAME=0
for arg in "${TRAIN_ARGS[@]}"; do
  if [[ "${arg}" == wandb.name=* ]]; then
    HAS_WANDB_NAME=1
    break
  fi
done
if [[ ${HAS_WANDB_NAME} -eq 0 ]]; then
  TRAIN_ARGS+=("wandb.name=${SLURM_JOB_NAME}-${SLURM_JOB_ID}")
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
export WANDB__SERVICE_WAIT=300

python "${SCRIPT}" "${TRAIN_ARGS[@]}"

echo "Finished at $(date)"
