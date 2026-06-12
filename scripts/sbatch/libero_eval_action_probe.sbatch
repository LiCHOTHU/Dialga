#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=libero_probe
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Action-prediction probe on the LIBERO-trained v5.1.2 checkpoint.
# Frozen encoder; tiny MLP maps z_dyn -> per-latent-frame (9, 7) action.
# Compares against z_static, raw Wan latent, and random-init encoder baselines
# across train/val/heldout-tasks splits.

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
LIBERO_CACHE="/storage/scratch1/8/lwang831/cache/libero_90_wan_W33"
CKPT_DIR="${CKPT_DIR:-/storage/scratch1/8/lwang831/dialga_outputs/libero_v512_big}"
CKPT="${CKPT:-${CKPT_DIR}/v5_best.pt}"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

echo "ckpt: ${CKPT}"; nvidia-smi || true
[ -f "${CKPT}" ] || { echo "MISSING ckpt ${CKPT}"; exit 1; }

python -u scripts/eval_libero_action_probe.py \
    --cache_dir "${LIBERO_CACHE}" \
    --ckpt "${CKPT}" \
    --n_holdout_tasks 10 --task_holdout_seed 0 \
    --val_every_k_episodes 10 --require_chunks_per_episode 3 \
    --batch_size 32 --num_workers 4 \
    --probe_epochs 20 --probe_lr 1e-3 --probe_hidden 128 \
    --features z_dyn z_static wan_mean random_init_z_dyn \
    --device cuda

echo "Done at $(date)"
