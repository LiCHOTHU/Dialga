#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200
#SBATCH --time=01:00:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=todo1_3ckpts
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# TODO 1: render 5 rollout videos from each of three headline
# candidates so we can pick the visually best one for TODO 2's
# decoder fine-tune base.
#
# Candidates:
#   A) v511_long v5.pt        — ep 800, no LN     (val_recon 0.0175)
#   B) v511_long v5_best.pt   — ep ~100, no LN
#   C) v512_e1 v5_best.pt     — ep 825,  LN       (val_recon 0.0187)
#
# Auto-detect on `norm_static.weight` in the encoder state_dict.
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
LOG_ROOT="/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs"
mkdir -p "${LOG_ROOT}"
cd "${WORKDIR}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

OUT_ROOT="${WORKDIR}/outputs/todo1_compare_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
echo "OutRoot: ${OUT_ROOT}"

CKPT_A="${WORKDIR}/outputs/v511_long_20260521_172330/v5.pt"
CKPT_B="${WORKDIR}/outputs/v511_long_20260521_172330/v5_best.pt"
CKPT_C="${WORKDIR}/outputs/v512_e1_scale_20260522_023749/v5_best.pt"

for tag in A B C; do
    case "${tag}" in
        A) CKPT="${CKPT_A}"; LABEL="A_v511_v5_ep800_noLN" ;;
        B) CKPT="${CKPT_B}"; LABEL="B_v511_v5best_noLN" ;;
        C) CKPT="${CKPT_C}"; LABEL="C_v512e1_v5best_LN" ;;
    esac
    OUT="${OUT_ROOT}/${LABEL}"
    mkdir -p "${OUT}"
    echo ""; echo "==== Rendering ${LABEL} ===="
    echo "ckpt: ${CKPT}"
    python -u scripts/viz/save_v51_overfit_videos.py \
        --ckpt "${CKPT}" \
        --cache_dir "${WAN_CACHE}" \
        --out_dir "${OUT}" \
        --n_videos 5 --use_gt_gate
done

echo ""; echo "Finished at $(date)"
echo "Compare videos under: ${OUT_ROOT}"
