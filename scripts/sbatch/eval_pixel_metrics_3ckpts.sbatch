#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-v100,gpu-a100,gpu-h100,gpu-h200
#SBATCH --time=01:30:00
#SBATCH --mem-per-gpu=32G
#SBATCH --job-name=pix_metrics
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# VAE-ceiling check (task #29). Computes PSNR/SSIM/LPIPS on val for:
#   A) ceiling vs raw    -- Wan2.2 VAE round-trip
#   B) model   vs raw    -- our end-to-end pixel quality
#   C) model   vs ceiling
# Run on three checkpoints so we get one table per candidate.
# ===================================================================

set -eo pipefail

WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CONDA_ENV="river"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
VIDEO_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
TORCH_CACHE="/storage/project/r-agarg35-0/lwang831/torch_cache"
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
export TORCH_HOME="${TORCH_CACHE}"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCH_CACHE}"

OUT_ROOT="${WORKDIR}/outputs/pixel_metrics_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
echo "OutRoot: ${OUT_ROOT}"

CKPT_A="${WORKDIR}/outputs/v511_long_20260521_172330/v5.pt"
CKPT_B="${WORKDIR}/outputs/v511_long_20260521_172330/v5_best.pt"
CKPT_C="${WORKDIR}/outputs/v512_e1_scale_20260522_023749/v5_best.pt"

N_CHUNKS=${N_CHUNKS:-300}

for tag in A B C; do
    case "${tag}" in
        A) CKPT="${CKPT_A}"; LABEL="A_v511_v5_ep800_noLN" ;;
        B) CKPT="${CKPT_B}"; LABEL="B_v511_v5best_noLN" ;;
        C) CKPT="${CKPT_C}"; LABEL="C_v512e1_v5best_LN" ;;
    esac
    OUT_JSON="${OUT_ROOT}/${LABEL}.json"
    echo ""; echo "==== ${LABEL} ===="
    echo "ckpt: ${CKPT}"
    python -u scripts/eval_pixel_metrics.py \
        --ckpt "${CKPT}" \
        --cache_dir "${WAN_CACHE}" \
        --video_dir "${VIDEO_DIR}" \
        --n_chunks "${N_CHUNKS}" \
        --val_frac 0.2 --seed 42 \
        --batch_size 4 --num_workers 2 \
        --dtype float16 \
        --out_json "${OUT_JSON}"
done

echo ""; echo "Finished at $(date)"
echo "Results under: ${OUT_ROOT}"
