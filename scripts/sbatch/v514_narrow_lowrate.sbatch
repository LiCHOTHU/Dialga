#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v514_narrow_lowrate
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ============================================================================
# SCALING FACTORIAL (params x rate), cell A = NARROW + LOW-RATE  (baseline)
# ----------------------------------------------------------------------------
# Goal: disentangle whether reconstruction quality is driven by network SIZE
# (enc/dec width = params) or by CODE RATE (size of z). 2x2 design, all cells
# identical EXCEPT width and rate. Everything else matches the final recipe:
#   pixel-dominant (lambda_recon 0.25, lambda_pixel 5.0) + MAE (lambda_mae 0.5)
#   + loss-balance annealing (--anneal_pixel), frozen Wan VAE, 10k videos.
#
#   cell A (this) : enc128/dec256  d64/64/32  -> 644 floats   [baseline]
#   cell B        : enc192/dec384  d64/64/32  -> 644 floats   [params++ only]
#   cell C        : enc128/dec256  d96/96/48  -> 964 floats   [rate++ only]
#   cell D        : enc192/dec384  d96/96/48  -> 964 floats   [= job 9956270
#                                                              v512_clevrer_pixdom_mae]
# Read-out: (B - A) = pure params effect; (C - A) = pure rate effect.
# Larger gap wins. Self-chains across the 8h embers wall.
# ============================================================================

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
DINO_CACHE="/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33"
VIDEO_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
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
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

RUN_TAG="v514_narrow_lowrate"
OUT_DIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

SCRIPT_PATH="${WORKDIR}/scripts/sbatch/v514_narrow_lowrate.sbatch"
ATTEMPT="${ATTEMPT:-1}"; MAX_ATTEMPTS=500
if [ -f "${OUT_DIR}/DONE" ]; then
    echo "[chain] ${OUT_DIR}/DONE present — training complete; exiting."; exit 0
fi
if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[chain] attempt ${ATTEMPT}: queueing successor (afterany:${SLURM_JOB_ID})"
    sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((ATTEMPT + 1))" \
           "${SCRIPT_PATH}" || echo "[chain] successor submit failed (non-fatal)"
fi

python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --video_dir "${VIDEO_DIR}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 0 \
    --batch_size 2 --num_workers 4 \
    --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
    --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 64 --d_dyn 64 --d_state 32 \
    --enc_hidden_ch 128 --dec_hidden_ch 256 \
    --lambda_recon 0.25 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lambda_attrs 0.5 \
    --lambda_mae 0.5 --mask_ratio 0.5 --aux_hidden_ch 128 \
    --dino_cache_dir "${DINO_CACHE}" \
    --infonce_temperature 0.1 \
    --anneal_pixel \
    --early_stop_patience 0 \
    --val_frac 0.2 --val_every 5 \
    --log_every 1 --ckpt_every 10 \
    --resume auto \
    --lambda_pixel 5.0 \
    --vae_dtype bfloat16 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}" \
    --device cuda

echo "Done at $(date)"
