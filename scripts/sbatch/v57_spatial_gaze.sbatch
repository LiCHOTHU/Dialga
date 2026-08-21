#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v57_spatial_gaze
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.7 SPATIAL-GRID z_static + GAZE (tube masking) — third arm.
# Identical to v57_spatial_clevrer except the MAE masking mechanism:
#   --mask_tube --mask_ratio 0.7 --mask_pool_frac 0.9  (v5.4 Stage A gaze:
#   saliency-selected whole spatial columns masked across ALL t, killing the
#   temporal-copy shortcut of per-cell masking). 3-arm readout:
#   v55_pool_mean (control) vs v57_spatial (grid) vs this (grid+gaze).
# Original v5.7 rationale below.
#
# v5.7 SPATIAL-GRID z_static — the fix for the global-pool bottleneck.
# Budget-matched apple-to-apple pair with v55_pool_mean (already done, ep200):
# the ONLY differences are pool_type / decoder_type. Both hold the SAME 96-float
# static budget:
#   mean arm (v55_pool_mean): pool=mean,    d_static=96 -> 1 global vector (96)
#   grid arm (this run)     : pool=spatial, 4x4 x 6ch   -> image-like grid  (96)
# The v5.6 latent-size sweep proved the CLEVRER blur is rate-INDEPENDENT (2/4/8x
# rate flat), isolating the global mean-pool (which destroys "what is where") as
# the bottleneck. This run keeps an image-like static grid (PV-VAE / PVDM /
# VidTwin / MAETok principle) so local identity survives per region. Everything
# else identical (d_dyn, hidden, lr, batch, epochs, full pixdom+MAE+anneal
# recipe, same data). Readout: val recon / obj-PSNR / attrs-CE / z_static_std vs
# the mean-pool control at matched budget.  Self-chains across the 8h embers wall.

# set -x through the preamble: the Jul-09 chained successors (10889166/10891142)
# died at exactly 2:26 with ZERO stdout — i.e. before the first echo — so the
# preamble needs to be traced to stderr to be diagnosable at all.
set -eo pipefail
set -x
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

RUN_TAG="v57_spatial_gaze"
OUT_DIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true
set +x

SCRIPT_PATH="${WORKDIR}/scripts/sbatch/v57_spatial_gaze.sbatch"
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
    --d_static 96 --d_dyn 96 --d_state 48 \
    --enc_hidden_ch 192 --dec_hidden_ch 384 \
    --pool_type spatial --static_grid 4 \
    --decoder_type spatial --dec_depth 3 \
    --lambda_recon 0.25 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lambda_attrs 0.5 \
    --lambda_mae 0.5 --mask_ratio 0.7 --mask_pool_frac 0.9 --mask_tube --aux_hidden_ch 128 \
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
