#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v512_clevrer_pixdom_mae
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# FINAL-RECIPE candidate: pixel-dominant reconstruction (v512_clevrer_pixdom)
# + MAETok masked-semantic prediction (v513). Combines the two levers that each
# won on their own axis:
#   - pixel-dominant (lambda_recon 0.25, lambda_pixel 5.0): reaches the pixel
#     floor ~15x faster; both Wan VAE halves stay FROZEN.
#   - MAE (lambda_mae 0.5, mask_ratio 0.5): masked DINOv2 patch-feature
#     prediction; fixes the val-attrs generalization failure (val attrs held
#     ~3.2 at ep20 vs big's blow-up to 12.1).
# These are orthogonal: pixel-dom decides which space the recon gradient
# optimizes; MAE enriches the representation. Rate unchanged: z = 964 floats.
#
# Loss-balance annealing (--anneal_pixel): the run STARTS latent-only
# (lambda_recon=1.0 anchor, lambda_pixel=0) so it first establishes Wan-latent
# fidelity, then — once the train L_recon goes flat (per-epoch improvement
# <0.5% for 6 epochs) — linearly ramps over 8 epochs to the pixel-dominant
# targets below (lambda_recon 0.25, lambda_pixel 5.0). The --lambda_recon /
# --lambda_pixel flags are the PHASE-B targets. Curriculum is stateless
# (trigger recomputed from train-recon history), so it survives self-chaining.
# Fresh run (new loss config). Self-chains across the 8h embers wall.

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

RUN_TAG="v512_clevrer_pixdom_mae"
OUT_DIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

SCRIPT_PATH="${WORKDIR}/scripts/sbatch/v512_clevrer_pixdom_mae.sbatch"
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
