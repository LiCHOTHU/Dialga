#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v512e1_frz
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Exp 1/4: Wan VAE FROZEN (control). Matched to e2/e3 for a FAIR comparison:
# identical losses (lambda_recon=0, pixel-space L_pixel only) and B=2/1000 vids/
# 200 ep — the ONLY difference vs e2/e3 is that NO VAE half is unfrozen here.
# --lambda_pixel>0 loads the Wan VAE frozen, so our model is supervised in pixel
# space through the frozen decoder exactly like e3 (but the decoder can't adapt).
# Needs L40S (48GB): the frozen-decoder pixel path retains decode activations, so
# the old V100-16GB control config would OOM.

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
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
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

RUN_TAG="v512_e1_frozen"
# STABLE out dir (no timestamp) so --resume auto finds last.pt across requeues.
OUT_DIR="/storage/scratch1/8/lwang831/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

# --- self-chaining: survive the 8h embers wall by requeueing until DONE ---
# Each window queues its successor (afterany on this job) BEFORE training, then
# trains with --resume auto. When the trainer reaches the final epoch it writes
# ${OUT_DIR}/DONE; the next queued job sees it and exits without chaining further.
SCRIPT_PATH="${WORKDIR}/scripts/sbatch/v512_e1_frozen.sbatch"
ATTEMPT="${ATTEMPT:-1}"; MAX_ATTEMPTS=20
if [ -f "${OUT_DIR}/DONE" ]; then
    echo "[chain] ${OUT_DIR}/DONE present — training complete; exiting."; exit 0
fi
if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[chain] attempt ${ATTEMPT}: queueing successor (afterany:${SLURM_JOB_ID})"
    sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((ATTEMPT + 1))" \
           "${SCRIPT_PATH}" || echo "[chain] successor submit failed (non-fatal)"
else
    echo "[chain] reached MAX_ATTEMPTS=${MAX_ATTEMPTS}; not requeueing."
fi

python -u scripts/train_v5.py \
    --cache_dir "${WAN_CACHE}" \
    --video_dir "${VIDEO_DIR}" \
    --out_dir "${OUT_DIR}" \
    --max_videos 1000 \
    --batch_size 2 --num_workers 4 \
    --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
    --lr 5e-4 --vae_lr 1e-5 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 64 --d_dyn 64 --d_state 32 \
    --enc_hidden_ch 128 --dec_hidden_ch 256 \
    --lambda_recon 0.0 --lambda_pred 0.0 --lambda_fwd 0.1 --lambda_consist 1.0 \
    --lambda_event_aux 0.1 --lambda_gate 0.1 \
    --lambda_attrs 0.05 \
    --infonce_temperature 0.1 \
    --early_stop_patience 0 \
    --val_frac 0.2 --val_every 5 \
    --log_every 1 --ckpt_every 10 \
    --resume auto \
    --lambda_pixel 5.0 \
    --vae_dtype bfloat16 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}" \
    --device cuda

echo "Done at $(date)"
