#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=e3_dec
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Exp 3/4: Wan VAE DECODER unfrozen. Encoder stays frozen.
# Adds L_pixel = MSE(wan_dec(recon_latent), raw_pix) to the loss; trains the
# decoder to produce sharper pixels from our model's latents.
# Scoped: 1000 vids, 200 ep, B=4.

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

RUN_TAG="v512_e3_dec_unfrozen"
# STABLE out dir (no timestamp) so --resume auto finds last.pt across requeues.
OUT_DIR="/storage/scratch1/8/lwang831/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

# Requeue across the 8h embers wall is handled EXTERNALLY by
# scripts/sbatch/monitor_e2e3.sh — it resubmits this script whenever the job
# leaves the queue and ${OUT_DIR}/DONE is absent. --resume auto continues from
# last.pt. Keep the DONE-guard so a stray resubmit after completion no-ops.
if [ -f "${OUT_DIR}/DONE" ]; then
    echo "[done] ${OUT_DIR}/DONE present — training complete; exiting."; exit 0
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
    --unfreeze_vae_dec \
    --lambda_pixel 5.0 \
    --vae_dtype bfloat16 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}" \
    --device cuda

echo "Done at $(date)"
