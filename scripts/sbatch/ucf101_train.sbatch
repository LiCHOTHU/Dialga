#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=ucf101_train
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.9 FINAL-CONFIG training on Something-Something-v2 (motion-centric real
# video, the standard video-representation benchmark). Same architecture as
# v59_clevrer: spatial static grid + SPATIAL z_dyn (--dyn_spatial), rebalanced
# lambda_pred=0.1. No camera pose (SSv2 has no pose) and no CLEVRER-style
# attrs/event/gate supervision (--lambda_* 0), mirroring the DROID no-GT path.
# Self-chains across the wall via --resume auto.
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CACHE="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/cache/ucf101_train_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"; export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

RUN_TAG="ucf101_train"
OUT_DIR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

SCRIPT_PATH="${WORKDIR}/scripts/sbatch/ucf101_train.sbatch"
ATTEMPT="${ATTEMPT:-1}"; MAX_ATTEMPTS=500
if [ -f "${OUT_DIR}/DONE" ]; then
    echo "[chain] DONE present — training complete; exiting."; exit 0
fi
if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[chain] attempt ${ATTEMPT}: queueing successor (afterany:${SLURM_JOB_ID})"
    sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((ATTEMPT + 1))" \
           "${SCRIPT_PATH}" || echo "[chain] successor submit failed (non-fatal)"
fi

python -u scripts/train_v5.py \
    --dataset ssv2 --cache_dir "${CACHE}" --out_dir "${OUT_DIR}" \
    --max_videos 0 --batch_size 8 --num_workers 6 \
    --epochs 200 --stage1_epochs 5 --stage2_epochs 195 \
    --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 96 --d_dyn 256 --d_state 48 \
    --enc_hidden_ch 192 --dec_hidden_ch 384 \
    --static_grid 4 --pool_type spatial --decoder_type spatial \
    --dyn_spatial --dyn_grid 8 \
    --lambda_recon 1.0 --lambda_pred 0.1 --lambda_fwd 0.1 --lambda_consist 1.0 \
    --lambda_attrs 0.0 --lambda_event_aux 0.0 --lambda_gate 0.0 \
    --lambda_pixel 0.0 --lambda_mae 0.0 \
    --infonce_temperature 0.1 --early_stop_patience 0 \
    --val_frac 0.1 --val_every 5 \
    --log_every 20 --ckpt_every 10 --resume auto \
    --vae_dtype bfloat16 \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}" \
    --device cuda

echo "Done at $(date)"
