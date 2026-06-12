#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=64G
#SBATCH --job-name=libero_v512_big
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v2 of LIBERO training. Latent-space supervision only (NO pixel loss — pixel
# loss was costly without changing the representation story; the v1 action
# probe already shows z_dyn is information-rich enough for the downstream).
#
# vs v1:
#   - enc_h 128 -> 192, dec_h 256 -> 384 (1.5x internal capacity)
#   - d_static, d_dyn 64 -> 96, d_state 32 -> 48 (1.5x representation slots)
#   - lr 5e-4 -> 3e-4 (slightly smaller; expect smoother trajectories at scale)
#   - lambda_fwd 0.1 -> 0.3 (v1 had L_fwd DIVERGING from 0.021 ep30 to 0.049
#                            ep200 because λ_fwd was too small relative to
#                            L_recon + L_consist's pull on z_dyn)
#   - epochs 200 -> 300 (1 sbatch easily fits this; v1 converged in <30 min)

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
LIBERO_CACHE="/storage/scratch1/8/lwang831/cache/libero_90_wan_W33"
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

RUN_TAG="libero_v512_big"
OUT_DIR="/storage/scratch1/8/lwang831/dialga_outputs/${RUN_TAG}"
mkdir -p "${OUT_DIR}"
echo "OutDir: ${OUT_DIR}"; nvidia-smi || true

# Self-chain (matches v1 + CLEVRER e1 sbatch).
SCRIPT_PATH="${WORKDIR}/scripts/sbatch/libero_train_v512_big.sbatch"
ATTEMPT="${ATTEMPT:-1}"; MAX_ATTEMPTS=6
if [ -f "${OUT_DIR}/DONE" ]; then
    echo "[chain] ${OUT_DIR}/DONE present — training complete; exiting."; exit 0
fi
if [ "${ATTEMPT}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "[chain] attempt ${ATTEMPT}: queueing successor (afterany:${SLURM_JOB_ID})"
    sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((ATTEMPT + 1))" \
           "${SCRIPT_PATH}" || echo "[chain] successor submit failed (non-fatal)"
fi

python -u scripts/train_v5_libero.py \
    --cache_dir "${LIBERO_CACHE}" \
    --out_dir "${OUT_DIR}" \
    --batch_size 16 --num_workers 4 \
    --epochs 300 --stage1_epochs 5 \
    --lr 3e-4 --lr_schedule constant --weight_decay 1e-3 \
    --d_static 96 --d_dyn 96 --d_state 48 \
    --enc_hidden_ch 192 --dec_hidden_ch 384 \
    --lambda_recon 1.0 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0 \
    --lambda_clip 0.5 \
    --lambda_event_aux 0.0 --lambda_gate 0.0 \
    --lambda_pixel 0.0 \
    --infonce_temperature 0.1 \
    --n_holdout_tasks 10 --task_holdout_seed 0 \
    --val_every_k_episodes 10 --require_chunks_per_episode 3 \
    --val_every 5 --log_every 1 --ckpt_every 20 \
    --resume auto \
    --wandb_project dialga --wandb_run_name "${RUN_TAG}" \
    --device cuda

echo "Done at $(date)"
