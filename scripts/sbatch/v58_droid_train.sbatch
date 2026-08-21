#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=08:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v58_droid_train
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.8 DROID stage — STEP 2/3: train ONE arm on the DROID wrist cache.
#   ARM=on     : encoder is GIVEN the real 6-DoF EE (=camera) pose and must
#                invert it (decoder re-applies it) -> camera-canonical z_dyn.
#   ARM=blind  : identical capacity + identical pose plumbing, pose WITHHELD
#                (zeros). Control. Camera motion has nowhere to go but z_dyn.
# DROID has no attribute/collision labels -> those losses are OFF. cam_inv is
# inert on real data (no un-warped reference); the readout is OFFLINE (step 3).
# Set ARM via env var. Latent-recon config (pool=mean, decoder=linear).
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
DROID_CACHE="/storage/scratch1/8/lwang831/cache/droid_wrist_W33"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
ARM="${ARM:-on}"
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
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"
nvidia-smi || true

COMMON=(
    --dataset droid --cam_source real
    --cache_dir "${DROID_CACHE}"
    --batch_size 16 --num_workers 6
    --epochs 200 --stage1_epochs 5 --stage2_epochs 195
    --lr 3e-4 --lr_schedule cosine --weight_decay 1e-3
    --d_static 96 --d_dyn 96 --d_state 48
    --enc_hidden_ch 192 --dec_hidden_ch 384
    --pool_type mean --decoder_type linear
    --lambda_recon 1.0 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0
    --lambda_attrs 0.0 --lambda_event_aux 0.0 --lambda_gate 0.0
    --infonce_temperature 0.1
    --lambda_pixel 0.0 --lambda_mae 0.0
    --early_stop_patience 0 --val_frac 0.2
    --log_every 10 --ckpt_every 50 --resume auto
    --use_camera_pose --pose_inject concat --d_pose 32 --pose_dim 6 --pose_n_trans 3
    --lambda_cam_inv 1.0
    --device cuda
)

if [ "${ARM}" = "blind" ]; then
    OUT="${CEDAR}/v58_droid_blind"; EXTRA=(--cam_pose_blind); RUN="v58_droid_blind"
else
    OUT="${CEDAR}/v58_droid_on";    EXTRA=();                 RUN="v58_droid_on"
fi
mkdir -p "${OUT}"
echo "############ DROID train ARM=${ARM} -> ${OUT} ############"
python -u scripts/train_v5.py "${COMMON[@]}" "${EXTRA[@]}" \
    --out_dir "${OUT}" \
    --wandb_project dialga --wandb_run_name "${RUN}"

echo "V58_DROID_TRAIN_${ARM}_DONE at $(date)"
