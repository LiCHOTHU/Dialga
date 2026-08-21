#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=03:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v58_droid_fast
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Self-contained: DROID cache already extracted (1335 chunks). Train ON + BLIND
# (num_workers=0, wandb DISABLED -> no NFS temp writes) -> readout + pixel PSNR.
# Runs on its OWN allocation so nothing external can SIGTERM it.
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
PT="/storage/project/r-agarg35-0/lwang831/tmp"
CACHE="${PT}/droid_fast_cache"
OUT_ON="${PT}/droid_on_v3"
OUT_BLIND="${PT}/droid_blind_v3"
EPOCHS="${EPOCHS:-150}"
cd "${WORKDIR}"

source ~/.bashrc; conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"
export TMPDIR="${PT}"; export TEMP="${PT}"; export TMP="${PT}"
export MPLCONFIGDIR="${PT}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}"; export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export WANDB_MODE=disabled
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
nvidia-smi || true

TRAIN=(--dataset droid --cam_source real --cache_dir "${CACHE}"
    --batch_size 16 --num_workers 0
    --epochs "${EPOCHS}" --stage1_epochs 5 --stage2_epochs $((EPOCHS-5))
    --lr 3e-4 --lr_schedule cosine --weight_decay 1e-3
    --d_static 96 --d_dyn 96 --d_state 48 --enc_hidden_ch 192 --dec_hidden_ch 384
    --pool_type mean --decoder_type linear
    --lambda_recon 1.0 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0
    --lambda_attrs 0.0 --lambda_event_aux 0.0 --lambda_gate 0.0
    --infonce_temperature 0.1 --lambda_pixel 0.0 --lambda_mae 0.0
    --early_stop_patience 0 --val_frac 0.2 --log_every 25 --ckpt_every 1000 --resume auto
    --use_camera_pose --pose_inject concat --d_pose 32 --pose_dim 6 --pose_n_trans 3
    --lambda_cam_inv 1.0 --device cuda)

echo "==================== [1/3] TRAIN pose-ON ===================="
python -u scripts/train_v5.py "${TRAIN[@]}" --out_dir "${OUT_ON}" --wandb_run_name droid_on_v3

echo "==================== [2/3] TRAIN pose-BLIND ===================="
python -u scripts/train_v5.py "${TRAIN[@]}" --cam_pose_blind --out_dir "${OUT_BLIND}" --wandb_run_name droid_blind_v3

echo "==================== [3/3] READOUT + PSNR ===================="
python -u scripts/probes/droid_camera_readout.py \
    --cache_dir "${CACHE}" --val_frac 0.2 --seed 42 --psnr \
    --ckpt on:"${OUT_ON}/v5.pt" --ckpt blind:"${OUT_BLIND}/v5.pt" \
    --out_json "${PT}/droid_fast_readout.json"

echo "V58_DROID_FAST_DONE at $(date)"
