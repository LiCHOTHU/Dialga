#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=03:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=v58_cam_pan_smoke
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# v5.8 CAMERA-PAN GATE (small first). 20-vid overfit, TWO arms in one job:
#   ARM A  camera-ON    : known synthetic 2D pan warps each chunk; the encoder is
#                         given the pose and must INVERT it (decode re-applies it).
#   ARM B  camera-BLIND : identical capacity + identical warp + identical
#                         invariance loss, but pose is WITHHELD (zeros). Control.
#
# GO/NO-GO read (the mechanism question, depth-free by construction):
#   cam_inv (invariance of z_static+z_dyn to the known camera) should DROP toward
#   0 for camera-ON and STAY HIGH for camera-BLIND. That gap = "known-pose
#   conditioning divides out the camera." recon (of the WARPED chunk) should also
#   be better for ON. If BLIND also drives cam_inv->0, the 2D pan is too easy
#   (pan-invariance learnable without pose) -> we still learn something and move
#   straight to the 3D DROID stage where parallax makes pose necessary.
#
# NOTE this is a MECHANISM gate: the camera path requires the latent-recon config
# (pool=mean, decoder=linear, lambda_pixel=0, lambda_mae=0). The full-recipe
# camera+pixel version is the DROID stage. Legacy path is untouched (d_pose=0).

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
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
nvidia-smi || true

COMMON=(
    --cache_dir "${WAN_CACHE}"
    --max_videos 20
    --batch_size 2 --num_workers 4
    --epochs 400 --stage1_epochs 5 --stage2_epochs 395
    --lr 3e-4 --lr_schedule constant --weight_decay 1e-3
    --d_static 96 --d_dyn 96 --d_state 48
    --enc_hidden_ch 192 --dec_hidden_ch 384
    --pool_type mean --decoder_type linear
    --lambda_recon 1.0 --lambda_pred 1.0 --lambda_fwd 0.3 --lambda_consist 1.0
    --lambda_event_aux 0.1 --lambda_gate 0.1 --lambda_attrs 0.5
    --infonce_temperature 0.1
    --lambda_pixel 0.0 --lambda_mae 0.0
    --early_stop_patience 0 --val_frac 0.0
    --log_every 10 --ckpt_every 200 --resume auto
    --use_camera_pose --pose_inject concat --d_pose 32 --pose_dim 3
    --lambda_cam_inv 1.0 --cam_max_shift 0.4 --cam_max_zoom 0.15
    --device cuda
)

echo "############ ARM A: camera-ON (pose given) ############"
OUT_A="${CEDAR}/v58_cam_pan_on"; mkdir -p "${OUT_A}"
python -u scripts/train_v5.py "${COMMON[@]}" \
    --out_dir "${OUT_A}" \
    --wandb_project dialga --wandb_run_name v58_cam_pan_on

echo
echo "############ ARM B: camera-BLIND (pose withheld, control) ############"
OUT_B="${CEDAR}/v58_cam_pan_blind"; mkdir -p "${OUT_B}"
python -u scripts/train_v5.py "${COMMON[@]}" --cam_pose_blind \
    --out_dir "${OUT_B}" \
    --wandb_project dialga --wandb_run_name v58_cam_pan_blind

echo "V58_CAM_PAN_SMOKE_DONE at $(date)"
