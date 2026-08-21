#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=05:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=rd_table
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# THE PAPER'S MAIN TABLE — rate-distortion vs standard codecs at matched bitrate.
#
# We lose to our own frozen Wan-VAE by ~15 dB, so distortion is not the story;
# RATE is. This table asks the only question that can carry a reconstruction
# paper: at ~23 kbps, do we beat H.264/H.265?
#
# The five model rows are WIDTH-MATCHED (enc_hidden_ch=192 / dec_hidden_ch=384)
# so the curve varies rate and ONLY rate:
#   v514_wide_lowrate    64 + 9*64  + 4 =  644 floats -> 15.6 kbps
#   v55_pool_mean        96 + 9*96  + 4 =  964 floats -> 23.4 kbps
#   v56_latent_x2       192 + 9*192 + 4 = 1924 floats -> 46.7 kbps
#   v56_latent_x4       384 + 9*384 + 4 = 3844 floats -> 93.3 kbps
#   v56_latent_x8       768 + 9*768 + 4 = 7684 floats -> 186  kbps
# (v514_narrow_* are enc128 and are deliberately EXCLUDED — they confound rate
#  with capacity. They belong to the separate params-vs-rate ablation.)
#
# Block A = ep50, all five rates (epoch-matched; the only epoch every arm has).
# Block B = ep200 for the three arms that reached it (converged curve).
# Codec rows are pinned to the 964-float rate (23.4 kbps) in both blocks.
#
# Bit accounting bills our latents at fp32 (32 b/float) — the worst case for us,
# since we do not quantise; and the codecs keep their container overhead free.

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
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
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/storage/project/r-agarg35-0/lwang831/torch_home"   # LPIPS alexnet weights
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
OUT="${WORKDIR}/outputs/rd_table"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "############ BLOCK A: epoch-matched ep50 R-D curve + codecs ############"
python -u scripts/probes/rd_table.py \
    --ckpts \
        rate0644_ep50="${CEDAR}/v514_wide_lowrate/ckpt_ep0050.pt" \
        rate0964_ep50="${CEDAR}/v55_pool_mean/ckpt_ep0050.pt" \
        rate1924_ep50="${CEDAR}/v56_latent_x2/ckpt_ep0050.pt" \
        rate3844_ep50="${CEDAR}/v56_latent_x4/ckpt_ep0050.pt" \
        rate7684_ep50="${CEDAR}/v56_latent_x8/ckpt_ep0050.pt" \
    --cache_dir "${WAN_CACHE}" --video_dir "${VIDEO_DIR}" \
    --n_videos 32 --codec_kbps 23.4 \
    --out "${OUT}/rd_ep50.json"

echo
echo "############ BLOCK B: converged (ep200) curve + codecs ############"
python -u scripts/probes/rd_table.py \
    --ckpts \
        rate0964_ep200="${CEDAR}/v55_pool_mean/ckpt_ep0200.pt" \
        rate1924_ep200="${CEDAR}/v56_latent_x2/ckpt_ep0200.pt" \
        rate3844_ep200="${CEDAR}/v56_latent_x4/ckpt_ep0200.pt" \
    --cache_dir "${WAN_CACHE}" --video_dir "${VIDEO_DIR}" \
    --n_videos 32 --codec_kbps 23.4 \
    --out "${OUT}/rd_ep200.json"

echo "RD_TABLE_SBATCH_DONE"
