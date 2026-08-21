#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=07:00:00
#SBATCH --mem-per-gpu=120G
#SBATCH --job-name=rfvd_table
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# E1 — THE GATE. rFVD is the metric VideoFlexTok / FlexTok / PV-VAE actually rank on.
# We have never measured it. See EXPERIMENT_PLAN_v7.md.
#
# Why this decides the paper:
#   VideoFlexTok headlines rFVD and reports NO PSNR at all. FlexTok headlines rFID and
#   buries PSNR in appendix Table 8, where its best model at full 256-token rate scores
#   17.70 dB and they ship it. A rectified-flow decoder hallucinates sharp detail: tanks
#   PSNR, wins FVD. Our deterministic decoder does the reverse. So our PSNR-based verdict
#   ("H.264 beats us by 10 dB, the recon story is dead") was measured on an axis this
#   field abandoned — and none of the three papers compares to a codec at all.
#
# DECISION RULE:
#   rFVD VARIES with rate  -> we have a real rate axis; the flat PSNR curve was an
#                             artifact of the metric. Headline rFVD, proceed to the
#                             flow-decoder head-to-head (E2).
#   rFVD FLAT too          -> the model genuinely ignores its bits. No tokenizer paper.
#                             Fall back to the embedding claim and report this as the
#                             central negative result.
#
# Layout mirrors FlexTok appendix Table 8: floats | bytes | rFVD | PSNR | SSIM | LPIPS,
# plus the Wan-VAE roundtrip ceiling and H.264/H.265 rows at matched bitrate.
# FVD = cd-fvd (I3D). Verified calibration: identical -> 0.0, same-process -> 2.93,
# vs pure noise -> 3246.
#
# Block A = epoch-matched ep50, all five width-matched rates (644..7684).
# Block B = converged ep200, the three arms that got there.
# 512 val videos, one 33-frame observed chunk each (FVD is distributional; stats stream).

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
export TORCH_HOME="/storage/project/r-agarg35-0/lwang831/torch_home"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${TORCH_HOME}"
OUT="${WORKDIR}/outputs/rfvd_table"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "############ BLOCK A: epoch-matched ep50, five rates ############"
python -u scripts/probes/rfvd_table.py \
    --ckpts \
        rate0644_ep50="${CEDAR}/v514_wide_lowrate/ckpt_ep0050.pt" \
        rate0964_ep50="${CEDAR}/v55_pool_mean/ckpt_ep0050.pt" \
        rate1924_ep50="${CEDAR}/v56_latent_x2/ckpt_ep0050.pt" \
        rate3844_ep50="${CEDAR}/v56_latent_x4/ckpt_ep0050.pt" \
        rate7684_ep50="${CEDAR}/v56_latent_x8/ckpt_ep0050.pt" \
    --cache_dir "${WAN_CACHE}" --video_dir "${VIDEO_DIR}" \
    --n_videos 512 --codec_kbps 23.4 --fvd_model i3d \
    --out "${OUT}/rfvd_ep50.json"

echo
echo "############ BLOCK B: converged ep200 ############"
python -u scripts/probes/rfvd_table.py \
    --ckpts \
        rate0964_ep200="${CEDAR}/v55_pool_mean/ckpt_ep0200.pt" \
        rate1924_ep200="${CEDAR}/v56_latent_x2/ckpt_ep0200.pt" \
        rate3844_ep200="${CEDAR}/v56_latent_x4/ckpt_ep0200.pt" \
    --cache_dir "${WAN_CACHE}" --video_dir "${VIDEO_DIR}" \
    --n_videos 512 --codec_kbps 23.4 --fvd_model i3d \
    --out "${OUT}/rfvd_ep200.json"

echo "RFVD_TABLE_SBATCH_DONE"
