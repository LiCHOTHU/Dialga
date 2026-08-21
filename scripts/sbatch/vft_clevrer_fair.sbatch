#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=02:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=vft_fair
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# FAIR VFT vs OURS on CLEVRER — reconstruction at matched embedding size.
# Runs the pretrained VideoFlexTok (zero-shot) on the SAME held-out val clips our
# v57 model is scored on (val split seed=42, val_frac=0.2 -> video ids below),
# sweeping num_keep_tokens so we read PSNR AT our latent budget.
#   VFT full rate  : K=256 registers = 1280 tokens/17f
#   our budget     : ~497 float-slots/17f  ->  K~=100 registers (500 tokens)
# Bracketed by K=128 / K=64 (VFT's power-of-2 nested-dropout training grid).
# Ours on these same 12 clips = 31.4 dB whole-frame (psnr_table, already run).
# CAVEAT (cannot be removed by "running"): VFT is ZERO-SHOT / out-of-distribution;
# our model TRAINED on the CLEVRER distribution. Closing that gap needs TRAINING a
# flow decoder on CLEVRER (repo is inference-only) — a separate build, not a run.

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
ENV="/storage/project/r-agarg35-0/lwang831/conda/envs/videoflextok"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc >/dev/null 2>&1

# HF cache (all four vars — .bashrc otherwise overrides HF_HOME)
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"

# scratch/compile caches -> project tmp (nodes block /triton, /tmp unreliable)
export TMPDIR="${PROJECT_TMP}"
export MPLCONFIGDIR="${TMPDIR}/mpl";              mkdir -p "${MPLCONFIGDIR}"
export TRITON_CACHE_DIR="${TMPDIR}/triton";       mkdir -p "${TRITON_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}"
export XDG_CACHE_HOME="${TMPDIR}/xdg";            mkdir -p "${XDG_CACHE_HOME}"
export TORCHINDUCTOR_COMPILE_THREADS=1

# held-out val clip ids (match v57 val split: seed=42, val_frac=0.2)
VAL_IDS="4 7 12 19 32 37 40 45 49 51 57 63"

conda run -p "${ENV}" python -u scripts/probes/vft_rate_sweep.py \
    --video_ids ${VAL_IDS} \
    --keeps 256 128 100 64 32 \
    --save_keeps 256 100 \
    --out "${WORKDIR}/outputs/vft_clevrer_fair" 2>&1

echo "VFT_FAIR_SBATCH_DONE"
