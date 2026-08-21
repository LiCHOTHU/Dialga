#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=07:30:00 --mem-per-gpu=120G --job-name=clevrer_rollout_baselines
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --exclude=atl1-1-03-007-29-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Leg-3 (predictable): learned forward model per frozen representation, skill vs copy-last.
set -e
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga
cd "$WD"
source ~/.bashrc 2>/dev/null || true
conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
CK=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/clevrer_distill/v5_best.pt
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines
mkdir -p "$OUT"
python -u scripts/probes/clevrer_rollout_baselines.py \
  --ckpt "$CK" \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
  --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
  --methods ours videomae videoflextok dinov2 \
  --max_videos 2500 --epochs 200 \
  --out "$OUT/clevrer_rollout_baselines.json"
echo CLEVRER_ROLLOUT_SBATCH_DONE
