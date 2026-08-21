#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=07:30:00 --mem-per-gpu=120G --job-name=clevrer_baselines
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --exclude=atl1-1-03-007-29-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Leg-1 (compressive+semantic): strong pretrained video encoders on the SAME
# CLEVRER presence-mAP val split / protocol as ours, full-dim AND PCA-96 matched.
set -e
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga
cd "$WD"
BAD="atl1-1-03-007-29-0"   # known bad-ECC node; site plugin ignores --exclude
if [ "$(hostname)" = "$BAD" ]; then
  echo "[guard] landed on bad node $BAD -> resubmitting and exiting"
  sbatch "$WD/scripts/sbatch/clevrer_baselines.sbatch"; exit 0
fi
source ~/.bashrc 2>/dev/null || true
conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines
mkdir -p "$OUT"
python -u scripts/probes/clevrer_baselines_probe.py \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
  --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
  --models videomae videoflextok dinov2 \
  --max_videos 3000 --val_frac 0.1 --pca_dims 96 --epochs 400 \
  --out "$OUT/clevrer_pretrained_baselines.json"
echo CLEVRER_BASELINES_SBATCH_DONE
