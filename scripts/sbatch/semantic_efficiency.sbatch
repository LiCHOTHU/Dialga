#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=120G --job-name=semantic_efficiency
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# 'More semantic AND efficient' vs the FAIR peer class (same frozen-VAE latent input,
# matched rate): rate-semantic curve + label-efficiency curve. No foundation encoders.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node -> sleep+resubmit"; sleep 60; sbatch "$WD/scripts/sbatch/semantic_efficiency.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
CK=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/clevrer_distill/v5_best.pt
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
python -u scripts/probes/semantic_efficiency.py --ckpt "$CK" \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
  --epochs 300 --pca_dims 48 96 192 352 768 \
  --label_fracs 0.02 0.05 0.1 0.25 0.5 1.0 \
  --out "$OUT/semantic_efficiency.json"
echo SEMANTIC_EFFICIENCY_SBATCH_DONE
