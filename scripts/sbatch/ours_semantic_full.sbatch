#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=05:00:00 --mem-per-gpu=120G --job-name=ours_semantic_full
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Clean, full-data ours+wan semantic probe (distilled ckpt) so the Q1 table's
# ours/wan-mean/pca/random/dino rows are internally consistent; DINOv2 here (~0.974)
# anchors to the external-baseline run's DINOv2 (0.977), making the two combinable.
set -e
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga
cd "$WD"
BAD="atl1-1-03-007-29-0"
HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*)
  echo "[guard] bad node -> sleep+resubmit"; sleep 90
  sbatch "$WD/scripts/sbatch/ours_semantic_full.sbatch"; exit 0;;
esac
source ~/.bashrc 2>/dev/null || true
conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
CK=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/clevrer_distill/v5_best.pt
DINO=/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines
mkdir -p "$OUT"
python -u scripts/probes/baseline_probe_table.py --ckpt "$CK" --dino_cache_dir "$DINO" \
  --epochs 400 --pca_dim 96 --out "$OUT/ours_semantic_full.json"
echo OURS_SEMANTIC_FULL_DONE
