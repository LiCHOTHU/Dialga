#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=02:00:00 --mem-per-gpu=100G --job-name=disent_headtohead
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# DISENTANGLEMENT head-to-head (the reframed paper's headline): our factorized model
# vs the entangled AE (--shared_trunk). For each, z_static identity mAP (want HIGH)
# and z_dyn identity-LEAKAGE mAP (want LOW). Clean factorization = big gap; entangled
# = identity leaks into the motion code (small gap). Guarded + self-resubmit on bad node.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node -> sleep+resubmit"; sleep 60; sbatch "$WD/scripts/sbatch/disent_headtohead.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
for tag in clevrer_distill clevrer_entangled; do
  CK=$(ls $CED/$tag/v5_best.pt $CED/$tag/ckpt_ep0200.pt $CED/$tag/last.pt 2>/dev/null | head -1)
  echo "==================== $tag  ($CK) ===================="
  [ -z "$CK" ] && { echo "  no ckpt"; continue; }
  python -u scripts/probes/clevrer_semantic_probe.py --ckpt "$CK" \
    --cache_dir "$WAN" --max_videos 1500 --out "$OUT/disent_${tag}.json" 2>&1 \
    | grep -E "\[Q1\]|\[Q2 leakage\]|z_static mAP|z_dyn mAP"
done
echo DISENT_HEADTOHEAD_DONE
