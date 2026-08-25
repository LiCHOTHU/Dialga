#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=05:00:00 --mem-per-gpu=120G --job-name=motion_probe
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# z_dyn motion probe (NO pooling): (A) motion content R^2, (B) spurious-correlation
# OOD test where factored z_dyn should beat ENTANGLED baselines (VideoMAE / full latent).
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); case "$HN" in *"$BAD"*) echo guard; sleep 60; sbatch "$WD/scripts/sbatch/motion_probe.sbatch"; exit 0;; esac
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
CED=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs
WAN=/storage/scratch1/8/lwang831/cache/wan_10000vid_W33
VID=/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines; mkdir -p "$OUT"
# baseline model (clevrer_distill) = strong z_dyn (the hero). RGB carries to both models.
TAGS="${TAGS:-clevrer_distill}"
RGB="${RGB-videomae videoflextok}"
for tag in $TAGS; do
  CK=$(ls $CED/$tag/ckpt_ep0200.pt $CED/$tag/v5_best.pt 2>/dev/null | head -1)
  echo "==================== $tag ($CK) [rgb='$RGB'] ===================="
  RGB_ARGS=""; [ -n "$RGB" ] && RGB_ARGS="--rgb_models $RGB --video_root $VID"
  python -u scripts/probes/motion_probe.py --ckpt "$CK" --cache_dir "$WAN" \
    --max_videos 2500 $RGB_ARGS --bias_sweep 0.80 0.90 0.95 0.98 \
    --out "$OUT/motion_${tag}.json" 2>&1 | grep -E "\[data\]|==|acc_|mean_speed|\[rgb\]|MOTION"
done
echo MOTION_FULL_DONE
