#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=07:30:00 --mem-per-gpu=120G --job-name=baseline_sem_labelcurve
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# STRONG-baseline semantics comparison (not PCA): VideoMAE / VideoFlexTok / DINOv2 on
# CLEVRER attribute mAP, full-dim + PCA-96 + LABEL-EFFICIENCY curve, same probe/split
# as ours. Lets us compare ours@96 vs real learned encoders across the label axis.
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node -> sleep+resubmit"; sleep 60; sbatch "$WD/scripts/sbatch/baseline_semantics_labelcurve.sbatch"; exit 0;; esac
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines
[ -f "$OUT/baseline_semantics_labelcurve.json" ] && { echo "already done"; exit 0; }
SP="$WD/scripts/sbatch/baseline_semantics_labelcurve.sbatch"; A="${ATTEMPT:-1}"
[ "$A" -lt 30 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" "$SP" || true
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR" "$OUT"
python -u scripts/probes/clevrer_baselines_probe.py \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
  --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
  --models videomae videoflextok dinov2 \
  --max_videos 2000 --pca_dims 96 --epochs 400 \
  --label_fracs 0.02 0.05 0.1 0.25 0.5 1.0 \
  --out "$OUT/baseline_semantics_labelcurve.json"
echo BASELINE_SEM_LABELCURVE_DONE
