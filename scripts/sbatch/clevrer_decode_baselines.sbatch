#!/bin/bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=07:30:00 --mem-per-gpu=120G --job-name=clevrer_decode_baselines
#SBATCH --qos=embers --account=gts-agarg35-ideas_l40s
#SBATCH --exclude=atl1-1-03-007-29-0
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# Leg-2 (decodable): equal-capacity decoder head from each FROZEN representation
# back to the Wan latent -> val latent-MSE + Wan-decoded pixel PSNR. Same split/head.
set -e
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga
cd "$WD"
BAD="atl1-1-03-007-29-0"   # known bad-ECC node; site plugin ignores --exclude
HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*)
  echo "[guard] landed on bad node $BAD (host=$HN) -> sleeping then resubmitting"
  sleep 90   # let the bad node get picked up by another job before we re-queue
  sbatch "$WD/scripts/sbatch/clevrer_decode_baselines.sbatch"; exit 0;;
esac
source ~/.bashrc 2>/dev/null || true
conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WD" TORCHDYNAMO_DISABLE=1 MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/mpl
mkdir -p "$MPLCONFIGDIR"
CK=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/clevrer_distill/v5_best.pt
OUT=/storage/project/r-agarg35-0/lwang831/outputs/dialga/clevrer_baselines
mkdir -p "$OUT"
# embers preempts this job repeatedly; self-chain until the JSON exists, and keep it
# small enough to finish inside a preemption window.
SP="$WD/scripts/sbatch/clevrer_decode_baselines.sbatch"; A="${ATTEMPT:-1}"
[ -f "$OUT/clevrer_decode_baselines_matched.json" ] && { echo "already done"; exit 0; }
[ "$A" -lt 40 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" "$SP" || true
python -u scripts/probes/clevrer_decode_baselines.py \
  --ckpt "$CK" \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 \
  --video_root /storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video \
  --methods ours videomae videoflextok dinov2 \
  --max_videos 600 --epochs 80 --pca_to 768 352 \
  --out "$OUT/clevrer_decode_baselines_matched.json"
echo CLEVRER_DECODE_SBATCH_DONE
