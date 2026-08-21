#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=07:30:00 --mem-per-gpu=80G --job-name=ssv2_cache_32k
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
# DATA-SCALING experiment: Wan-encode 32k SSv2 videos (4x the 8k baseline) to test
# whether more data lifts SSv2 top-1. Resume-safe + self-chaining so it continues
# across the 8h embers wall until DONE. Consumed by ssv2_train on the bigger cache.
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"; cd "$WORKDIR"
BAD="atl1-1-03-007-29-0"; HN=$(hostname); HNS=$(hostname -s 2>/dev/null)
case "$HN$HNS" in *"$BAD"*) echo "[guard] bad node -> sleep+resubmit"; sleep 90; sbatch "$WORKDIR/scripts/sbatch/ssv2_cache_32k.sbatch"; exit 0;; esac
SSV2="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ssv2"
CACHE="/storage/scratch1/8/lwang831/cache/ssv2_32000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
source ~/.bashrc; conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" TMPDIR="$PROJECT_TMP" MPLCONFIGDIR="$PROJECT_TMP/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WORKDIR" TORCHDYNAMO_DISABLE=1
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$CACHE"
SP="$WORKDIR/scripts/sbatch/ssv2_cache_32k.sbatch"; A="${ATTEMPT:-1}"
[ -f "$CACHE/DONE" ] && { echo "cache DONE"; exit 0; }
# self-chain: keep a follow-on queued; it resumes (skips encoded) and exits early if DONE
[ "$A" -lt 60 ] && sbatch --dependency="afterany:${SLURM_JOB_ID}" --export="ALL,ATTEMPT=$((A+1))" "$SP" || true
python -u scripts/cache_wan_ssv2.py \
    --video_dir "$SSV2/videos_extracted/20bn-something-something-v2" \
    --label_json "$SSV2/labels.json" --split_json "$SSV2/train.json" \
    --max_videos 32000 --windows_per_video 3 --window_frames 33 --image_size 128 --seed 0 \
    --dtype float16 --device cuda --out_dir "$CACHE"
touch "$CACHE/DONE"
echo "Done at $(date)"
