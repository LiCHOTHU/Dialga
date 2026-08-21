#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=80G --job-name=rollout_eval
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib TORCHDYNAMO_DISABLE=1 PYTHONPATH="$WD"
python -u scripts/probes/rollout_eval.py \
  --ckpt /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/v59_clevrer/v5_best.pt \
  --cache_dir /storage/scratch1/8/lwang831/cache/wan_10000vid_W33 --max_videos 500 --pixel \
  --out /storage/project/r-agarg35-0/lwang831/outputs/dialga/q3_rollout.json
echo "ROLLOUT_DONE $(date)"
