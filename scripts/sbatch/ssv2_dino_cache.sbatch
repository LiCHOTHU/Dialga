#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=06:00:00 --mem-per-gpu=80G --job-name=ssv2_dino_cache
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"; SSV2=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ssv2
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib TORCHDYNAMO_DISABLE=1 PYTHONPATH="$WD"
python -u scripts/cache_dino_ssv2.py \
  --wan_cache_dir /storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33 \
  --video_root "$SSV2/videos_extracted/20bn-something-something-v2" \
  --out_dir /storage/scratch1/8/lwang831/cache/dino_ssv2_8000vid_W33
touch /storage/scratch1/8/lwang831/cache/dino_ssv2_8000vid_W33/DONE
echo "SSV2_DINO_CACHE_DONE $(date)"
