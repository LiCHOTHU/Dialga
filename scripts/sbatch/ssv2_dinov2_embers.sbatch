#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=06:00:00 --mem-per-gpu=80G --job-name=ssv2_dinov2
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"; SSV2=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ssv2
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib TORCHDYNAMO_DISABLE=1 PYTHONPATH="$WD"
python -u scripts/probes/ssv2_dinov2_probe.py \
  --video_root "$SSV2/videos_extracted/20bn-something-something-v2" \
  --label_json "$SSV2/labels.json" --train_json "$SSV2/train.json" --val_json "$SSV2/validation.json" \
  --max_train 4000 --max_test 1500 --out /storage/project/r-agarg35-0/lwang831/outputs/dialga/ssv2_dinov2.json
echo "SSV2_DINOV2_DONE $(date)"
