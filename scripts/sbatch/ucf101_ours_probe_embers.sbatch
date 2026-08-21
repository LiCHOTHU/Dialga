#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=03:00:00 --mem-per-gpu=80G --job-name=ucf101_ours_probe
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WD=/storage/home/hcoda1/8/lwang831/workspace/Dialga; cd "$WD"
source ~/.bashrc; conda activate river
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export MPLCONFIGDIR=/storage/project/r-agarg35-0/lwang831/tmp/matplotlib TORCHDYNAMO_DISABLE=1 PYTHONPATH="$WD"
C=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/cache
python -u scripts/probes/ssv2_action_probe.py \
  --ckpt /storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/ucf101_train/v5_best.pt \
  --train_cache "$C/ucf101_train_W33" --val_cache "$C/ucf101_val_W33" --device cuda
echo "PROBE_DONE $(date)"
