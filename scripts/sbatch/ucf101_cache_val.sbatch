#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=06:00:00 --mem-per-gpu=80G --job-name=ucf101_cache_val
#SBATCH --qos=inferno --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WORKDIR=/storage/home/hcoda1/8/lwang831/workspace/Dialga
UCF=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ucf101
CACHE=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/cache/ucf101_val_W33
PT=/storage/project/r-agarg35-0/lwang831/tmp
cd "$WORKDIR"; source ~/.bashrc; conda activate river
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TMPDIR="$PT" MPLCONFIGDIR="$PT/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}" TORCHDYNAMO_DISABLE=1
mkdir -p "$MPLCONFIGDIR" "$CACHE"
for i in $(seq 1 720); do [ -f "$UCF/EXTRACT_DONE" ] && break; sleep 30; done
python -u scripts/cache_wan_ucf101.py \
  --video_root "$UCF/UCF-101" \
  --split_list "$UCF/ucfTrainTestlist/testlist01.txt" \
  --class_ind  "$UCF/ucfTrainTestlist/classInd.txt" \
  --max_videos 3000 --window_frames 33 --image_size 128 --seed 0 \
  --dtype float16 --device cuda --out_dir "$CACHE"
touch "$CACHE/DONE"; echo "Done $(date)"
