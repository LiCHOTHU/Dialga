#!/usr/bin/env bash
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --gres=gpu:1 -p gpu-l40s
#SBATCH --time=06:00:00 --mem-per-gpu=80G --job-name=ucf101_basePCA
#SBATCH --qos=embers --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err
set -eo pipefail
WORKDIR=/storage/home/hcoda1/8/lwang831/workspace/Dialga
UCF=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ucf101
PT=/storage/project/r-agarg35-0/lwang831/tmp
OUT=/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs/ucf101_baselines
cd "$WORKDIR"; source ~/.bashrc; conda activate river
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TMPDIR="$PT" MPLCONFIGDIR="$PT/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub" HUGGINGFACE_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export PYTHONPATH="$WORKDIR:${PYTHONPATH:-}" TORCHDYNAMO_DISABLE=1
mkdir -p "$MPLCONFIGDIR" "$OUT"
MODEL="${MODEL:?set MODEL=videomae|videoflextok}"
echo "MODEL=$MODEL"; nvidia-smi || true
python -u scripts/probes/ucf101_baselines_probe.py --model "$MODEL" \
  --video_root "$UCF/UCF-101" --split_dir "$UCF/ucfTrainTestlist" \
  --max_train 4000 --max_test 1500 --seed 0 \
  --out "$OUT/ucf101_${MODEL}.json"
echo "Done $(date)"
