#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=12:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=ssv2_cache
#SBATCH --qos=inferno
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# Wan-encode a subset of Something-Something-v2 (train split) into the DIALGA
# latent cache. Resume-safe (skips already-encoded windows), so a re-queued job
# continues where it left off. Consumed by ssv2_train.sbatch (--dataset ssv2).
set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
SSV2="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/datasets/ssv2"
CACHE="/storage/scratch1/8/lwang831/cache/ssv2_8000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"; export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}" "${CACHE}"
echo "Cache: ${CACHE}"; nvidia-smi || true

# Wait until the webm extraction is complete (sentinel dropped by the watcher).
# Guards against caching a partially-extracted tree (win_idx alignment).
for i in $(seq 1 720); do
    [ -f "${SSV2}/EXTRACT_DONE" ] && { echo "[gate] extraction complete"; break; }
    echo "[gate] waiting for extraction… (${i})"; sleep 30
done
[ -f "${SSV2}/EXTRACT_DONE" ] || { echo "[gate] extraction never finished — abort"; exit 1; }

python -u scripts/cache_wan_ssv2.py \
    --video_dir "${SSV2}/videos_extracted/20bn-something-something-v2" \
    --label_json "${SSV2}/labels.json" \
    --split_json "${SSV2}/train.json" \
    --max_videos 8000 --windows_per_video 3 \
    --window_frames 33 --image_size 128 --seed 0 \
    --dtype float16 --device cuda \
    --out_dir "${CACHE}"

touch "${CACHE}/DONE"
echo "Done at $(date)"
