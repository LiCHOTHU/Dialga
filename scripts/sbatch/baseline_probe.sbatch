#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=04:00:00
#SBATCH --mem-per-gpu=180G
#SBATCH --job-name=baseline_probe
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# THE GATE TABLE for the video-embedding claim.
#
# frozen_set_probe proved our z_static beats a base-rate prior. That is not a
# paper. This asks whether it beats the alternatives a reviewer names first:
# a dim-matched PCA of the raw Wan latent, the raw latent itself, an UNTRAINED
# copy of our own encoder, and DINOv2. All free — no training, cached features.
#
# The claim "96 learned floats are a better video embedding than 27648 raw VAE
# floats" lives or dies here. On a 256-sample smoke it held (ours 0.869 colour
# vs pca96 0.594, random 0.639, dino 0.834), but wan_flat needs the FULL 16k
# train set to be meaningful — a 27648-dim linear probe on 256 samples is pure
# overfit and its smoke number (0.574) is not trustworthy.
#
# mem 180G: wan_flat is 16000 x 27648 (~1.8 GB fp32) plus the PCA workspace.
#
# Block A = our best converged arm (v55_pool_mean ep200).
# Block B = the spatial arm at ep50, to check the pool conclusion on THIS metric
#           too (frozen_set_probe already had mean-pool ahead at ep50).

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
DINO_CACHE="/storage/scratch1/8/lwang831/cache/dino_patch_10000vid_W33"
PROJECT_TMP="/storage/project/r-agarg35-0/lwang831/tmp"
cd "${WORKDIR}"

source ~/.bashrc
conda activate river
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TMPDIR="${PROJECT_TMP}"
export TEMP="${TMPDIR}"; export TMP="${TMPDIR}"
export MPLCONFIGDIR="${PROJECT_TMP}/matplotlib"
export HF_HOME="/storage/project/r-agarg35-0/lwang831/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export TORCH_HOME="/storage/project/r-agarg35-0/lwang831/torch_home"
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"
OUT="${WORKDIR}/outputs/baseline_probe"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "############ BLOCK A: v55_pool_mean ep200 (headline arm) ############"
python -u scripts/probes/baseline_probe_table.py \
    --ckpt "${CEDAR}/v55_pool_mean/ckpt_ep0200.pt" \
    --dino_cache_dir "${DINO_CACHE}" \
    --epochs 400 --pca_dim 96 \
    --out "${OUT}/baselines_v55_ep200.json"

echo
echo "############ BLOCK B: v57_spatial ep50 (pool check on this metric) ############"
python -u scripts/probes/baseline_probe_table.py \
    --ckpt "${CEDAR}/v57_spatial/ckpt_ep0050.pt" \
    --dino_cache_dir "${DINO_CACHE}" \
    --epochs 400 --pca_dim 96 \
    --out "${OUT}/baselines_v57_ep50.json"

echo "BASELINE_PROBE_SBATCH_DONE"
