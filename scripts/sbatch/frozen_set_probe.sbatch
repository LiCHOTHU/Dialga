#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=03:00:00
#SBATCH --mem-per-gpu=80G
#SBATCH --job-name=set_probe
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# FROZEN-ENCODER set-prediction probe — the paper's semantic table.
#
# Replaces the in-training AttrsHead readout, which is invalid: `_modal_labels`
# collapses per-slot attrs to ONE tie-broken modal class per video (effectively
# "lowest colour index present"), AND it backprops into the encoder at lam=0.5.
# Its val CE bottoms near chance (3.87) at ep15 then diverges to ~16 in every
# arm — it cannot rank architectures.
#
# Here the encoder is FROZEN and a single Linear predicts the permutation-
# invariant multi-hot PRESENCE set (which colours/materials/shapes appear).
# Scored as val mAP against a base-rate prior floor.
#
# Two blocks:
#   A) epoch-matched ablation @ ep50 — mean vs spatial vs spatial+gaze, all at
#      the SAME 96-float static budget. The pool ablation, honestly matched.
#   B) v55 epoch trend ep10..ep200 — shows the frozen probe does NOT diverge
#      with training while the old CE does. This is the "the metric was broken,
#      not the model" figure.
#
# Note: material (2 classes) and shape (3) have near-1.0 priors — almost every
# CLEVRER video contains all of them, so they are uninformative by construction.
# COLOUR (8 classes, prior ~0.51) is the discriminative axis; read that column.

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
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
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"
OUT="${WORKDIR}/outputs/frozen_set_probe"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "############ BLOCK A: epoch-matched pool ablation @ ep50 ############"
python -u scripts/probes/frozen_set_probe.py \
    --ckpts \
        mean_ep50="${CEDAR}/v55_pool_mean/ckpt_ep0050.pt" \
        spatial_ep50="${CEDAR}/v57_spatial/ckpt_ep0050.pt" \
        spatial_gaze_ep50="${CEDAR}/v57_spatial_gaze/ckpt_ep0050.pt" \
    --epochs 400 \
    --out "${OUT}/ablation_ep50.json"

echo
echo "############ BLOCK B: v55 epoch trend (metric-validity figure) ############"
python -u scripts/probes/frozen_set_probe.py \
    --ckpts \
        mean_ep10="${CEDAR}/v55_pool_mean/ckpt_ep0010.pt" \
        mean_ep50="${CEDAR}/v55_pool_mean/ckpt_ep0050.pt" \
        mean_ep100="${CEDAR}/v55_pool_mean/ckpt_ep0100.pt" \
        mean_ep150="${CEDAR}/v55_pool_mean/ckpt_ep0150.pt" \
        mean_ep200="${CEDAR}/v55_pool_mean/ckpt_ep0200.pt" \
    --epochs 400 \
    --out "${OUT}/trend_v55.json"

echo "SET_PROBE_SBATCH_DONE"
