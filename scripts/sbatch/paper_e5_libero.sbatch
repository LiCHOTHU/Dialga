#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=05:00:00
#SBATCH --mem-per-gpu=120G
#SBATCH --job-name=paper_e5_libero
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# E5 — SECOND DATASET. The single biggest reviewer risk for the embedding paper is
# "CLEVRER-only", and CLEVRER is synthetic with a static camera. This answers it with
# ZERO new training: a LIBERO-90 v5.1.2 encoder already trained to ep200 (DONE marker) at
# /storage/scratch1/8/lwang831/dialga_outputs/libero_v512.
#
# LIBERO has no colour/material/shape labels, so the semantic axis there is ACTIONS: a tiny
# shared-across-time MLP on FROZEN features predicts the 6 continuous action dims + binary
# gripper. The script already ships the right controls, which is why this is worth running
# as-is rather than rebuilding:
#     z_dyn              <- the claim (motion should live here)
#     z_static           <- wrong tool by construction; should FAIL (identity, not motion)
#     wan_mean           <- raw Wan latent, spatially meaned (the "just use the VAE" baseline)
#     random_init_z_dyn  <- same arch untrained (capacity sanity)
#
# Two generalization axes, both already implemented:
#   in-task  : held-out val EPISODES of training tasks
#   cross-task: 10 tasks held out from representation training entirely (ids
#               [7,8,11,19,26,29,41,44,57,75]) — the honest generalization number.
#
# Block A re-runs at ep200 (prior JSON was v5_best.pt); Block B is the ep100 midpoint so we
# can state whether the action signal is stable across training rather than a lucky ckpt.

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
LIB_OUT="/storage/scratch1/8/lwang831/dialga_outputs/libero_v512"
LIB_CACHE="/storage/scratch1/8/lwang831/cache/libero_90_wan_W33"
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
OUT="${WORKDIR}/outputs/paper_e5"; mkdir -p "${OUT}"
nvidia-smi || true
set +x

echo "############ BLOCK A: LIBERO action probe @ep200 (converged) ############"
python -u scripts/eval_libero_action_probe.py \
    --cache_dir "${LIB_CACHE}" \
    --ckpt "${LIB_OUT}/ckpt_ep0200.pt" \
    --features z_dyn z_static wan_mean random_init_z_dyn \
    --probe_epochs 40 --probe_hidden 128 \
    --out "${OUT}/libero_action_ep200.json"

echo
echo "############ BLOCK B: LIBERO action probe @ep100 (stability check) ############"
python -u scripts/eval_libero_action_probe.py \
    --cache_dir "${LIB_CACHE}" \
    --ckpt "${LIB_OUT}/ckpt_ep0100.pt" \
    --features z_dyn z_static wan_mean random_init_z_dyn \
    --probe_epochs 40 --probe_hidden 128 \
    --out "${OUT}/libero_action_ep100.json"

echo "E5_LIBERO_SBATCH_DONE"
