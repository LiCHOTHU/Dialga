#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p gpu-l40s
#SBATCH --time=05:00:00
#SBATCH --mem-per-gpu=120G
#SBATCH --job-name=paper_disent
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35-ideas_l40s
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# THE FACTORIZATION TABLE. Our architecture's actual novelty is that the embedding is
# SPLIT — z_static (what) + z_dyn (motion) + z_event. The frozen-probe result so far only
# shows z_static carries identity; it says nothing about whether the SPLIT is meaningful.
# A reviewer will ask: is this factorization real, or is it one 964-float blob you happened
# to slice? This probe answers it by cross-testing each part against both label types:
#
#     z_static -> object identity   should WIN
#     z_static -> motion/position   should FAIL
#     z_dyn    -> motion/position   should WIN
#     z_dyn    -> object identity   should FAIL
#
# The off-diagonal FAILURES are the evidence; a factorization that predicts everything from
# everything is not a factorization. Uses CLEVRER trajectory GT (outputs/trajectory_gt_val.pt,
# built 2026-05-21).
#
# Three arms, so the claim is not resting on one checkpoint:
#   A v55_pool_mean ep200      — the headline arm (lambda_attrs=0.5)
#   B v512_clevrer_noattrs ep70 — the LABEL-FREE arm; if the split holds here it is not an
#                                 artifact of attribute supervision
#   C v512_clevrer_nopixel ep200 — attrs-on, pixel-off; controls for the pixel loss

set -eo pipefail
set -x
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
CEDAR="/storage/cedar/cedar0/cedarp-agarg35-0/liquan.w/dialga_outputs"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
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
OUT="${WORKDIR}/outputs/paper_disentangle"; mkdir -p "${OUT}"
GT="${WORKDIR}/outputs/trajectory_gt_val.pt"
nvidia-smi || true
set +x

echo "############ A: v55_pool_mean ep200 (headline arm) ############"
python -u scripts/probes/probe_v512_disentangle.py \
    --cache_dir "${WAN_CACHE}" --gt "${GT}" \
    --ckpt "${CEDAR}/v55_pool_mean/ckpt_ep0200.pt" \
    --probe_epochs 60 \
    --out_json "${OUT}/disent_v55_ep200.json"

echo
echo "############ B: v512_clevrer_noattrs ep70 (LABEL-FREE arm) ############"
python -u scripts/probes/probe_v512_disentangle.py \
    --cache_dir "${WAN_CACHE}" --gt "${GT}" \
    --ckpt "${CEDAR}/v512_clevrer_noattrs/ckpt_ep0070.pt" \
    --probe_epochs 60 \
    --out_json "${OUT}/disent_noattrs_ep70.json"

echo
echo "############ C: v512_clevrer_nopixel ep200 (attrs-on, pixel-off) ############"
python -u scripts/probes/probe_v512_disentangle.py \
    --cache_dir "${WAN_CACHE}" --gt "${GT}" \
    --ckpt "${CEDAR}/v512_clevrer_nopixel/ckpt_ep0200.pt" \
    --probe_epochs 60 \
    --out_json "${OUT}/disent_nopixel_ep200.json"

echo "DISENTANGLE_SBATCH_DONE"
