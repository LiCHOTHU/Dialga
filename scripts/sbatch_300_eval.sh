#!/bin/bash
#SBATCH --job-name=dlg_300_eval
#SBATCH --account=gts-agarg35
#SBATCH --partition=gpu-a100,gpu-v100,gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eval_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/slurm/dlg_300_eval_%j.err

# Run rungs 4-7 on the 300-video shakedown.
# Submit with --dependency=afterok:<gtpos_id>:<encpos_id>.

set -euo pipefail

REPO=/storage/home/hcoda1/8/lwang831/workspace/Dialga
PYTHON=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
OUTBASE=/storage/project/r-agarg35-0/lwang831/outputs/dialga

mkdir -p "${OUTBASE}/slurm"

GROUP=${GROUP:-scale_300_$(date +%m%d)}
SLOT_RUN=${SLOT_RUN:-slot_300_${GROUP}}
WANFLOW_GT_RUN=${WANFLOW_GT_RUN:-wanflow_300_gtpos_${GROUP}}
WANFLOW_ENC_RUN=${WANFLOW_ENC_RUN:-wanflow_300_encpos_${GROUP}}

SLOT_CKPT="${OUTBASE}/${SLOT_RUN}/stage2.pt"
WF_GT_CKPT="${OUTBASE}/${WANFLOW_GT_RUN}/decoder.pt"
WF_ENC_CKPT="${OUTBASE}/${WANFLOW_ENC_RUN}/decoder.pt"

EVAL_DIR="${OUTBASE}/eval_${GROUP}"
mkdir -p "${EVAL_DIR}"

cd "${REPO}"

for f in "${SLOT_CKPT}" "${WF_GT_CKPT}" "${WF_ENC_CKPT}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: missing checkpoint: ${f}"
        exit 2
    fi
done

echo "=========================================="
echo " Rung 4-7 evaluation @ 300 videos"
echo "=========================================="
echo "  JobID            : ${SLURM_JOB_ID}"
echo "  Group            : ${GROUP}"
echo "  Slot ckpt        : ${SLOT_CKPT}"
echo "  Wan-flow GT-q    : ${WF_GT_CKPT}"
echo "  Wan-flow enc-q   : ${WF_ENC_CKPT}"
echo "  Eval out dir     : ${EVAL_DIR}"
echo "  GPU              : $(nvidia-smi -L 2>/dev/null | head -1)"
echo "  Start            : $(date)"
echo "=========================================="

export PYTHONUNBUFFERED=1

# Rung 4 + 5: encoder + dynamics per-video
echo ""
echo "--- Rung 4 + 5: encoder state, 1-step, multi-step rollout ---"
"${PYTHON}" -u scripts/eval_overfit_per_video.py \
    --ckpt "${SLOT_CKPT}" \
    | tee "${EVAL_DIR}/rung4_5.txt"

# Rung 6: rollout-then-decode (with both decoders)
echo ""
echo "--- Rung 6 (GT-q decoder): full pipeline pixel MSE ---"
"${PYTHON}" -u scripts/eval_rollout_decode.py \
    --ckpt_slot "${SLOT_CKPT}" \
    --ckpt_decoder "${WF_GT_CKPT}" \
    --out_dir "${EVAL_DIR}/rung6_gtpos_decoder" \
    --num_steps 8 \
    | tee "${EVAL_DIR}/rung6_gtpos_decoder.txt"

echo ""
echo "--- Rung 6 (enc-q decoder): full pipeline pixel MSE ---"
"${PYTHON}" -u scripts/eval_rollout_decode.py \
    --ckpt_slot "${SLOT_CKPT}" \
    --ckpt_decoder "${WF_ENC_CKPT}" \
    --out_dir "${EVAL_DIR}/rung6_encpos_decoder" \
    --num_steps 8 \
    | tee "${EVAL_DIR}/rung6_encpos_decoder.txt"

# Rung 7: counterfactual (using enc-q decoder, the realistic path)
echo ""
echo "--- Rung 7: counterfactual edit ---"
"${PYTHON}" -u scripts/eval_counterfactual.py \
    --ckpt_slot "${SLOT_CKPT}" \
    --ckpt_decoder "${WF_ENC_CKPT}" \
    --out_dir "${EVAL_DIR}/rung7_counterfactual" \
    --target_video 4242 \
    | tee "${EVAL_DIR}/rung7.txt"

echo ""
echo "[$(date)] All rungs done."

# Mirror to repo outputs/
LOCAL_MIRROR="${REPO}/outputs/eval_${GROUP}"
mkdir -p "${LOCAL_MIRROR}"
cp -rf "${EVAL_DIR}"/* "${LOCAL_MIRROR}/" 2>/dev/null || true
echo "  -> ${LOCAL_MIRROR}/"
