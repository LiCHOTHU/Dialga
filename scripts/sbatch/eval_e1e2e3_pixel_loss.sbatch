#!/usr/bin/env bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-a100,gpu-h100,gpu-h200,gpu-v100
#SBATCH --time=01:30:00
#SBATCH --mem-per-gpu=48G
#SBATCH --job-name=evalpix
#SBATCH --qos=embers
#SBATCH --account=gts-agarg35
#SBATCH --output=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.out
#SBATCH --error=/storage/project/r-agarg35-0/lwang831/outputs/dialga/logs/%x_%j.err

# ===================================================================
# Matched val L_pixel for e1 / e2 / e3 (apples-to-apples).
#
# All three trained on max_videos=1000, val_frac=0.2, seed=42, so the val
# split is identical and held-out for each. eval_pixel_loss.py auto-handles:
#   e1 (frozen VAE, no wan_vae in ckpt) -> pretrained VAE + cached latent
#   e2 (enc unfrozen)  -> load trained VAE, RE-ENCODE raw pix through trained enc
#   e3 (dec unfrozen)  -> load trained VAE, decode through trained decoder
# bfloat16 throughout to match the trainer's vae_dtype.
# ===================================================================

set -eo pipefail
WORKDIR="/storage/home/hcoda1/8/lwang831/workspace/Dialga"
WAN_CACHE="/storage/scratch1/8/lwang831/cache/wan_10000vid_W33"
VIDEO_DIR="/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video"
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
export PYTHONPATH="${WORKDIR}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${MPLCONFIGDIR}"

OUT_ROOT="${WORKDIR}/outputs/pixel_loss_e1e2e3_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${OUT_ROOT}"
echo "OutRoot: ${OUT_ROOT}"; nvidia-smi || true

OUT_BASE="/storage/scratch1/8/lwang831/dialga_outputs"
CKPT_E1="${OUT_BASE}/v512_e1_frozen_20260526_001020/v5_best.pt"
CKPT_E2="${OUT_BASE}/v512_e2_enc_unfrozen/last.pt"
CKPT_E3="${OUT_BASE}/v512_e3_dec_unfrozen/last.pt"

N_CHUNKS=${N_CHUNKS:-200}
BATCH=${BATCH:-2}

for tag in e1 e2 e3; do
    case "${tag}" in
        e1) CKPT="${CKPT_E1}"; LABEL="e1_frozen" ;;
        e2) CKPT="${CKPT_E2}"; LABEL="e2_enc_unfrozen" ;;
        e3) CKPT="${CKPT_E3}"; LABEL="e3_dec_unfrozen" ;;
    esac
    echo ""; echo "==== ${LABEL} ===="; echo "ckpt: ${CKPT}"
    python -u scripts/eval_pixel_loss.py \
        --ckpt "${CKPT}" \
        --cache_dir "${WAN_CACHE}" \
        --video_dir "${VIDEO_DIR}" \
        --split val --val_frac 0.2 --seed 42 --max_videos 1000 \
        --n_chunks "${N_CHUNKS}" \
        --batch_size "${BATCH}" --num_workers 2 \
        --dtype bfloat16 \
        --out_json "${OUT_ROOT}/${LABEL}.json"
done

echo ""; echo "================ SUMMARY ================"
python - "${OUT_ROOT}" <<'PY'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
order = ["e1_frozen", "e2_enc_unfrozen", "e3_dec_unfrozen"]
print(f"{'run':<18} {'L_pixel(obs)':>13} {'per-chunk mean±std':>24} {'n':>5}")
print("-"*64)
for name in order:
    f = root/f"{name}.json"
    if not f.exists():
        print(f"{name:<18} {'(missing)':>13}"); continue
    d = json.loads(f.read_text())
    print(f"{name:<18} {d['L_pixel_obs']:13.6f} "
          f"{d['per_chunk_mean']:11.6f} ± {d['per_chunk_std']:<9.6f} {d['n_chunks']:>5}")
PY

echo ""; echo "Finished at $(date)"
echo "Results under: ${OUT_ROOT}"
