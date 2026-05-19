#!/bin/bash
# ===================================================================
# Iter 22 end-to-end sbatch — runs every verification stage in order.
#
# What this submits:
#   Phase 0 — smoke verification (5-vid Wan cache + 5-vid DINO cache +
#             1-epoch DINO smoke train + 1-epoch attrs smoke train).
#             ~10 min. Bails out if any contract is wrong before we
#             commit to the long runs.
#   Phase 1 — full DINO cache on the 10k Wan cache (resume-safe).
#             ~1-3 h.
#   Phase 2 — Iter 22a training: Iter 21 recipe + --lambda_dino 0.5.
#             ~3 h.
#   Phase 3 — Iter 22c training: Iter 21 recipe + --lambda_attrs 1.0.
#             ~3 h. Does not depend on DINO cache.
#   Phase 4 — identity probes (linear + z_dyn diagnostic) on both
#             checkpoints. ~15 min total.
#
# Total walltime budget: 12 h (sequential; comfortably under PACE
# 24h limit). Probes also run for Iter 22a/22c so the json files
# land next to each checkpoint.
#
# Outputs:
#   outputs/iter22a_<stamp>/trajectory.pt           Iter 22a checkpoint
#   outputs/iter22a_<stamp>/probe_identity_linear.json
#   outputs/iter22a_<stamp>/probe_identity_diag.json
#   outputs/iter22c_<stamp>/trajectory.pt           Iter 22c checkpoint
#   outputs/iter22c_<stamp>/probe_identity_linear.json
#   outputs/iter22c_<stamp>/probe_identity_diag.json
#   outputs/logs/iter22_<jobid>.{out,err}
#
# Adjust the SBATCH headers below for your cluster (account, partition,
# GPU type). The defaults assume a PACE H200 partition; edit as needed.
# ===================================================================

#SBATCH --job-name=dialga_iter22
#SBATCH --output=outputs/logs/iter22_%j.out
#SBATCH --error=outputs/logs/iter22_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H200:1
#SBATCH --mem=64G

set -euo pipefail

# ----- paths / env --------------------------------------------------
PROJECT_DIR=/storage/home/hcoda1/8/lwang831/workspace/Dialga
RIVER_PY=/storage/project/r-agarg35-0/lwang831/conda/envs/river/bin/python
DATA_DIR=/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/train_video
ANNOTATION_DIR=/storage/project/r-agarg35-0/lwang831/dataset/CLEVRER/annotations
WAN_CACHE=/storage/scratch1/8/lwang831/cache/wan_10000vid_W12
PROJECT_TMP=/storage/project/r-agarg35-0/lwang831/tmp

export TMPDIR="$PROJECT_TMP"
export MPLCONFIGDIR="$PROJECT_TMP/matplotlib"
export HF_HOME=/storage/project/r-agarg35-0/lwang831/hf_cache
export HF_HUB_CACHE=/storage/project/r-agarg35-0/lwang831/hf_cache/hub
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$HF_HOME" "$HF_HUB_CACHE"

cd "$PROJECT_DIR"
LOG_DIR="$PROJECT_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOG_DIR/iter22_${STAMP}_summary.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }
gate() {
  if [ "$1" -ne 0 ]; then
    log "*** FAILED at stage: $2 (exit $1) ***"
    exit "$1"
  fi
}

log "==== iter22 sbatch start  stamp=$STAMP ===="
log "project_dir = $PROJECT_DIR"
log "wan_cache   = $WAN_CACHE"
log "hf_home     = $HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv 2>&1 | tee -a "$SUMMARY" || true

# ===================================================================
# Phase 0 — smoke verification (5-vid mini cache + 1-epoch trains)
# ===================================================================
log ""
log "==== Phase 0: smoke verification ===="
SMOKE_DIR="$PROJECT_TMP/iter22_smoke_${STAMP}"
mkdir -p "$SMOKE_DIR"
log "smoke_dir = $SMOKE_DIR"

log "Phase 0a: tiny Wan cache (5 vids × 4 windows)"
$RIVER_PY -u scripts/cache_wan_latents.py \
    --data_dir "$DATA_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --max_videos 5 --windows_per_video 4 --window_length 12 \
    --frames_per_video 128 --image_size 128 \
    --out_dir "$SMOKE_DIR" --device cuda --dtype float16 \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase0a_wan_cache"

log "Phase 0b: tiny DINO cache (downloads dinov2-small if absent)"
$RIVER_PY -u scripts/cache_dino_features.py \
    --cache_dir "$SMOKE_DIR" \
    --data_dir "$DATA_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --device cuda --dtype float16 \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase0b_dino_cache"

log "Phase 0c: inspect DINO blob shape contract"
$RIVER_PY -c "
import torch, sys
b = torch.load('$SMOKE_DIR/dino/000000.pt', map_location='cpu', weights_only=False)
cls = b['cls']
print('cls.shape =', tuple(cls.shape))
print('cls.dtype =', cls.dtype)
print('cls.finite =', bool(torch.isfinite(cls).all()))
print('cls mean/std/min/max =',
      float(cls.mean()), float(cls.std()),
      float(cls.min()), float(cls.max()))
assert cls.shape == (12, 384), f'expected (12, 384), got {tuple(cls.shape)}'
assert cls.dtype == torch.float32, f'expected float32 on disk, got {cls.dtype}'
assert torch.isfinite(cls).all(), 'non-finite DINO features'
print('OK contract holds')
" 2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase0c_contract_check"

log "Phase 0d: 1-epoch smoke train with --lambda_dino 0.5"
$RIVER_PY -u scripts/train_trajectory.py \
    --cache_dir "$SMOKE_DIR" \
    --out_dir "$SMOKE_DIR/smoke_dino" \
    --epochs 1 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_dino 0.5 --d_dino 384 \
    --val_frac 0.0 --log_every 1 --ckpt_every 0 \
    --device cuda \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase0d_smoke_dino_train"

log "Phase 0e: 1-epoch smoke train with --lambda_attrs 1.0"
$RIVER_PY -u scripts/train_trajectory.py \
    --cache_dir "$SMOKE_DIR" \
    --out_dir "$SMOKE_DIR/smoke_attrs" \
    --epochs 1 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_attrs 1.0 \
    --val_frac 0.0 --log_every 1 --ckpt_every 0 \
    --device cuda \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase0e_smoke_attrs_train"

log "Phase 0 PASSED — all smoke checks green"

# ===================================================================
# Phase 1 — full DINO cache (resume-safe)
# ===================================================================
log ""
log "==== Phase 1: full DINO cache on 10k Wan cache ===="
mkdir -p "$WAN_CACHE/dino"
log "target dir: $WAN_CACHE/dino  (resume-safe: existing blobs skipped)"

$RIVER_PY -u scripts/cache_dino_features.py \
    --cache_dir "$WAN_CACHE" \
    --data_dir "$DATA_DIR" \
    --annotation_dir "$ANNOTATION_DIR" \
    --device cuda --dtype float16 \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase1_full_dino_cache"

N_DINO=$(ls "$WAN_CACHE/dino" 2>/dev/null | wc -l)
log "DINO cache complete: $N_DINO blobs"

# ===================================================================
# Phase 2 — Iter 22a training (DINO-CLS auxiliary)
# ===================================================================
log ""
log "==== Phase 2: Iter 22a training (DINO-CLS aux, λ=0.5) ===="
ITER22A_DIR="$PROJECT_DIR/outputs/iter22a_dino_${STAMP}"
mkdir -p "$ITER22A_DIR"

$RIVER_PY -u scripts/train_trajectory.py \
    --cache_dir "$WAN_CACHE" \
    --out_dir "$ITER22A_DIR" \
    --epochs 60 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_dino 0.5 --d_dino 384 \
    --val_frac 0.2 --val_every 5 \
    --log_every 50 --ckpt_every 5 \
    --device cuda \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase2_iter22a_train"
log "Iter 22a ckpt: $ITER22A_DIR/trajectory.pt"

# ===================================================================
# Phase 3 — Iter 22c training (supervised attrs head)
# ===================================================================
log ""
log "==== Phase 3: Iter 22c training (supervised attrs, λ=1.0) ===="
ITER22C_DIR="$PROJECT_DIR/outputs/iter22c_attrs_${STAMP}"
mkdir -p "$ITER22C_DIR"

$RIVER_PY -u scripts/train_trajectory.py \
    --cache_dir "$WAN_CACHE" \
    --out_dir "$ITER22C_DIR" \
    --epochs 60 --batch_size 4 --num_workers 0 \
    --lr 5e-4 --weight_decay 1e-3 --dropout 0.1 \
    --K 8 --d_model 192 --d_static 16 --d_dyn 32 \
    --lambda_smooth 0.1 --lambda_entropy 0.01 --lambda_vicreg 0.01 \
    --lambda_event_sup 0.02 \
    --lambda_attrs 1.0 \
    --val_frac 0.2 --val_every 5 \
    --log_every 50 --ckpt_every 5 \
    --device cuda \
    2>&1 | tee -a "$SUMMARY"
gate "${PIPESTATUS[0]}" "phase3_iter22c_train"
log "Iter 22c ckpt: $ITER22C_DIR/trajectory.pt"

# ===================================================================
# Phase 4 — identity probes (linear + z_dyn diagnostic) for both ckpts
# ===================================================================
log ""
log "==== Phase 4: probes ===="
for tag in "22a:$ITER22A_DIR" "22c:$ITER22C_DIR"; do
  name="${tag%%:*}"; dir="${tag##*:}"
  log "--- probe Iter $name ---"
  $RIVER_PY -u scripts/probe_iter21_identity.py \
      --cache_dir "$WAN_CACHE" \
      --ckpt "$dir/trajectory.pt" \
      --val_frac 0.2 --seed 42 --probe_split_seed 0 \
      2>&1 | tee -a "$SUMMARY"
  gate "${PIPESTATUS[0]}" "phase4_probe_${name}_linear"

  $RIVER_PY -u scripts/probe_iter21_zdyn_diag.py \
      --cache_dir "$WAN_CACHE" \
      --ckpt "$dir/trajectory.pt" \
      --val_frac 0.2 --seed 42 --probe_split_seed 0 \
      2>&1 | tee -a "$SUMMARY"
  gate "${PIPESTATUS[0]}" "phase4_probe_${name}_diag"
done

log ""
log "==== iter22 sbatch DONE  stamp=$STAMP ===="
log "Iter 22a: $ITER22A_DIR"
log "Iter 22c: $ITER22C_DIR"
log "Summary log: $SUMMARY"
