#!/usr/bin/env bash
# SSv2: the honest testbed for a static-scene memory, and the MosaicMem alignment
# ablation on real video.
#
# Why SSv2 changes the question. On CLEVRER the camera is fixed and the background is
# a constant grey, so z_static has almost nothing video-specific to hold -- deleting
# it costs 7% while deleting z_dyn costs 1116%. On SSv2 (handheld, real scenes) the
# smoke run already reads +38.6% / +24.7%: the two codes start out COMPARABLE. So the
# "z_static is decorative" problem is substantially a CLEVRER artifact, and SSv2 is
# where a memory has to earn its keep for real.
#
# SSv2 has NO camera pose, so the warped alignments are identity there and the patch
# memory must align by attention alone (S3/S5). The pan block adds a known synthetic
# camera on top of the real motion purely so latent/rope/both can be separated -- the
# MosaicMem Table-1 ablation, on real video.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh
conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga

CACHE=outputs/cache/ssv2_W17_6k
CLOG=outputs/logs/ssv2_cache.log
OUT=outputs/ssv2_sweep
EPOCHS="${EPOCHS:-40}"
BS="${BS:-16}"
mkdir -p "$OUT"

while ! grep -q "^\[done\]" "$CLOG" 2>/dev/null; do
  echo "[wait] ssv2 cache: $(tail -1 "$CLOG" 2>/dev/null | tr -d '\n')"
  sleep 120
done
echo "[wait] ssv2 cache complete."

# W=17 -> T_lat=5
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir $CACHE --epochs $EPOCHS --batch_size $BS --preload"

FAILED=0
run () {
  local name="$1"; shift
  local dir="$OUT/$name"
  if [ -f "$dir/DONE" ]; then echo "[skip] $name"; return; fi
  echo "=================== ARM $name @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py $COMMON --out_dir "$dir" "$@"; then
    touch "$dir/DONE"
  else
    echo "[FAIL] arm $name"; FAILED=$((FAILED+1)); sleep 20
  fi
}

# --- pose-free: does retrieve-and-compose pay when nothing says where things went? ---
run S1_base            --mem_update none
run S2_gru             --mem_update gru
run S3_patch           --mem_update patch
run S4_bd              --mem_update none  --decoder basedelta
run S5_bd_patch        --mem_update patch --decoder basedelta

# --- known pose (synthetic pan on top of real motion): the MosaicMem ablation ---
run S6_pan_latent  --synth_pan --d_pose 32 --mem_update patch_latent
run S7_pan_rope    --synth_pan --d_pose 32 --mem_update patch_rope
run S8_pan_both    --synth_pan --d_pose 32 --mem_update patch
run S9_pan_gru     --synth_pan --d_pose 32 --mem_update gru

if [ "$FAILED" -gt 0 ]; then
  echo "[ssv2] $FAILED arm(s) failed; exiting non-zero to retry"
  exit 1
fi
echo "SSV2_SWEEP_DONE"
