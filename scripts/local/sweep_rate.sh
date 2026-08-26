#!/usr/bin/env bash
# Does z_static RATE buy reconstruction back, and only where it is needed?
#
# Measured (scripts/local/diag_static_rate.py, PCA on mean_t(x) = the target the
# base+delta decoder hands to z_static alone):
#     96 floats  rel-MSE 0.317      <- the current budget discards ~32%
#    384 floats  rel-MSE 0.093
#    768 floats  rel-MSE 0.029
#   4x4 grid     rel-MSE 0.083 extra, purely resolution (8x8 is exact)
#
# PREDICTION: extra static rate is worth a lot under base+delta (E1/E2/E4), where the
# decoder must get the time-constant part from z_static, and worth almost nothing
# under the standard decoder (E3), where z_static is decorative -- which is exactly
# what the cluster's clevrer_static8 arm found (~+3%).
# E3 is the control that makes this a claim rather than tuning. Without it, "more
# rate helped" is uninformative.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh
conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga

CACHE=outputs/cache/clevrer_W33_10k
OUT=outputs/rate_sweep
EPOCHS="${EPOCHS:-60}"
BS="${BS:-16}"
mkdir -p "$OUT"

# chain behind the main sweep so timings stay comparable
while pgrep -f 'local/train_memory.py --cache_dir outputs/cache/clevrer_W33_10k --out_dir outputs/mem_sweep' >/dev/null 2>&1; do
  echo "[wait] main sweep still running: $(ls outputs/mem_sweep/*/DONE 2>/dev/null | wc -l)/14 arms done"
  sleep 180
done
echo "[wait] main sweep finished."

FAILED=0
run () {
  local name="$1"; shift
  local dir="$OUT/$name"
  if [ -f "$dir/DONE" ]; then echo "[skip] $name"; return; fi
  echo "=================== ARM $name @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py \
      --cache_dir "$CACHE" --out_dir "$dir" \
      --epochs "$EPOCHS" --batch_size "$BS" --preload "$@"; then
    touch "$dir/DONE"
  else
    echo "[FAIL] arm $name"; FAILED=$((FAILED+1)); sleep 20
  fi
}

#            d_static 384 = 6 channels on an 8x8 grid (no upsampling loss)
run E1_bd_r384      --decoder basedelta --d_static 384 --static_grid 8 --mem_update none
run E2_bd_r384_gru  --decoder basedelta --d_static 384 --static_grid 8 --mem_update gru
run E3_grid_r384    --decoder grid      --d_static 384 --static_grid 8 --mem_update none
run E4_bd_r768_gru  --decoder basedelta --d_static 768 --static_grid 8 --mem_update gru

# leftovers the live-edited main sweep skipped
run A5b_attn_open   --mem_update attn --mem_collapse mean --attn_gate_bias 0.0

if [ "$FAILED" -gt 0 ]; then
  echo "[rate] $FAILED arm(s) failed; exiting non-zero to retry"
  exit 1
fi
echo "RATE_SWEEP_DONE"
