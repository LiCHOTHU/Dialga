#!/usr/bin/env bash
# Re-run the memory arms with VARIABLE chunk count.
#
# With a fixed K=4 every training video has identical length, so a recurrent memory
# can encode chunk INDEX instead of scene content. Measured: a trained ConvGRU fed
# the SAME chunk four times returns cos(z_k,z_0) = [1.0, 0.63, 0.67, 0.999] -- a
# learned period-4 trajectory, independent of what it is shown. That makes the drift
# metric meaningless for those arms (it was measuring the recurrence, not the scene)
# and means the memory would not transfer to other video lengths.
#
# --rand_chunks samples K in [2,4] per batch so position carries no information.
# Smoke check: idempotence goes [1.0,.993,.982,.970] (smooth settling) and drift
# becomes monotonic 0.032/0.077/0.123, the shape accumulation should have.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
CACHE=outputs/cache/clevrer_W33_10k; OUT=outputs/vark_sweep
EPOCHS="${EPOCHS:-60}"; BS="${BS:-16}"; mkdir -p "$OUT"
# chain behind the rate sweep
while pgrep -f 'train_memory.py .*--out_dir outputs/(mem_sweep|rate_sweep)' >/dev/null 2>&1; do
  echo "[wait] earlier CLEVRER sweeps still running"; sleep 180
done
FAILED=0
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py --cache_dir "$CACHE" --out_dir "$d" \
       --epochs "$EPOCHS" --batch_size "$BS" --preload --rand_chunks "$@"; then
    touch "$d/DONE"; else echo "[FAIL] $n"; FAILED=$((FAILED+1)); sleep 20; fi; }
run V1_gru_randk       --mem_update gru
run V2_bd_gru_randk    --mem_update gru   --decoder basedelta
run V3_patch_randk     --mem_update patch
run V4_bd_patch_randk  --mem_update patch --decoder basedelta
run V5_pan_patch_randk --synth_pan --d_pose 32 --mem_update patch
[ "$FAILED" -gt 0 ] && { echo "[varK] $FAILED failed"; exit 1; }
echo "VARK_SWEEP_DONE"
