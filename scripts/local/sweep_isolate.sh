#!/usr/bin/env bash
# Isolate SHARING from RESOLUTION.
#
# The headline H0->H2 gap (-9.4%) changes two things at once: per-chunk -> video-level,
# AND static grid 4x4 -> 8x8. z_dyn was already 8x8; z_static was the one at half the
# Wan lattice, and the 4x4 grid alone costs 8.3% rel-MSE from resolution (measured,
# diag_static_rate.py). So the gap is confounded.
#
# H6 is video-level at H0's EXACT code shape (96 floats, 4x4): same per-code size, same
# resolution, only sharing differs -- and it uses 4x FEWER static bits per clip (96 vs
# 4x96=384). If H6 still matches or beats H0, sharing is real independently of grid.
# H7 is per-chunk at grid 8 with the smallest legal code (128 = 2x8x8), isolating the
# resolution upgrade on its own.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/seed_sweep; mkdir -p "$OUT"
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10"
FAILED=0
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@"; then
    touch "$d/DONE"; else echo "[FAIL] $n"; FAILED=$((FAILED+1)); fi; }
for S in 0 1 2; do
  run H6_s$S --seed $S --d_static 96  --static_grid 4 --mem_update video   # sharing only
  run H7_s$S --seed $S --d_static 128 --static_grid 8 --mem_update none    # grid only
done
[ "$FAILED" -gt 0 ] && exit 1
echo "ISOLATE_DONE"
