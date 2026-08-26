#!/usr/bin/env bash
# Three seeds of the decisive comparison, at EXACTLY the configuration that produced
# the single-seed result (SSv2, 2000 train clips, 30 epochs, eval every 10):
#   H0  today's design      per-chunk, 96 floats on a 4x4 grid
#   H2  video-level 384     SAME total static bits as H0 (4x96 = 1x384)
#   H5  video-level 768     +7% total rate
# The claim under test is recon: H2 -7.6% and H5 -12.3% vs H0 on one seed.
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
  run H0_s$S --seed $S --d_static 96  --static_grid 4 --mem_update none
  run H2_s$S --seed $S --d_static 384 --static_grid 8 --mem_update video
  run H5_s$S --seed $S --d_static 768 --static_grid 8 --mem_update video \
             --static_target video_median --lambda_static_tgt 1.0
done
[ "$FAILED" -gt 0 ] && { echo "[seeds] $FAILED failed"; exit 1; }
echo "SEED_SWEEP_DONE"
