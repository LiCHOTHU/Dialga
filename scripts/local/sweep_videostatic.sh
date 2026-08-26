#!/usr/bin/env bash
# The two-point plan: (1) ONE z_static for the whole video, (2) explicit memory used
# to guide it -- as a computed teacher and, with pose, as a canonical frame projected
# into each chunk's view.
#
# Measured before committing GPU to it:
#   a video-level static image still explains ~86% of every chunk (per-chunk: 91%),
#   so the video-level constraint costs only ~6 points -- cheap for what it buys;
#   a MEDIAN mosaic disagrees with the mean 3.8x more at high-motion cells, so the
#   teacher really encodes the non-moving scene rather than the average frame.
#
# RATE IS THE POINT, and it only pays with a LARGE static code. Per video the saving
# is d_static/(d_static+d_dyn): 4% at d_static=96 (nothing to see), 25% at 768. So
# every arm here runs at 384/768 on an 8x8 grid, never 96 -- a small video-level code
# is the configuration that cannot work. PCA says 96 floats keeps 68.3% of the
# time-constant content, 384 keeps 90.7%, 768 keeps 97.1%.
# G4 is the rate-matched per-chunk control: without it "bigger static code
# reconstructs better" says nothing.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
CACHE=outputs/cache/clevrer_W33_10k; OUT=outputs/vstatic_sweep
EPOCHS="${EPOCHS:-60}"; BS="${BS:-16}"; mkdir -p "$OUT"
# SKIP_WAIT=1 runs concurrently with the earlier sweeps (the models are ~10M params
# and the box has 32 GB of VRAM with <5 GB in use; compute is shared, memory is not
# a constraint). Default is to queue politely behind them.
if [ "${SKIP_WAIT:-0}" != "1" ]; then
while pgrep -f 'train_memory.py .*--out_dir outputs/(mem_sweep|rate_sweep|vark_sweep)' >/dev/null 2>&1; do
  echo "[wait] earlier CLEVRER sweeps running"; sleep 180
done
fi
R384="--d_static 384 --static_grid 8"
FAILED=0
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py --cache_dir "$CACHE" --out_dir "$d" \
       --epochs "$EPOCHS" --batch_size "$BS" --preload "$@"; then touch "$d/DONE"
  else echo "[FAIL] $n"; FAILED=$((FAILED+1)); sleep 20; fi; }

run G4_perchunk_r384  $R384 --mem_update none                 # rate-matched control
run G1_video_r384     $R384 --mem_update video                # point 1 alone
run G2_video_tgt      $R384 --mem_update video --static_target video_median --lambda_static_tgt 1.0
run G5_gru_tgt        $R384 --mem_update gru --rand_chunks --static_target video_median --lambda_static_tgt 1.0
run G3_videoproj_tgt  $R384 --mem_update video_proj --synth_pan --d_pose 32 \
                            --static_target video_median --lambda_static_tgt 1.0
run G6_video_r768     --d_static 768 --static_grid 8 --mem_update video \
                            --static_target video_median --lambda_static_tgt 1.0
[ "$FAILED" -gt 0 ] && { echo "[vstatic] $FAILED failed"; exit 1; }
echo "VSTATIC_SWEEP_DONE"
