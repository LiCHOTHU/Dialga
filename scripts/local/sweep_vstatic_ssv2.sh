#!/usr/bin/env bash
# The two-point plan on SSv2 (real handheld video), nothing else running.
#
#   point 1  ONE z_static for the whole clip (mem_update=video)
#   point 2  explicit memory as a TEACHER: a median mosaic of the clip's own latent
#            frames, which z_static must predict (--static_target video_median)
#
# SSv2 has no camera pose, so the projection half of point 2 is not testable here;
# that needs LIBERO/DROID. What IS testable is whether a single video-level code plus
# an explicit teacher makes z_static carry the scene.
#
# RATE. Per clip the saving from a video-level code is d_static/(d_static+d_dyn).
# Here T_lat=5 so z_dyn is 5*256=1280/chunk: 23% at d_static=384, 37% at 768, but only
# 7% at the current 96. That is why the arms run at 384/768 -- a small video-level code
# is the configuration that cannot pay. PCA on the static target: 96 floats keep
# 68.3% of it, 384 keep 90.7%, 768 keep 97.1%.
# H0 is today's design for reference; H1 is the rate-matched per-chunk control, without
# which "bigger static code is better" would say nothing.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/vstatic_ssv2; mkdir -p "$OUT"
# sized to finish the whole 6-arm sweep in well under 20 minutes
EPOCHS="${EPOCHS:-30}"; BS="${BS:-16}"; NVID="${NVID:-2000}"
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs $EPOCHS --batch_size $BS --preload --max_videos $NVID \
        --eval_every 10"
R384="--d_static 384 --static_grid 8"
TGT="--static_target video_median --lambda_static_tgt 1.0"
FAILED=0
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  if python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@"; then
    touch "$d/DONE"; else echo "[FAIL] $n"; FAILED=$((FAILED+1)); sleep 15; fi; }

run H0_today_r96        --d_static 96 --static_grid 4 --mem_update none
run H1_perchunk_r384    $R384 --mem_update none
run H2_video_r384       $R384 --mem_update video
run H3_video_tgt_r384   $R384 --mem_update video $TGT
run H4_gru_tgt_r384     $R384 --mem_update gru --rand_chunks $TGT
run H5_video_tgt_r768   --d_static 768 --static_grid 8 --mem_update video $TGT
[ "$FAILED" -gt 0 ] && { echo "[vstatic-ssv2] $FAILED failed"; exit 1; }
echo "VSTATIC_SSV2_DONE"
