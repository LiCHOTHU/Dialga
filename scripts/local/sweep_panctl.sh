#!/usr/bin/env bash
# PAN-MATCHED CONTROL. H11 (PatchMemory + pose) reads 0.0603 but runs with
# --synth_pan, which WARPS the input, so its reconstruction target is not the same
# tensor the no-pan arms are scored on (border-padded warping smears edge detail and
# may simply be easier). These arms put the other mechanisms under the IDENTICAL pan
# so the comparison is about the mechanism and not the target.
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
OUT=outputs/seed_sweep
COMMON="--dataset ssv2 --chunk_size_lat 5 --cache_dir outputs/cache/ssv2_W17_6k \
        --epochs 30 --batch_size 16 --preload --max_videos 2000 --eval_every 10 \
        --d_static 384 --static_grid 8 --synth_pan --d_pose 32"
run(){ n="$1"; shift; d="$OUT/$n"; [ -f "$d/DONE" ] && { echo "[skip] $n"; return; }
  echo "=================== ARM $n @ $(date -Is) ==================="
  python -u scripts/local/train_memory.py $COMMON --out_dir "$d" "$@" && touch "$d/DONE"; }
run P1_pan_perchunk_s0 --seed 0 --mem_update none        # baseline under the same pan
run P2_pan_video_s0    --seed 0 --mem_update video_proj  # video-level + projection
run P3_pan_patch_tgt_s0 --seed 0 --mem_update patch \
                        --static_target video_median --lambda_static_tgt 1.0
echo "PANCTL_DONE"
