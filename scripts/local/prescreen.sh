#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
source /home/licho/anaconda3/etc/profile.d/conda.sh; conda activate dialga
export PYTHONPATH=/home/licho/workspace/Dialga
PRE=outputs/cache/prescreen_6k; OUT=outputs/prescreen; mkdir -p $OUT
EP=${EP:-12}; BS=${BS:-16}
run(){ n="$1"; shift; [ -f "$OUT/$n/DONE" ] && { echo "[skip] $n"; return; }
  echo "===== ARM $n ====="
  python -u scripts/local/train_memory.py --cache_dir $PRE --out_dir $OUT/$n \
     --epochs $EP --batch_size $BS --preload "$@" && touch "$OUT/$n/DONE"; }
run A1_base                      --mem_update none --mem_collapse mean
run A4_gru                       --mem_update gru  --mem_collapse mean
run B1_basedelta                 --mem_update none --mem_collapse mean --decoder basedelta
run B2_basedelta_gru             --mem_update gru  --mem_collapse mean --decoder basedelta
run C1_pan_none      --synth_pan --mem_update none --mem_collapse mean
run C2_pan_gru       --synth_pan --mem_update gru  --mem_collapse mean
run C3_pan_gru_world --synth_pan --mem_update gru  --mem_collapse world --d_pose 32
echo PRESCREEN_DONE
