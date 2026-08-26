#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEE_DONE outputs/logs/phaseE.log 2>/dev/null; do sleep 45; done
while [ ! -f outputs/cache/dino_clevrer_W33/index.json ]; do sleep 45; done
exec scripts/local/phaseF.sh
