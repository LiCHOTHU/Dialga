#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEF_DONE outputs/logs/phaseF.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseG.sh
