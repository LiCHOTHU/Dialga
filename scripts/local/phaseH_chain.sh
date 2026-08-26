#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEG_DONE outputs/logs/phaseG.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseH.sh
