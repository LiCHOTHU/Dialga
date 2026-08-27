#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEV_DONE outputs/logs/phaseV.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseW.sh
