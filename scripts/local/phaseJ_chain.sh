#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEI_DONE outputs/logs/phaseI.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseJ.sh
