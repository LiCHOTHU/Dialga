#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEM_DONE outputs/logs/phaseM.log 2>/dev/null; do sleep 60; done
exec scripts/local/phaseN.sh
