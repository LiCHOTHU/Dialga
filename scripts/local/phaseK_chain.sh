#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEJ_DONE outputs/logs/phaseJ.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseK.sh
