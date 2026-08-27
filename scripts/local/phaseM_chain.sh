#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASEK_DONE outputs/logs/phaseK.log 2>/dev/null; do sleep 60; done
exec scripts/local/phaseM.sh
