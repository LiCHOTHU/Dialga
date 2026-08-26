#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q OVERNIGHT_DONE outputs/logs/overnight.log 2>/dev/null; do sleep 45; done
exec scripts/local/phaseE.sh
