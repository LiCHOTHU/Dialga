#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASES_DONE outputs/logs/phaseS.log 2>/dev/null; do sleep 60; done
exec scripts/local/phaseT.sh
