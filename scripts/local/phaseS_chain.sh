#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q PHASER_DONE outputs/logs/phaseR.log 2>/dev/null; do sleep 60; done
exec scripts/local/phaseS.sh
