#!/usr/bin/env bash
set -uo pipefail
cd /home/licho/workspace/Dialga
while ! grep -q SEMANTIC_DONE outputs/logs/semantic.log 2>/dev/null; do sleep 60; done
exec scripts/local/overnight.sh
