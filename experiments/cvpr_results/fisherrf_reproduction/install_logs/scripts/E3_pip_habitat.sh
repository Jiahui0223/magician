#!/bin/bash
source "$(dirname "$0")/common.sh"
echo "[start] $(date -Is) host=$(hostname)"
set -x
"$ENV/bin/python" -m pip install --progress-bar off --no-deps "habitat-lab==0.2.420230405" "habitat-baselines==0.2.420230405"
rc=$?
set +x
echo "[end] $(date -Is) exit=$rc"
exit $rc
