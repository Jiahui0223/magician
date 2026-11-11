#!/bin/bash
source "$(dirname "$0")/common.sh"
echo "[start] $(date -Is) host=$(hostname)"
set -x
"$ENV/bin/python" -m pip install --progress-bar off --prefer-binary "torch==2.2.2+cu118" "torchvision==0.17.2+cu118" \
  --extra-index-url https://download.pytorch.org/whl/cu118
rc=$?
set +x
echo "[end] $(date -Is) exit=$rc"
exit $rc
