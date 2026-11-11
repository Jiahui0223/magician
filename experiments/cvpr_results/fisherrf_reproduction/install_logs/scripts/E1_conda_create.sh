#!/bin/bash
source "$(dirname "$0")/common.sh"
echo "[start] $(date -Is) host=$(hostname)"
set -x
conda create -y -p "$ENV" --override-channels -c conda-forge -c aihabitat \
  python=3.9 pip "setuptools=68.2.2" wheel \
  "habitat-sim=0.2.4=py3.9_headless_linux_f179b584bcd713c5a2a998132211e2cae881d6d1" \
  gcc_linux-64=11 gxx_linux-64=11 sysroot_linux-64=2.17
rc=$?
set +x
echo "[end] $(date -Is) exit=$rc"
exit $rc
