#!/bin/bash
# Re-run of E6 section 2 (E6 lost $ENV: /etc/profile.d/modules.sh overwrites the shell variable ENV).
source "$(dirname "$0")/common.sh"
FENV="$ENV"
echo "[start] $(date -Is) host=$(hostname)"
source /etc/profile.d/modules.sh; module load cuda11.8/toolkit/11.8.0
for so in "$FENV"/lib/python3.9/site-packages/diff_gaussian_rasterization/_C*.so "$FENV"/lib/python3.9/site-packages/simple_knn/_C*.so; do
  echo "$so"
  echo "  cuobjdump --list-elf archs: $(cuobjdump --list-elf "$so" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
  echo "  cuobjdump --list-ptx: $(cuobjdump --list-ptx "$so" 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
  echo "  max GLIBC: $(objdump -T "$so" | grep -o 'GLIBC_[0-9.]*' | sort -V -u | tail -1)  max GLIBCXX: $(objdump -T "$so" | grep -o 'GLIBCXX_[0-9.]*' | sort -V -u | tail -1)  max CXXABI: $(objdump -T "$so" | grep -o 'CXXABI_[0-9.]*' | sort -V -u | tail -1)"
  echo "  ldd 'not found' (outside the env, torch libs resolve via python at import): $(ldd "$so" | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
done
for so in "$FENV"/lib/python3.9/site-packages/habitat_sim*/habitat_sim/_ext/*.so "$FENV"/lib/python3.9/site-packages/_magnum*.so; do
  echo "$so  ldd not found: $(ldd "$so" | grep 'not found' | awk '{print $1}' | tr '\n' ' ')"
done
echo "[end] $(date -Is)"
