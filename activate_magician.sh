#!/bin/bash
# Source this to activate the MAGICIAN env:   source activate_magician.sh
#
# The env binaries are glibc-2.17 portable + compiled for sm_60/70/80/89
# (P100 / V100 / A100 / L40), so this works BOTH on the bare host (CentOS 7 /
# glibc 2.17) AND inside the apptainer container (Debian / glibc 2.36) — wherever
# your shell happens to be, on any of those four GPU types.
#
# At RUNTIME no CUDA module is needed: torch 2.6.0+cu118 bundles its own CUDA 11.8
# runtime; only the NVIDIA driver (host, or --nv in the container) is required.

source /cm/shared/apps/conda/23.1.0/etc/profile.d/conda.sh
conda activate /bsuscratch/jiahuizhang/envs/magician
# SPATIALINDEX_C_LIBRARY (needed by rtree) is auto-exported by the env's activate.d hook.

echo "magician env active: $(python --version 2>&1) @ $CONDA_PREFIX"
python - <<'PY' 2>/dev/null
import torch
g = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"
print(f"  torch {torch.__version__} | cuda {torch.version.cuda} | GPU: {g}")
PY
echo "  run:  python test_magician_planning.py"
