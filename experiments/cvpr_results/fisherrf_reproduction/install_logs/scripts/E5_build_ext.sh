#!/bin/bash
# Build the two CUDA extensions from COPIES of the clone sources (clones stay untouched).
source "$(dirname "$0")/common.sh"
echo "[start] $(date -Is) host=$(hostname)"
B="$ENV/_build"
set -x
mkdir -p "$B" || exit 1
rm -rf "$B/diff-gaussian-rasterization-modified" "$B/simple-knn"
cp -r "$AM/thirdparty/diff-gaussian-rasterization-modified" "$B/" || exit 1
cp -r "$AM/thirdparty/simple-knn" "$B/" || exit 1
rm -rf "$B/simple-knn/.git" "$B/diff-gaussian-rasterization-modified/third_party/glm/.git"
conda activate "$ENV" || exit 1
source /etc/profile.d/modules.sh
module load cuda11.8/toolkit/11.8.0
export CUDA_HOME=/cm/shared/apps/cuda11.8/toolkit/11.8.0
export PATH="$CUDA_HOME/bin:$PATH"
export NVCC_PREPEND_FLAGS="-ccbin $CXX"
export TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 MAX_JOBS=8
set +x
echo "CC=$CC CXX=$CXX"; $CXX --version | head -1; nvcc --version | tail -2; which python; python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
set -x
cd "$B/diff-gaussian-rasterization-modified" && python -m pip install --progress-bar off --no-build-isolation --no-cache-dir -v . ; rc1=$?
cd "$B/simple-knn" && python -m pip install --progress-bar off --no-build-isolation --no-cache-dir -v . ; rc2=$?
set +x
echo "[end] $(date -Is) exit_dgr=$rc1 exit_sknn=$rc2"
[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ]
