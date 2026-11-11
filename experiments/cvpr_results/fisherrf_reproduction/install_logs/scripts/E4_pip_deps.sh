#!/bin/bash
source "$(dirname "$0")/common.sh"
echo "[start] $(date -Is) host=$(hostname)"
set -x
"$ENV/bin/python" -m pip install --progress-bar off --prefer-binary \
  "numpy==1.26.4" "setuptools==68.2.2" \
  "gym==0.23.0" "hydra-core==1.3.2" "omegaconf==2.3.0" "opencv-python==4.9.0.80" "protobuf==3.20.1" \
  "threadpoolctl==3.4.0" "moviepy==1.0.3" "faster-fifo==1.5.2" "ifcfg==0.24" "tensorboard==2.8.0" \
  "lmdb==1.4.1" "webdataset==0.1.40" \
  "yacs==0.1.8" "einops==0.7.0" "kornia==0.7.3" "torchmetrics==1.3.2" "tensorboardX==2.6.2.2" \
  "wandb==0.16.6" "rich==13.7.1" "scikit-learn==1.4.2" "matplotlib==3.5.3" "ipython==8.18.1" \
  "pyyaml==6.0.1" "pytorch-msssim==1.0.0" "lpips==0.1.4" "plyfile==1.0.3"
rc=$?
set +x
echo "[end] $(date -Is) exit=$rc"
exit $rc
