#!/bin/bash
# B2: load the arm config, verify the registry checkpoint (link == original, sha256, step, model_cfg == model_v1.json,
# strict load into V2Model, parameter count). CPU only.
source $(dirname $0)/../../shared/env.sh; cd $R
CUDA_VISIBLE_DEVICES="" $PY $X2/shared/v2_arm.py B2 --json $X2/B2/results/verify_config_checkpoint.json
