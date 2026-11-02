#!/bin/bash
# G0 smokes round 2 (after the surface-loss fix): gradient probe (GPU 0), V5 overfit (GPU 1), resume test (GPU 3).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$S2E/bin/python experiments/exp6_v5/scripts/train_v5.py"; W=work_dirs/exp6_v5/runs/g0; L=work_dirs/exp6_v5/logs
rm -rf $W/v5_gradscale $W/v5_overfit
CUDA_VISIBLE_DEVICES=0 $PY --arm v5 --smoke gradscale --out $W/v5_gradscale --override lambda_surf=1 lambda_bridge0=1 > $L/g0_v5_gradscale.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY --arm v5 --smoke overfit --out $W/v5_overfit --override lambda_surf=0.1 lambda_bridge0=1 bridge_steps=100 bridge_end_step=150 > $L/g0_v5_overfit.log 2>&1 &
bash experiments/exp6_v5/scripts/g0_resume.sh 0 > $L/g0_resume_driver.log 2>&1 &
wait
echo G0-SMOKES2-DONE
