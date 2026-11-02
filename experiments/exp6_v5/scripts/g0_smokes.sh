#!/bin/bash
# G0 smokes of V5 / V4-Continue on one 4 x L40S node (run inside the allocation):
#   GPU 0  gradient-scale probe (330 steps Bridge with provisional lambdas = 1; grad norms of each term on the new modules, last 30 steps)
#   GPU 1  V5 tiny overfit (300 steps on one fixed batch; Bridge 100 steps then Joint)
#   GPU 2  V4-Continue tiny overfit (300 steps)
#   GPU 3  V5 single-process resume test: A = 40 steps straight, B = 20 + resume to 40 (Bridge 20 -> Joint across the resume)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$S2E/bin/python experiments/exp6_v5/scripts/train_v5.py"; W=work_dirs/exp6_v5/runs/g0; L=work_dirs/exp6_v5/logs
rm -rf $W; mkdir -p $W
CUDA_VISIBLE_DEVICES=0 $PY --arm v5 --smoke gradscale --out $W/v5_gradscale --override lambda_surf=1 lambda_bridge0=1 > $L/g0_v5_gradscale.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY --arm v5 --smoke overfit --out $W/v5_overfit --override lambda_surf=0.1 lambda_bridge0=1 bridge_steps=100 bridge_end_step=150 > $L/g0_v5_overfit.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY --arm v4c --smoke overfit --out $W/v4c_overfit > $L/g0_v4c_overfit.log 2>&1 &
( OV="lambda_surf=0.1 lambda_bridge0=1 bridge_steps=20 bridge_end_step=30 ckpt_every=20"
  CUDA_VISIBLE_DEVICES=3 $PY --arm v5 --smoke resume --out $W/resume_A --override steps=40 $OV > $L/g0_resume_A.log 2>&1; echo "A exit $?"
  CUDA_VISIBLE_DEVICES=3 $PY --arm v5 --smoke resume --out $W/resume_B --override steps=20 $OV > $L/g0_resume_B1.log 2>&1; echo "B1 exit $?"
  CUDA_VISIBLE_DEVICES=3 $PY --arm v5 --smoke resume --out $W/resume_B --override steps=40 $OV > $L/g0_resume_B2.log 2>&1; echo "B2 exit $?" ) &
wait
echo G0-SMOKES-DONE
