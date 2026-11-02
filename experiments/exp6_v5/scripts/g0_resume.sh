#!/bin/bash
# G0-9 resume test of V5 on one GPU (pinned smoke trajectories): A = 40 steps straight, B = 20 + resume to 40.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$S2E/bin/python experiments/exp6_v5/scripts/train_v5.py"; W=work_dirs/exp6_v5/runs/g0; L=work_dirs/exp6_v5/logs; G=${1:-3}
rm -rf $W/resume_A $W/resume_B
OV="lambda_surf=0.1 lambda_bridge0=1 bridge_steps=20 bridge_end_step=30 ckpt_every=20"
CUDA_VISIBLE_DEVICES=$G $PY --arm v5 --smoke resume --out $W/resume_A --override steps=40 $OV > $L/g0_resume_A.log 2>&1; echo "A exit $?"
CUDA_VISIBLE_DEVICES=$G $PY --arm v5 --smoke resume --out $W/resume_B --override steps=20 $OV > $L/g0_resume_B1.log 2>&1; echo "B1 exit $?"
CUDA_VISIBLE_DEVICES=$G $PY --arm v5 --smoke resume --out $W/resume_B --override steps=40 $OV > $L/g0_resume_B2.log 2>&1; echo "B2 exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python experiments/exp6_v5/scripts/g0_resume_check.py > $L/g0_resume_check.log 2>&1; echo "check exit $?"
echo RESUME-DONE
