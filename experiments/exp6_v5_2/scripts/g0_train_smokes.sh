#!/bin/bash
# G0 training smokes of V5.2 on one 4-GPU node (pinned smoke trajectories of exp6_v5):
#   GPU 0  gradient probe in the Joint setting (hazard_steps 100, 320 steps; grad norms of L_surface / L_firsthit on the shared backbone over the last 20)
#   GPU 1  overfit (300 steps on one fixed batch; hazard 100 -> joint)
#   GPU 2  single-process resume: A = 40 steps, B = 20 + resume to 40 (hazard 20 -> joint across the resume)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
PY="$S2E/bin/python experiments/exp6_v5_2/scripts/train_v5_2.py"; W=work_dirs/exp6_v5_2/runs/g0; L=work_dirs/exp6_v5_2/logs
rm -rf $W; mkdir -p $W $L
G0=${G0:-0}; G1=${G1:-1}; G2=${G2:-2}
CUDA_VISIBLE_DEVICES=$G0 $PY --smoke gradscale --out $W/gradscale --override hazard_steps=100 > $L/g0_gradscale.log 2>&1 &
CUDA_VISIBLE_DEVICES=$G1 $PY --smoke overfit --out $W/overfit --override hazard_steps=100 > $L/g0_overfit.log 2>&1 &
( OV="hazard_steps=20 ckpt_every=20"
  CUDA_VISIBLE_DEVICES=$G2 $PY --smoke resume --out $W/resume_A --override steps=40 $OV > $L/g0_resume_A.log 2>&1; echo "A exit $?"
  CUDA_VISIBLE_DEVICES=$G2 $PY --smoke resume --out $W/resume_B --override steps=20 $OV > $L/g0_resume_B1.log 2>&1; echo "B1 exit $?"
  CUDA_VISIBLE_DEVICES=$G2 $PY --smoke resume --out $W/resume_B --override steps=40 $OV > $L/g0_resume_B2.log 2>&1; echo "B2 exit $?" ) &
wait
echo G0-TRAIN-SMOKES-DONE
