#!/bin/bash
# G0 training smokes of V5.2-M (pinned smoke trajectories, per-rank batch of an 8-rank run = 8 views / 1 state):
#   GPU 0 overfit 200 steps | GPU 1 gradient probe (beta) | GPU 2 single-process resume A 30 vs B 15 + 15 | GPU 3 short 200 (throughput)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
PY="$S2E/bin/python experiments/exp6_v5_2_scratch70k/scripts/train_v52m.py"; W=work_dirs/exp6_v5_2_scratch70k/runs/g0; L=work_dirs/exp6_v5_2_scratch70k/logs
rm -rf $W; mkdir -p $W $L
CUDA_VISIBLE_DEVICES=0 $PY --smoke overfit --out $W/overfit > $L/g0_overfit.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY --smoke gradscale --out $W/gradscale > $L/g0_gradscale.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY --smoke short --out $W/short > $L/g0_short.log 2>&1 &
( CUDA_VISIBLE_DEVICES=2 $PY --smoke resume --out $W/resume_A --override steps=30 ckpt_every=15 > $L/g0_resume_A.log 2>&1; echo "A exit $?"
  CUDA_VISIBLE_DEVICES=2 $PY --smoke resume --out $W/resume_B --override steps=15 ckpt_every=15 > $L/g0_resume_B1.log 2>&1; echo "B1 exit $?"
  CUDA_VISIBLE_DEVICES=2 $PY --smoke resume --out $W/resume_B --override steps=30 ckpt_every=15 > $L/g0_resume_B2.log 2>&1; echo "B2 exit $?" ) &
wait; echo G0-TRAIN-DONE
