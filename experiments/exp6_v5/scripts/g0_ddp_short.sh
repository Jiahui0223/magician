#!/bin/bash
# G0-9 DDP: 4-rank torchrun short run (pinned smoke trajectories, the real per-rank batch 16 views / 2 states) of one arm.
arm=$1; shift
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=$R/work_dirs/exp6_v5
cd $R; export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
rm -rf $W/runs/g0/${arm}_ddp_short
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v5/scripts/train_v5.py --arm $arm --smoke short --out $W/runs/g0/${arm}_ddp_short "$@" > $W/logs/g0_${arm}_ddp_short.log 2>&1
echo "EXIT $? $arm ddp short $(date '+%F %T')"
