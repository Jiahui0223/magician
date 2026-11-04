#!/bin/bash
# EXP6 V3 / H -- G2 worker: eval_fixed.py on one GPU for one stratum part (resumable). Usage: run_g2.sh GPU STRATUM PART
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=$R/work_dirs/exp6_v3/H
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p $W/logs; p=${3/\//of}
CUDA_VISIBLE_DEVICES=$1 $S2E/bin/python experiments/exp6_v3/H/scripts/eval_fixed.py --stratum $2 --part $3 >> $W/logs/g2_$2_$p.log 2>&1
echo "EXIT $? $(hostname -s) $(date '+%F %T')" >> $W/logs/g2_$2_$p.log
