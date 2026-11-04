#!/bin/bash
# EXP6 V5 / F1 -- G1/G2 worker: eval_g12.py on one GPU for one stratum part (resumable: finished tags are skipped).
# Usage: run_g12.sh GPU STRATUM PART   (e.g. run_g12.sh 0 B0 0/4)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; W=$R/work_dirs/exp6_v5/F1
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4
mkdir -p $W/logs
p=${3/\//of}
echo "start g12 $2 $3 $(hostname -s) gpu$1 $(date '+%F %T')" >> $W/logs/g12_$2_$p.log
CUDA_VISIBLE_DEVICES=$1 $S2E/bin/python experiments/exp6_v5/F1/scripts/eval_g12.py --stratum $2 --part $3 >> $W/logs/g12_$2_$p.log 2>&1
echo "EXIT $? $(hostname -s) $(date '+%F %T')" >> $W/logs/g12_$2_$p.log
