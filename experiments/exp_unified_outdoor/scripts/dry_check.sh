#!/bin/bash
# dry_check.sh : CPU-only dry check of every unified arm (no GPU: CUDA_VISIBLE_DEVICES is emptied). Exit 0 iff all pass.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R || exit 2
export CUDA_VISIBLE_DEVICES="" LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=8
mkdir -p $R/work_dirs/exp_unified_outdoor/verify
$S2E/bin/python experiments/exp_unified_outdoor/scripts/dry_check.py "$@" 2>&1 | tee $R/work_dirs/exp_unified_outdoor/verify/dry_check.log
exit ${PIPESTATUS[0]}
