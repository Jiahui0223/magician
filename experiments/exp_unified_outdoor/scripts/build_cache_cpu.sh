#!/bin/bash
# EXP Unified Outdoor -- S10 4x TSDF cache on CPU while the GPUs run Stage 1 (24 single-thread workers; same builder,
# claims and bitwise coarse-grid check as build_tsdf_cache_s10.sh; finished trajectories are skipped on rerun).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
W=$R/work_dirs/exp_unified_outdoor; mkdir -p $W/logs
for i in $(seq 0 23); do
  nice -n 5 $S2E/bin/python experiments/exp_unified_outdoor/scripts/build_tsdf_cache_s10.py $i 24 --device cpu >> $W/logs/cache_s10_cpu_w$i.log 2>&1 &
done
wait
echo "CPU cache: $(find $W/cache/tsdf4_s10/outdoor -name '*.npz' | wc -l) / 180 $(date '+%F %T')" >> $W/logs/chain.log
