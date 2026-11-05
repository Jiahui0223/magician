#!/bin/bash
# worker: run_c.sh GPU scene:start ...
cd /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; export LD_PRELOAD=/bsuscratch/jiahuizhang/envs/seen2scene/lib/libtbb.so LD_LIBRARY_PATH=/bsuscratch/jiahuizhang/envs/seen2scene/lib PYTHONPATH=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
g=$1; shift
for sk in "$@"; do s=${sk%:*}; k=${sk#*:}; CUDA_VISIBLE_DEVICES=$g /bsuscratch/jiahuizhang/envs/seen2scene/bin/python /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/cvpr_results/geometry_roi_audit/scripts/c_ray_audit.py --scene $s --start $k > /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/cvpr_results/geometry_roi_audit/logs/c_${s}_s${k}.log 2>&1; echo "exit $? $s $k $(date +%T)" >> /bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/cvpr_results/geometry_roi_audit/logs/c_queue.log; done
