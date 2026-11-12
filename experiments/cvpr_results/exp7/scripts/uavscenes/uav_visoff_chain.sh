#!/bin/bash
# usage: uav_visoff_chain.sh GPU TILE [TILE...]  -- runs uav_visibility_official.py sequentially on one GPU
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R; g=$1; shift
export MACARONS_PLUSPLUS_ROOT=$R/work_dirs/cvpr_results/exp7/uav_macpp
for t in "$@"; do
  CUDA_VISIBLE_DEVICES=$g env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python -I experiments/cvpr_results/exp7/scripts/uavscenes/uav_visibility_official.py $t > $R/work_dirs/cvpr_results/exp7/logs/uav_visoff_$t.log 2>&1
  echo "rc=$?" >> $R/work_dirs/cvpr_results/exp7/logs/uav_visoff_$t.log
done
