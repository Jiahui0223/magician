#!/bin/bash
# EXP7 UAV-v0: CPU stage (uav_build_tile.py) for every kept tile of manifests/uav_tiles_v0.csv. Existing tile folders are skipped.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
tail -n +2 experiments/cvpr_results/exp7/manifests/uav_tiles_v0.csv | tr -d "\r" | while IFS=, read scene tile e0 n0 e1 n1 fp relief keep; do
  [ "$keep" = "1" ] || continue
  t=${tile#${scene}_}
  [ -d work_dirs/cvpr_results/exp7/uav_macpp/uav_${scene}_${t} ] && { echo "skip $tile"; continue; }
  CUDA_VISIBLE_DEVICES="" /bsuscratch/jiahuizhang/envs/magician/bin/python -I experiments/cvpr_results/exp7/scripts/uavscenes/uav_build_tile.py --scene $scene --tile $t --box $e0 $n0 $e1 $n1 > work_dirs/cvpr_results/exp7/logs/uav_build_${tile}.log 2>&1
  echo "$tile rc=$? $(grep -o '"faces_tile": [0-9]*' work_dirs/cvpr_results/exp7/logs/uav_build_${tile}.log) $(grep -o '"aspect_horizontal_over_height": [0-9.]*' work_dirs/cvpr_results/exp7/logs/uav_build_${tile}.log)"
done
