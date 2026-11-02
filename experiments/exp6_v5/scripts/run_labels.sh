#!/bin/bash
# run_labels.sh : multi-surface label cache of all Train-6 trajectories (train + S-c), 16 peeled layers, <= 8 stored surfaces, 1,024 stored rays per view,
# on the 4 GPUs of this node (6 scene processes). Output work_dirs/exp6_v5/labels/v1/<scene>/<traj>.npz.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; SC=$R/experiments/exp6_v5/scripts; L=$R/work_dirs/exp6_v5/labels/v1
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; mkdir -p $L/logs
g=0
for s in bannerman dunnottar liberty pisa redeemer fushimi; do
  CUDA_VISIBLE_DEVICES=$g $MG $SC/build_multisurface_labels.py run --scene $s --K 16 --k-store 8 --n-sel 1024 --out $L > $L/logs/$s.log 2>&1 &
  g=$(( (g + 1) % 4 ))
done
wait
echo "LABELS DONE $(date '+%F %T')"; grep -h "views" $L/logs/*.log | wc -l
