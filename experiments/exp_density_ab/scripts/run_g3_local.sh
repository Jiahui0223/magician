#!/bin/bash
# EXP Density A/B -- G3 workers on the interactive node (gpu115): 2 da_g3_queue2 workers per GPU, batch 1 then batch 2.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/exp_density_ab/scripts/da_g3_queue2.sh; C=experiments/exp_density_ab/configs
for g in 0 1 2 3; do
  for w in 1 2; do (bash $Q $g $C/g3_batch1.txt; bash $Q $g $C/g3_batch2.txt) > /dev/null 2>&1 & sleep 7; done
done
wait
echo "G3_LOCAL_DONE $(date '+%F %T')" >> work_dirs/exp_density_ab/logs/chain.log
