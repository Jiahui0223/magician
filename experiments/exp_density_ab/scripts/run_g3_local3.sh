#!/bin/bash
# EXP Density A/B -- G3 workers v3 on gpu115: 2 da_g3_queue3 workers per GPU, batch 1 then batch 2 (FROZEN_PROTOCOL §4, §9).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/exp_density_ab/scripts/da_g3_queue3.sh; C=experiments/exp_density_ab/configs
for g in 0 1 2 3; do
  for w in 1 2; do (bash $Q $g $C/g3_batch1.txt; bash $Q $g $C/g3_batch2.txt) > /dev/null 2>&1 & sleep 9; done
done
wait
echo "G3_LOCAL_DONE $(date '+%F %T')" >> work_dirs/exp_density_ab/logs/chain.log
