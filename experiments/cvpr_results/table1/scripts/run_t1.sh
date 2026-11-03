#!/bin/bash
# CVPR Table 1 -- launch the missing-episode workers on gpu115 (4 GPUs): per GPU two workers start with MAG-B (slowest,
# ~30 min / episode) and one with the other main models; every worker then continues with the remaining lists
# (main -> MAG-B -> optional). Shared noclobber claims prevent duplicates.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/cvpr_results/table1/scripts/t1_queue.sh; C=experiments/cvpr_results/table1/configs
for g in 0 1 2 3; do
  (bash $Q $g $C/jobs_magb.txt; bash $Q $g $C/jobs_main.txt; bash $Q $g $C/jobs_optional.txt) > /dev/null 2>&1 & sleep 5
  (bash $Q $g $C/jobs_magb.txt; bash $Q $g $C/jobs_main.txt; bash $Q $g $C/jobs_optional.txt) > /dev/null 2>&1 & sleep 5
  (bash $Q $g $C/jobs_main.txt; bash $Q $g $C/jobs_magb.txt; bash $Q $g $C/jobs_optional.txt) > /dev/null 2>&1 & sleep 5
done
wait
echo "T1_RUN_DONE $(date '+%F %T')" >> work_dirs/cvpr_results/table1/g3/logs/chain.log
