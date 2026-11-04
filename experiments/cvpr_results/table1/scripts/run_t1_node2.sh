#!/bin/bash
# CVPR Table 1 -- extra workers on a second node (gpu114, user job 3461041, GPUs 0-3): 3 workers per GPU, main list first,
# then MAG-B, then optional. Shares claims / outputs with the gpu115 workers (run_t1.sh).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/cvpr_results/table1/scripts/t1_queue.sh; C=experiments/cvpr_results/table1/configs
for g in 0 1 2 3; do
  for w in 1 2 3; do (bash $Q $g $C/jobs_main.txt; bash $Q $g $C/jobs_magb.txt; bash $Q $g $C/jobs_optional.txt) > /dev/null 2>&1 & sleep 5; done
done
wait
echo "T1_RUN_DONE_$(hostname -s) $(date '+%F %T')" >> work_dirs/cvpr_results/table1/g3/logs/chain.log
