#!/bin/bash
# CVPR Table 3 (K x planning depth) -- closed-loop workers inside the Slurm allocation of the node (gpu114, job 3489889, 4 x L40S).
# Uses the Table 1 queue (experiments/cvpr_results/table1/scripts/t1_queue.sh): per episode closed loop -> official coverage
# replay (must equal the stored curve bitwise) -> canonical record work_dirs/cvpr_results/table1/episodes/<MODEL>/; noclobber
# claims, <= 3 tries per episode, resumable (finished records are skipped, so re-running this script only fills gaps).
#   phase 1  the three missing Table 3 cells (K4D1 and K1D2 first, K1D1 last)       jobs t3_dun.txt / t3_rest.txt
#   phase 2  pre-declared K = 1 noise-set replicates (noise maps 1, 2, 3)            jobs t3n_dun.txt / t3n_rest.txt
# GPU 0: ONE worker runs every dunnottar episode (planner peak 30-40 GB, needs the GPU alone), then the remaining lists.
# GPUs 1-3: three workers each on the non-dunnottar lists.   Usage: run_table3.sh [phase2]   (default: phase 1 then phase 2)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/cvpr_results/table1/scripts/t1_queue.sh; J=$R/work_dirs/cvpr_results/overnight_20261110/jobs
L=$R/work_dirs/cvpr_results/overnight_20261110/logs; host=$(hostname -s)
echo "T3_RUN_START $host slurm_job ${SLURM_JOB_ID:-3489889} pid $$ $(date '+%F %T')" >> $L/table3_chain.log
if [ "$1" != "phase2" ]; then
  (bash $Q 0 $J/t3_dun.txt; bash $Q 0 $J/t3_rest.txt) > /dev/null 2>&1 &
  for g in 1 2 3; do for w in 1 2 3; do (bash $Q $g $J/t3_rest.txt) > /dev/null 2>&1 & sleep 3; done; done
  wait
  echo "T3_PHASE1_DONE $host $(date '+%F %T')" >> $L/table3_chain.log
fi
(bash $Q 0 $J/t3n_dun.txt; bash $Q 0 $J/t3n_rest.txt) > /dev/null 2>&1 &
for g in 1 2 3; do for w in 1 2 3; do (bash $Q $g $J/t3n_rest.txt) > /dev/null 2>&1 & sleep 3; done; done
wait
echo "T3_PHASE2_DONE $host $(date '+%F %T')" >> $L/table3_chain.log
