#!/bin/bash
# CVPR overnight -- stage 2 inside Slurm allocation 3489889 (gpu114): MACARONS baseline (P1) + K = 1 noise-set replicates (P2).
# Waits for the Table 3 phase-1 marker, then (Table 1 queue = closed loop -> official replay -> canonical record; resumable):
#   GPU 0     one worker: MACARONS dunnottar (alone) -> MACARONS rest -> noise-replicate dunnottar chain (alone; N1 depth 2 first)
#   GPU 1-3   worker A: MACARONS rest -> noise replicates (rest);   worker B: noise replicates (rest), started immediately
# One MACARONS process per GPU only (it holds ~23 GB of GPU memory; a second process next to closed-loop planners caused the
# 05:25 out-of-memory incident). Noise replicates run in the pre-declared order N1 (depth 2, depth 1), N2, N3 until the window ends.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/cvpr_results/table1/scripts/t1_queue.sh; J=$R/work_dirs/cvpr_results/overnight_20261110/jobs
L=$R/work_dirs/cvpr_results/overnight_20261110/logs; host=$(hostname -s)
echo "PHASE2_WAITING $host pid $$ $(date '+%F %T')" >> $L/table3_chain.log
while ! grep -q "^T3_PHASE1_DONE" $L/table3_chain.log; do sleep 20; done
echo "PHASE2_START $host slurm_job 3489889 (MACARONS 1 worker / GPU + noise replicates) $(date '+%F %T')" >> $L/table3_chain.log
(bash $Q 0 $J/mac_dun.txt; bash $Q 0 $J/mac_rest.txt; echo "MACARONS_GPU0_DONE $(date '+%F %T')" >> $L/table3_chain.log; bash $Q 0 $J/t3n_dun.txt) > /dev/null 2>&1 &
for g in 1 2 3; do
  (bash $Q $g $J/mac_rest.txt; bash $Q $g $J/t3n_rest.txt) > /dev/null 2>&1 & sleep 3
  (bash $Q $g $J/t3n_rest.txt) > /dev/null 2>&1 & sleep 3
done
wait
echo "PHASE2_DONE $host $(date '+%F %T')" >> $L/table3_chain.log
