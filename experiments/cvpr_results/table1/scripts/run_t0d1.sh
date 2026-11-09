#!/bin/bash
# CVPR Table 1 -- V3-T0-D1 workers on one node (4 GPUs). Per GPU: worker A runs dunnottar first (planner peak ~40 GB, must be
# alone on its GPU) and then the other 13 scenes; workers B and C start once this GPU has no unfinished dunnottar claim, then run
# the other scenes. Shared noclobber claims prevent duplicates; failures are released and retried by t1_queue.sh (<= 3 tries).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=experiments/cvpr_results/table1/scripts/t1_queue.sh; C=experiments/cvpr_results/table1/configs
T1W=$R/work_dirs/cvpr_results/table1/g3; host=$(hostname -s)
gate() {   # block while this GPU holds a dunnottar claim without a finished record
  while true; do
    busy=0
    for c in $T1W/claims/V3-T0-D1_dunnottar_*.claim; do
      [ -f "$c" ] || continue; grep -q "^$host:$1:" "$c" || continue
      k=${c##*_}; k=${k%.claim}; [ -f $R/work_dirs/cvpr_results/table1/episodes/V3-T0-D1/dunnottar_s$k.json ] || busy=1
    done
    [ $busy = 0 ] && return; sleep 60
  done
}
echo "T0D1_RUN_START $host $(date '+%F %T')" >> $T1W/logs/chain.log
for g in 0 1 2 3; do
  (bash $Q $g $C/jobs_t0d1_dun.txt; bash $Q $g $C/jobs_t0d1_rest.txt) > /dev/null 2>&1 & sleep 5
  (sleep 120; gate $g; bash $Q $g $C/jobs_t0d1_rest.txt) > /dev/null 2>&1 & sleep 2
  (sleep 120; gate $g; bash $Q $g $C/jobs_t0d1_rest.txt) > /dev/null 2>&1 & sleep 2
done
wait
echo "T0D1_RUN_DONE $host $(date '+%F %T')" >> $T1W/logs/chain.log
