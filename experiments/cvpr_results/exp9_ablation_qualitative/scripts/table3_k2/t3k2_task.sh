#!/bin/bash
# T3K2 gpu_queue task body (2026-11-16): closed loops of V3-T0-K2D{1,2}-N2 through the Table 1 queue (t1_queue.sh: closed loop -> official
# coverage replay (must equal the stored curve) -> canonical record; noclobber claims shared by every task, <= 3 tries; resumable).
# Usage: t3k2_task.sh NAME PHASES      PHASES = phase[,phase...] run one after the other (barrier between phases);
#   phase = list[+list...]:N  -> N parallel t1_queue.sh workers on this worker's GPU, each running the lists in order.
# Lists: work_dirs/cvpr_results/exp9_ablation_qualitative/t3k2/jobs/<list>.txt (unseen = Unseen-6, dun = dunnottar, rest = the other 7 scenes).
# Each task works on private copies of the lists (jobs/run/); on the gpu115 allocation (job 3497835, ends 2026-11-16 19:23:22) the copies are
# emptied at 18:45, so no new episode starts there after that (running episodes finish; the remaining jobs stay for the gpu114 workers).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
Q=$R/experiments/cvpr_results/table1/scripts/t1_queue.sh
B=$R/work_dirs/cvpr_results/exp9_ablation_qualitative/t3k2; J=$B/jobs; L=$B/logs
name=$1; g=$CUDA_VISIBLE_DEVICES
mkdir -p $J/run $L
echo "T3K2_TASK_START $name $(hostname -s) gpu$g worker $WORKER_ID phases $2 pid $$ $(date '+%F %T')" >> $L/t3k2_chain.log
copies=()
for ph in ${2//,/ }; do
  lists=${ph%%:*}
  for lst in ${lists//+/ }; do cp $J/$lst.txt $J/run/${name}_$lst.txt || exit 3; copies+=($J/run/${name}_$lst.txt); done
done
wd=""
if [[ $WORKER_ID == 3497835_* ]]; then
  ( dl=$(date -d '2026-11-16 18:45:00' +%s); while [ $(date +%s) -lt $dl ]; do sleep 60; done
    for c in "${copies[@]}"; do : > $c; done; echo "T3K2_DEADLINE $name lists emptied $(date '+%F %T')" >> $L/t3k2_chain.log ) &
  wd=$!
fi
for ph in ${2//,/ }; do
  lists=${ph%%:*}; n=${ph##*:}; pids=()
  echo "T3K2_PHASE $name $lists x$n $(date '+%F %T')" >> $L/t3k2_chain.log
  for w in $(seq 1 $n); do
    ( for lst in ${lists//+/ }; do bash $Q $g $J/run/${name}_$lst.txt; done ) > /dev/null 2>&1 < /dev/null &
    pids+=($!); sleep 3
  done
  wait "${pids[@]}"
done
[ -n "$wd" ] && kill $wd 2>/dev/null
echo "T3K2_TASK_END $name $(hostname -s) gpu$g $(date '+%F %T')" >> $L/t3k2_chain.log
exit 0
