#!/bin/bash
# EXP Temporal Refresh -- unattended Stage-1 chain (OVERNIGHT_PLAN §3). Waits for the Density A/B G3 workers to end, then:
#  (1) fixed-state G1/G2 (tr_fixed.py, 7 processes on the 4 GPUs);
#  (2) gate: all 39 episodes written and every R10 regression check true, else BLOCKED (stop, no G3);
#  (3) G3: 8 tr_g3_queue workers (2 per GPU, <= 1 MP3D episode per GPU): R10 replay (2 episodes) -> R2/R1 (160) -> R5 (80).
# Status lines go to work_dirs/exp_temporal_refresh/logs/chain.log.
source /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp_density_ab/scripts/env.sh
cd $R
X=experiments/exp_temporal_refresh; L=work_dirs/exp_temporal_refresh/logs; C=$X/configs
log() { echo "$1 $(date '+%F %T')" >> $L/chain.log; }
until grep -q "G3_LOCAL_DONE" work_dirs/exp_density_ab/logs/chain.log 2>/dev/null && ! pgrep -f "[d]a_g3_queue3.sh" > /dev/null; do sleep 60; done
log "S1 start (Density A/B G3 workers ended)"
S=$X/scripts/tr_fixed.py
CUDA_VISIBLE_DEVICES=0 $PY $S --set valsel --part 0/3 > $L/fixed_valsel_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S --set valsel --part 1/3 > $L/fixed_valsel_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 $PY $S --set valsel --part 2/3 > $L/fixed_valsel_2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $S --set dev3 --stratum B0 --part 0/2 > $L/fixed_dev3_B0_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 $PY $S --set dev3 --stratum B0 --part 1/2 > $L/fixed_dev3_B0_1.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $PY $S --set dev3 --stratum B2 --part 0/2 > $L/fixed_dev3_B2_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $PY $S --set dev3 --stratum B2 --part 1/2 > $L/fixed_dev3_B2_1.log 2>&1 &
wait
n=$(ls work_dirs/exp_temporal_refresh/fixed/*/*/*.jsonl 2>/dev/null | wc -l)
nok=$(cat $L/fixed_*.log | grep -c "all ok True")
nbad=$(cat $L/fixed_*.log | grep -c "all ok False\|STOP\|Traceback")
log "S1 fixed done: episodes $n / 39, logged all-ok $nok, problems $nbad"
if [ "$n" -lt 39 ] || [ "$nbad" -gt 0 ]; then
  # one retry for episodes lost to an infrastructure error (finished episodes are skipped); then gate again
  for p in "valsel B0" "dev3 B0" "dev3 B2"; do set -- $p; CUDA_VISIBLE_DEVICES=2 $PY $S --set $1 --stratum $2 >> $L/fixed_retry.log 2>&1; done
  n=$(ls work_dirs/exp_temporal_refresh/fixed/*/*/*.jsonl 2>/dev/null | wc -l)
  nbad=$(grep -c "all ok False\|STOP" $L/fixed_*.log $L/fixed_retry.log | awk -F: '{s+=$2} END {print s}')
  log "S1 fixed after retry: episodes $n / 39, stop/false lines $nbad"
  if [ "$n" -lt 39 ] || [ "$nbad" -gt 0 ]; then log "S1 BLOCKED: fixed-state gate failed, no G3"; exit 1; fi
fi
log "S1 G3 start"
Q=$X/scripts/tr_g3_queue.sh
for g in 0 1 2 3; do
  for w in 1 2; do (bash $Q $g $C/g3_r10_replay.txt; bash $Q $g $C/g3_main.txt; bash $Q $g $C/g3_r5.txt) > /dev/null 2>&1 & sleep 9; done
done
wait
log "S1_G3_DONE"
