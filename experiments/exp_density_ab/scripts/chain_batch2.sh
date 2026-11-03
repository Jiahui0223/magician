#!/bin/bash
# EXP Density A/B -- start G3 batch-2 workers on gpu115 once every batch-1 episode is claimed (priority order of
# FROZEN_PROTOCOL §4), keeping at most 2 queue workers per GPU (1 on GPUs 0/2/3 and 0 on GPU 1 while a fixed-state
# process is still running). Ends when all batch-2 episodes are claimed.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
X=experiments/exp_density_ab; L=work_dirs/exp_density_ab/logs; C=work_dirs/exp_density_ab/g3/claims
Q=$X/scripts/da_g3_queue.sh; J2=$X/configs/g3_batch2.txt
n_claimed() { ls $C 2>/dev/null | grep -c -E "_($1)_"; }
B1="B2_S10_D100|E2S_D100_S10|E2S_D100_D100|E2D_D100_D100"; B2="E2S_S10_D100|E2D_S10_S10|E2D_S10_D100"
until [ "$(n_claimed "$B1")" -ge 160 ]; do sleep 60; done
echo "batch1 fully claimed $(date '+%F %T')" >> $L/chain.log
while [ "$(n_claimed "$B2")" -lt 120 ]; do
  fixed=$(pgrep -f "[d]a_fixed.py" | wc -l)
  for g in 0 1 2 3; do
    if [ "$fixed" -gt 0 ]; then cap=1; [ $g = 1 ] && cap=0; else cap=2; fi
    n=$(ps -eo args | grep "[d]a_g3_queue.sh $g " | wc -l)
    if [ "$n" -lt "$cap" ]; then nohup bash $Q $g $J2 > /dev/null 2>&1 & echo "batch2 worker gpu$g $(date '+%F %T')" >> $L/chain.log; sleep 5; fi
  done
  sleep 60
done
echo "batch2 fully claimed $(date '+%F %T')" >> $L/chain.log
