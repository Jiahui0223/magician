#!/bin/bash
# T0 on this node; T1 starts on the auxiliary allocation as soon as one of the holder jobs runs (parallel), otherwise
# on this node after T0. Both arms resume automatically from ckpt_last.pt if relaunched.
# Tracked copy of the launcher; the running copy is work_dirs/exp6_v3/train_chain.sh (identical content).
W=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp6_v3
cd $W
bash $W/train_arm.sh t0 &
T0PID=$!
t1_started=""
while kill -0 $T0PID 2>/dev/null; do
  for j in 3435033 3435072; do
    if [ -z "$t1_started" ] && [ "$(squeue -h -j $j -o %T 2>/dev/null)" = "RUNNING" ]; then
      echo "aux job $j running -> T1 in parallel $(date '+%F %T')"
      setsid srun --jobid=$j --overlap -N1 -n1 bash $W/train_arm.sh t1 > $W/logs/srun_train_t1.log 2>&1 < /dev/null &
      t1_started=$j
    fi
  done
  sleep 120
done
echo "T0 finished $(date '+%F %T')"
if [ -z "$t1_started" ]; then
  echo "no aux allocation -> T1 on this node $(date '+%F %T')"
  bash $W/train_arm.sh t1
fi
echo "TRAIN CHAIN DONE $(date '+%F %T')"
