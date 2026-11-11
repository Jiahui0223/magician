#!/bin/bash
# exits (prints a reason) on the first failed try / OOM / Traceback, or when every unit of both job lists has an episode.json
W7=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/exp7
RAW=$W7/episodes/mp3d150/raw
while true; do
  f=$(grep -h "^exit [1-9]\|FAILED_3" $W7/logs/worker_*.log 2>/dev/null | head -3)
  [ -n "$f" ] && { echo "FAILURE: $f"; exit 1; }
  o=$(grep -l "out of memory\|OutOfMemoryError" $W7/logs/episodes/*.log $RAW/*/*/*/planner.err 2>/dev/null | head -3)
  [ -n "$o" ] && { echo "OOM: $o"; exit 1; }
  n=0; tot=0
  for jl in $W7/jobs/mp3d150_slotA.txt $W7/jobs/mp3d150_slotB.txt; do
    while IFS=: read m s l st; do
      tot=$((tot+1)); case $m in OURS) d=OURS/V2M;; OURS-H0) d=OURS-H0/V2M;; MAG-B) d=MAG-B/MAG-B;; RANDOM) d=RANDOM/G-RAND;; esac
      [ -f $RAW/$d/${s}_L${l}_S${st}/episode.json ] && n=$((n+1))
    done < $jl
  done
  [ $n -ge $tot ] && { echo "ALL DONE $n/$tot $(date '+%F %T')"; exit 0; }
  sleep 120
done
