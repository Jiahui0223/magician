#!/bin/bash
# EXP5 Task 0 -- one Matterport3D download stream (run inside tmux on borah-login; resumable, safe to re-run).
# $1 = stream name (log file stem), $2 = file with scan ids, remaining args = file types.
# The terms-of-use prompt of download_mp_py3.py is answered here on behalf of the licensee (authorization email
# received; user instruction 2026-10-23); every call records a "tos_confirmed" event in the JSON-lines log.
set -u
REPO=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
OUT=/bsuhome/jiahuizhang/scratch/data/matterport3d
L=$REPO/work_dirs/exp5_future_view/task0/download_logs
name=$1; ids=$2; shift 2
cd $REPO
until printf '\n' | python3 -I experiments/exp5_future_view/task0/scripts/download_mp_py3.py -o $OUT --ids-file $ids \
      --type "$@" --extract --log $L/$name.jsonl >> $L/$name.out 2>&1; do
  echo "$(date '+%F %T') stream $name exited non-zero, retrying in 300 s" >> $L/$name.out
  sleep 300
done
echo "$(date '+%F %T') stream $name complete" >> $L/$name.out
