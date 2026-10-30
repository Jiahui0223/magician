#!/bin/bash
# EXP5 Task 0 -- move the Matterport3D copy from /bsushare (2 TB limit) to the user's scratch data folder
# (user decision 2026-10-23). $1 = stream id 0..3: rsync of every 4th scan (plus the top-level files in stream 0).
# Copy only; the source is deleted separately after verify_mp3d.py has passed.
SRC=/bsushare/yzhang-shared/data/matterport3d
DST=/bsuhome/jiahuizhang/scratch/data/matterport3d
L=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/download_logs/migrate_$1.log
mkdir -p $DST/v1/scans
echo "$(date '+%F %T') start stream $1" >> $L
if [ "$1" = "0" ]; then rsync -a --no-g --no-o --exclude v1/scans $SRC/ $DST/ >> $L 2>&1; fi
ls $SRC/v1/scans | awk -v i=$1 'NR % 4 == i' | while read s; do
  rsync -a --no-g --no-o --partial $SRC/v1/scans/$s $DST/v1/scans/ >> $L 2>&1
  echo "$(date '+%F %T') $s rc=$?" >> $L
done
echo "$(date '+%F %T') stream $1 done" >> $L
