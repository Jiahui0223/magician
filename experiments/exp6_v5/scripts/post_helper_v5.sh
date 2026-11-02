#!/bin/bash
# post_helper_v5.sh : on the node that finished its own chain (V4C, gpu115): once "POST V4C DONE" and v5_final.pt exist,
# join the V5 G3 queue (4 workers, claims shared with the V5 node), then its coverage pass.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5; SC=$R/experiments/exp6_v5/scripts
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "POST V4C DONE" $W/logs/post_v4c.log 2>/dev/null && [ -f $W/checkpoints/v5_final.pt ]; do sleep 120; done
say "joining the V5 G3 queue on $(hostname -s)"
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_jobs_V5.txt & done; wait
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_cov_V5.txt & done; wait
say "HELPER DONE"
