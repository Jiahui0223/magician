#!/bin/bash
# extra_workers_v5.sh GRP [wait_file] : 4 additional G3 workers on this node (GPUs 0-3, i.e. 2 per GPU next to the chain's
# own workers; claim files prevent duplicates) for the closed loops of GRP, then its coverage pass. With wait_file, start
# only once that file exists (the published checkpoint). The per-user QOS caps us at 8 GPUs, so we add workers instead.
GRP=$1; WF=$2
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5; SC=$R/experiments/exp6_v5/scripts
cd $R
[ -n "$WF" ] && until [ -f $WF ]; do sleep 60; done
echo "$(date '+%F %T') extra workers $GRP on $(hostname -s)"
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_jobs_$GRP.txt & done; wait
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_cov_$GRP.txt & done; wait
echo "$(date '+%F %T') extra workers $GRP done"
