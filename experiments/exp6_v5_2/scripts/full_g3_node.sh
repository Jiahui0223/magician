#!/bin/bash
# full_g3_node.sh N : N workers per GPU (GPUs 0-3) on this node for the full G3 of V52L0 / V52L01 (closed loops, then the
# coverage pass); claims are shared with the other node and with the smoke (already finished episodes are skipped).
N=${1:-2}
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5_2; SC=$R/experiments/exp6_v5_2/scripts
cd $R; echo "$(date '+%F %T') full G3 start on $(hostname -s), $N workers / GPU"
for g in 0 1 2 3; do for i in $(seq $N); do bash $SC/run_queue_v52.sh $g $W/g3_full_jobs.txt & done; done; wait
echo "$(date '+%F %T') closed loops of this node done"
for g in 0 1 2 3; do for i in $(seq $N); do bash $SC/run_queue_v52.sh $g $W/g3_full_cov.txt & done; done; wait
echo "$(date '+%F %T') FULL G3 NODE DONE ($(hostname -s))"
