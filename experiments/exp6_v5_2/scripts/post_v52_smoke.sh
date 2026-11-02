#!/bin/bash
# post_v52_smoke.sh (second node): once v52_final.pt is published and score_v52.json is frozen (version score_v52_v1)
# -> pre-registered G3 smoke closed loops (V52L0 / V52L01; the V5 / V4C VALSEL references if not done) -> coverage pass.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5_2; SC=$R/experiments/exp6_v5_2/scripts
say() { echo "$(date '+%F %T') $*"; }
cd $R
until [ -f $W/checkpoints/v52_final.pt ] && grep -q '"score_v52_v1"' $R/experiments/exp6_v5_2/configs/score_v52.json; do
  grep -q "STOP\|ABORT" $W/logs/post_v52_main.log 2>/dev/null && { say "main chain stopped -- no smoke"; exit 1; }; sleep 30; done
say "G3 smoke start on $(hostname -s)"
for g in 0 1 2 3; do bash $SC/run_queue_v52.sh $g $W/g3_smoke_jobs.txt & done; wait
for g in 0 1 2 3; do bash $SC/run_queue_v52.sh $g $W/g3_smoke_cov.txt & done; wait
say "SMOKE DONE: $(ls $W/eval/closed_loop/indoor/*/V2M 2>/dev/null | wc -l) indoor dirs, $(ls $W/eval/metrics/outdoor/*/V2M 2>/dev/null | wc -l) outdoor metrics"
