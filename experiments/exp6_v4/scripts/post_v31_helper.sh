#!/bin/bash
# post_v31_helper.sh : second node helping V3.1's evaluation (job 3435072 / gpu115): wait for the gpu117 chain to publish
# final.pt and pass the smokes -> join V31's G3 queue with 4 workers (claims shared) -> wait until all 40 closed loops exist
# -> outdoor coverage pass -> wait for 'POST V31 DONE' -> V31 G1/G2 rows -> analyses (G1/G2 merge, G3, tables, training
# summary). Tracked copy.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v4; W6=$W/eval; SC=$R/experiments/exp6_v4/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "SMOKE OK" $W/logs/post_v31.log 2>/dev/null; do grep -q "ABORT\|SMOKE FAIL" $W/logs/post_v31.log 2>/dev/null && { say "gpu117 chain aborted"; exit 1; }; sleep 60; done
say "joining V31 G3 queue on $(hostname -s)"
for g in 0 1 2 3; do bash $SC/run_queue_v4.sh $g $W/g3_jobs_V31.txt & done; wait
t_end=$(( $(date +%s) + 4 * 3600 ))
until { [ $(ls $W6/closed_loop/outdoor/V31/V2M/episodes/*.json 2>/dev/null | wc -l) -ge 30 ] && [ $(ls $W6/closed_loop/indoor/V31/V2M/*/episode.json 2>/dev/null | wc -l) -ge 10 ]; } || [ $(date +%s) -gt $t_end ]; do sleep 60; done
say "all V31 closed loops present -> coverage pass"
for g in 0 1 2 3; do bash $SC/run_queue_v4.sh $g $W/g3_cov_V31.txt & done; wait
until grep -q "POST V31 DONE" $W/logs/post_v31.log 2>/dev/null; do sleep 60; done
say "V31 G3 complete: $(ls $W6/metrics/outdoor/V31/V2M 2>/dev/null | wc -l) metrics -> G1/G2 v31"
bash $SC/run_g12_arm.sh v31 > $W/logs/run_g12_v31.log 2>&1; say "run_g12_arm v31 exit $?"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g12_v4.py --tag s_final_v4 --tag31 s_final_v31 > $W/logs/analyze_g12_v4.log 2>&1; say "analyze_g12_v4 exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g3_v4.py > $W/logs/analyze_g3_v4.log 2>&1; say "analyze_g3_v4 exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/training_summary.py v31 > $W/logs/training_summary_v31.log 2>&1; say "training_summary v31 exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/make_tables_v4.py --g12 g12_v4_s_final_v4 > $W/logs/make_tables.log 2>&1; say "make_tables exit $?"
say "HELPER V31 ALL DONE"
