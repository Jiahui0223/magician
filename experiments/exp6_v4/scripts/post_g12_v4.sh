#!/bin/bash
# post_g12_v4.sh : once V4's post chain (G3) has finished on this node and V3.1's final.pt exists -> G1 / G2 of both arms
# (run_g12_v4.sh, 4 GPUs) -> once V3.1's G3 has finished too -> analyze_g3_v4.py. Tracked copy.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v4; SC=$R/experiments/exp6_v4/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "POST V4 DONE" $W/logs/post_v4.log 2>/dev/null && [ -f $R/work_dirs/exp6_v3_1/checkpoints/final.pt ]; do sleep 180; done
say "both final.pt present and V4 G3 done -> G1/G2 on $(hostname -s)"
bash $SC/run_g12_v4.sh > $W/logs/run_g12_v4.log 2>&1; say "run_g12_v4 exit $?"
until grep -q "POST V31 DONE" $W/logs/post_v31.log 2>/dev/null; do sleep 180; done
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g3_v4.py > $W/logs/analyze_g3_v4.log 2>&1; say "analyze_g3_v4 exit $? (exp6_v4/reports/g3_v4.md)"
say "POST G12+G3 ANALYSIS DONE"
