#!/bin/bash
# run_g12_v4.sh : G1 / G2 fixed-state evaluation of V3.1 + V4 (eval_fixed_v4.py: VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts on
# GPUs 0-3 of this node) -> analyze_g12_v4.py. Requires both final.pt files. Tracked copy.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v4; SC=$R/experiments/exp6_v4/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
say() { echo "$(date '+%F %T') $*"; }
cd $R; mkdir -p $W/logs
V31=$R/work_dirs/exp6_v3_1/checkpoints/final.pt; V4=$W/checkpoints/final.pt
[ -f $V31 ] && [ -f $V4 ] || { say "ABORT: final.pt missing ($V31 / $V4)"; exit 1; }
say "G1/G2 start on $(hostname -s): V31 $(sha256sum $V31 | cut -c1-12) V4 $(sha256sum $V4 | cut -c1-12)"
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_fixed_v4.py --set $1 --stratum $2 --v31 $V31 --v4 $V4 --part $g/4 > $W/logs/eval_fixed_$1_$2_$g.log 2>&1 &
  done; wait
  say "eval_fixed_v4 $1 $2 done: $(ls $W/eval_fixed/$1_s_final/$2/*.jsonl 2>/dev/null | wc -l) episode files"
done
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g12_v4.py --tag s_final > $W/logs/analyze_g12_v4.log 2>&1; say "analyze_g12_v4 exit $? (exp6_v4/reports/g12_v4_s_final.md)"
say "G12 V4/V31 DONE"
