#!/bin/bash
# a_run_fixed.sh : fixed-state A0 / A1 rows (VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts on GPUs 0-3 of this node) -> a_fixed_analyze.py
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; A=$R/experiments/exp6_v4/ablation_study/experiment_A_distp1_dense_vs_sparse; W=$R/work_dirs/exp6_v4/ablation_study/experiment_A
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
say() { echo "$(date '+%F %T') $*"; }
cd $R; mkdir -p $W/logs
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $A/scripts/a_fixed.py --set $1 --stratum $2 --part $g/4 > $W/logs/a_fixed_$1_$2_$g.log 2>&1 &
  done; wait
  say "a_fixed $1 $2 done: $(ls $W/fixed/$1/$2/*.jsonl 2>/dev/null | wc -l) episode files; STOPs: $(grep -l 'STOP' $W/logs/a_fixed_$1_$2_*.log 2>/dev/null | wc -l)"
done
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $A/scripts/a_fixed_analyze.py > $W/logs/a_fixed_analyze.log 2>&1; say "a_fixed_analyze exit $?"
say "A FIXED DONE"
