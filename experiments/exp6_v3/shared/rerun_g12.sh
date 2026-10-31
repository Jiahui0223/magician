#!/bin/bash
# rerun_g12.sh : complete the G1 / G2 fixed-state rows (eval_fixed.py skips finished episodes; frozen bitwise replay rule,
# EVAL_FIXED_REPLAY_TOL unset = 0), up to 3 passes for transient STOPs, then analyze_g12.py.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v3; SC=$R/experiments/exp6_v3/shared
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
say() { echo "$(date '+%F %T') $*"; }
cd $R
T0=$W/task0_vae/checkpoints/final.pt; T1=$W/task1_xattn/checkpoints/final.pt
want() { case "$1_$2" in valsel_B0) echo 9 ;; dev3_B0) echo 15 ;; dev3_B2) echo 15 ;; esac; }
for pass in 1 2 3; do
  for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
    set -- $sx
    n=$(ls $W/eval_fixed/$1_s_final/$2/*.jsonl 2>/dev/null | wc -l); [ $n -ge $(want $1 $2) ] && continue
    for g in 0 1 2 3; do
      CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_fixed.py --set $1 --stratum $2 --t0 $T0 --t1 $T1 --part $g/4 > $W/logs/eval_fixed_${1}_${2}_${g}_pass$pass.log 2>&1 &
    done; wait
    say "pass $pass $1 $2: $(ls $W/eval_fixed/$1_s_final/$2/*.jsonl 2>/dev/null | wc -l)/$(want $1 $2) episode files; STOPs: $(grep -h "^STOP" $W/logs/eval_fixed_${1}_${2}_*_pass$pass.log | cut -c1-160 | tr '\n' ' ')"
  done
done
say "files: valsel $(ls $W/eval_fixed/valsel_s_final/B0/*.jsonl | wc -l)/9 dev3 B0 $(ls $W/eval_fixed/dev3_s_final/B0/*.jsonl | wc -l)/15 dev3 B2 $(ls $W/eval_fixed/dev3_s_final/B2/*.jsonl | wc -l)/15"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g12.py --tag s_final > $W/logs/analyze_g12_rerun.log 2>&1; say "analyze_g12 exit $?"
say "RERUN G12 DONE"
