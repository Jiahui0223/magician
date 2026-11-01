#!/bin/bash
# run_g12_arm.sh v4|v31 : G1 / G2 fixed-state rows of ONE arm (eval_fixed_v4.py with that checkpoint only; B2 replay check
# included; VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts on GPUs 0-3 of this node) under the tag s_final_<arm>. The two arms'
# rows are merged by analyze_g12_v4.py --tag s_final_v4 --tag31 s_final_v31 (same states, candidates and noise).
arm=$1; R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v4; SC=$R/experiments/exp6_v4/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
say() { echo "$(date '+%F %T') $*"; }
cd $R; mkdir -p $W/logs
if [ $arm = v4 ]; then ck=$W/checkpoints/final.pt; opt="--v4 $ck"; else ck=$R/work_dirs/exp6_v3_1/checkpoints/final.pt; opt="--v31 $ck"; fi
[ -f $ck ] || { say "ABORT: $ck missing"; exit 1; }
say "G1/G2 $arm start on $(hostname -s): $(sha256sum $ck | cut -c1-12)"
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_fixed_v4.py --set $1 --stratum $2 $opt --tag s_final_$arm --part $g/4 > $W/logs/eval_fixed_${arm}_$1_$2_$g.log 2>&1 &
  done; wait
  say "eval_fixed_v4 $arm $1 $2 done: $(ls $W/eval_fixed/$1_s_final_$arm/$2/*.jsonl 2>/dev/null | wc -l) episode files"
done
say "G12 $arm DONE"
