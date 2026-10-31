#!/bin/bash
# post_g12.sh : once T0's G3 queue has finished on this node and T1's final.pt exists -> G1 / G2 fixed-state evaluation
# of T0 + T1 (eval_fixed.py: VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts on GPUs 0-3) -> analyze_g12.py -> this node joins
# T1's G3 queue (claims shared with the gpu115 workers) -> T1 outdoor coverage pass.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v3; SC=$R/experiments/exp6_v3/shared
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
say() { echo "$(date '+%F %T') $*"; }
cd $R
T0=$W/task0_vae/checkpoints/final.pt; T1=$W/task1_xattn/checkpoints/final.pt
until grep -q "POST T0 DONE" $W/logs/post_t0.log 2>/dev/null && [ -f $T1 ]; do sleep 120; done
say "G1/G2 start on $(hostname -s): T0 $(sha256sum $T0 | cut -c1-12) T1 $(sha256sum $T1 | cut -c1-12)"
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_fixed.py --set $1 --stratum $2 --t0 $T0 --t1 $T1 --part $g/4 > $W/logs/eval_fixed_$1_$2_$g.log 2>&1 &
  done; wait
  say "eval_fixed $1 $2 done: $(ls $W/eval_fixed/$1_s_final/$2/*.jsonl 2>/dev/null | wc -l) episode files"
done
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g12.py --tag s_final > $W/logs/analyze_g12.log 2>&1; say "analyze_g12 exit $? (reports/g12_s_final.md)"
say "G12 DONE -> joining T1 G3 queue"
for g in 0 1 2 3; do bash $W/run_queue_g3.sh $g $W/g3_jobs_T1.txt & done; wait
t_end=$(( $(date +%s) + 6 * 3600 ))
until { [ $(ls $W/eval/closed_loop/outdoor/T1/V2M/episodes/*.json 2>/dev/null | wc -l) -ge 30 ] && [ $(ls $W/eval/closed_loop/indoor/T1/V2M/*/episode.json 2>/dev/null | wc -l) -ge 10 ]; } || [ $(date +%s) -gt $t_end ]; do sleep 120; done
for g in 0 1 2 3; do bash $W/run_queue_g3.sh $g $W/g3_cov_T1.txt & done; wait
say "POST G12+T1G3 DONE: T1 indoor $(ls $W/eval/closed_loop/indoor/T1/V2M/*/episode.json 2>/dev/null | wc -l) outdoor $(ls $W/eval/closed_loop/outdoor/T1/V2M/episodes/*.json 2>/dev/null | wc -l) metrics $(ls $W/eval/metrics/outdoor/T1/V2M/*.json 2>/dev/null | wc -l)"
