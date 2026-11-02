#!/bin/bash
# post_arm_v5.sh v5|v4c : after the arm's training ends cleanly on this node -> publish EMA@12000 as <arm>_final.pt ->
# G3 closed loops (4 workers on GPUs 0-3; other nodes may join the same job file through claims) -> outdoor coverage ->
# G1 / G2 fixed states (VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts) -> (V5) evaluation-side belief GT + belief metrics.
arm=$1; GRP=$([ $arm = v5 ] && echo V5 || echo V4C); OPT=$([ $arm = v5 ] && echo --v5 || echo --v4c)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5; SC=$R/experiments/exp6_v5/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "^EXIT" $W/logs/train_$arm.log; do sleep 60; done
code=$(grep "^EXIT" $W/logs/train_$arm.log | tail -1 | awk '{print $2}')
say "$GRP training ended with exit $code on $(hostname -s)"
[ "$code" = "0" ] || { say "ABORT: nonzero exit"; exit 1; }
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/select_final.py $arm || { say "ABORT: publish failed"; exit 1; }
ck=$W/checkpoints/${arm}_final.pt; [ -f $ck ] || { say "ABORT: no final"; exit 1; }
say "G3 queue $GRP on GPUs 0-3 ($(wc -l < $W/g3_jobs_$GRP.txt) jobs)"
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_jobs_$GRP.txt & done; wait
say "closed loops of this node finished; coverage pass"
for g in 0 1 2 3; do bash $SC/run_queue_v5.sh $g $W/g3_cov_$GRP.txt & done; wait
say "G3 $GRP: $(ls $W/eval/closed_loop/indoor/$GRP/V2M 2>/dev/null | wc -l) indoor, $(ls $W/eval/closed_loop/outdoor/$GRP/V2M/episodes 2>/dev/null | wc -l) outdoor, $(ls $W/eval/metrics/outdoor/$GRP/V2M 2>/dev/null | wc -l) metrics"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_g1_g2.py --set $1 --stratum $2 $OPT $ck --tag s_final_$arm --part $g/4 > $W/logs/eval_fixed_${arm}_$1_$2_$g.log 2>&1 &
  done; wait
  say "eval_g1_g2 $arm $1 $2: $(ls $W/eval_fixed/$1_s_final_$arm/$2/*.jsonl 2>/dev/null | wc -l) episode files"
done
if [ $arm = v5 ]; then
  unset LD_PRELOAD LD_LIBRARY_PATH
  i=0; for d in $W/eval_fixed/valsel_s_final_v5/B0 $W/eval_fixed/dev3_s_final_v5/B0 $W/eval_fixed/dev3_s_final_v5/B2; do
    CUDA_VISIBLE_DEVICES=$i $MG $SC/eval_belief_gt.py --dir $d > $W/logs/belief_gt_$(basename $(dirname $d))_$(basename $d).log 2>&1 & i=$((i + 1)); done; wait
  say "belief GT done"
  export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib
  CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/eval_belief_metrics.py --ckpt $ck --dirs $W/eval_fixed/valsel_s_final_v5/B0 $W/eval_fixed/dev3_s_final_v5/B0 $W/eval_fixed/dev3_s_final_v5/B2 \
    --out $R/experiments/exp6_v5/reports/belief_metrics_eval.json > $W/logs/belief_metrics.log 2>&1
  say "belief metrics exit $?"
fi
say "POST $GRP DONE"
