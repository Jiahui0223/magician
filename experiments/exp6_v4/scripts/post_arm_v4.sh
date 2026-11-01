#!/bin/bash
# post_arm_v4.sh v4|v31 : after the arm's training ends cleanly -> publish the frozen EMA@40k as final.pt -> 2-move planner
# smokes (eval/smoke, not results) -> G3 closed loops (4 workers on GPUs 0-3 of this node) -> outdoor coverage.
arm=$1; GRP=$([ $arm = v4 ] && echo V4 || echo V31); WA=$([ $arm = v4 ] && echo exp6_v4 || echo exp6_v3_1)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/$WA; W6=$R/work_dirs/exp6_v4/eval; SC=$R/experiments/exp6_v4/scripts; SC3=$R/experiments/exp6_v3/shared
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "^EXIT" $W/logs/train.log; do sleep 120; done
code=$(grep "^EXIT" $W/logs/train.log | tail -1 | awk '{print $2}')
say "$GRP training ended with exit $code on $(hostname -s)"
[ "$code" = "0" ] || { say "ABORT: nonzero exit"; exit 1; }
PYTHONPATH=$R CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/select_final.py $arm 2>&1 | grep -v -i warn || { say "ABORT: publish failed"; exit 1; }
ck=$W/checkpoints/final.pt; [ -f $ck ] || { say "ABORT: no final.pt"; exit 1; }
say "final.pt: $(grep -E 'chosen_step|sha256' $W/checkpoints/final.json | tr -d ' \n')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W6/smoke/indoor/$GRP $W6/smoke/outdoor/$GRP/V2M $W6/logs
if [ $arm = v4 ]; then SI="$SC/sim_in_v4.py --v4-ckpt"; SO="$SC/sim_out_v4.py --v4-ckpt"; else SI="$SC3/sim_in_v3.py --v3-ckpt"; SO="$SC3/sim_out_v3.py --v3-ckpt"; fi
CUDA_VISIBLE_DEVICES=0 $MG $SI $ck --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 2 --out $W6/smoke/indoor/$GRP > $W6/logs/smoke_in_$GRP.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 $MG $SO $ck --group V2M --scene barts --start 0 --moves 2 --root $W6/smoke/outdoor/$GRP/V2M > $W6/logs/smoke_out_$GRP.log 2>&1 &
p2=$!
wait $p1; e1=$?; wait $p2; e2=$?
say "SMOKE indoor exit $e1 outdoor exit $e2"
[ "$e1" = "0" ] && [ "$e2" = "0" ] || { say "SMOKE FAIL -> stopped before the G3 queue"; exit 1; }
say "SMOKE OK -> G3 queue $GRP on GPUs 0-3 ($(wc -l < $R/work_dirs/exp6_v4/g3_jobs_$GRP.txt) jobs)"
for g in 0 1 2 3; do bash $SC/run_queue_v4.sh $g $R/work_dirs/exp6_v4/g3_jobs_$GRP.txt & done; wait
say "closed loops finished; outdoor coverage pass"
for g in 0 1 2 3; do bash $SC/run_queue_v4.sh $g $R/work_dirs/exp6_v4/g3_cov_$GRP.txt & done; wait
say "POST $GRP DONE: $(ls $W6/closed_loop/indoor/$GRP/V2M 2>/dev/null | wc -l) indoor dirs, $(ls $W6/closed_loop/outdoor/$GRP/V2M/episodes 2>/dev/null | wc -l) outdoor episodes, $(ls $W6/metrics/outdoor/$GRP/V2M 2>/dev/null | wc -l) metrics"
