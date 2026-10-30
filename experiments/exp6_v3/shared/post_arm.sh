#!/bin/bash
# post_arm.sh t0|t1 : after the arm's training ends cleanly -> frozen checkpoint selection (final.pt) -> 2-move planner
# smokes (eval/smoke, not part of the results) -> G3 closed loops (4 workers, GPUs 0-3 of this node) -> outdoor coverage.
arm=$1; A=$(echo $arm | tr a-z A-Z); T=$([ $arm = t0 ] && echo task0_vae || echo task1_xattn)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v3; W6=$W/eval; SC=$R/experiments/exp6_v3/shared
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "^EXIT" $W/$T/logs/train.log; do sleep 120; done
code=$(grep "^EXIT" $W/$T/logs/train.log | tail -1 | awk '{print $2}')
say "$A training ended with exit $code on $(hostname -s)"
[ "$code" = "0" ] || { say "ABORT: nonzero exit"; exit 1; }
PYTHONPATH=$R CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/select_final.py $arm 2>&1 | grep -v -i warn || { say "ABORT: selection failed"; exit 1; }
ck=$W/$T/checkpoints/final.pt; [ -f $ck ] || { say "ABORT: no final.pt"; exit 1; }
say "final.pt: $(grep -E 'chosen_step|sha256' $W/$T/checkpoints/final.json | tr -d ' \n')"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W6/smoke/indoor/$A $W6/smoke/outdoor/$A/V2M $W6/logs
CUDA_VISIBLE_DEVICES=0 $MG $SC/sim_in_v3.py --v3-ckpt $ck --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 2 --out $W6/smoke/indoor/$A > $W6/logs/smoke_in_$A.log 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 $MG $SC/sim_out_v3.py --v3-ckpt $ck --group V2M --scene barts --start 0 --moves 2 --root $W6/smoke/outdoor/$A/V2M > $W6/logs/smoke_out_$A.log 2>&1 &
p2=$!
wait $p1; e1=$?; wait $p2; e2=$?
say "SMOKE indoor exit $e1 outdoor exit $e2 (logs eval/logs/smoke_*_$A.log)"
[ "$e1" = "0" ] && [ "$e2" = "0" ] || { say "SMOKE FAIL -> stopped before the G3 queue"; exit 1; }
say "SMOKE OK -> G3 queue $A on GPUs 0-3 ($(wc -l < $W/g3_jobs_$A.txt) jobs)"
for g in 0 1 2 3; do bash $W/run_queue_g3.sh $g $W/g3_jobs_$A.txt & done; wait
say "closed loops finished; outdoor coverage pass"
for g in 0 1 2 3; do bash $W/run_queue_g3.sh $g $W/g3_cov_$A.txt & done; wait
say "POST $A DONE: $(ls $W6/closed_loop/indoor/$A/V2M 2>/dev/null | wc -l) indoor dirs, $(ls $W6/closed_loop/outdoor/$A/V2M/episodes 2>/dev/null | wc -l) outdoor episodes, $(ls $W6/metrics/outdoor/$A/V2M 2>/dev/null | wc -l) metrics"
