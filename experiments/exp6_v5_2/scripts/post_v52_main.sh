#!/bin/bash
# post_v52_main.sh (node with the training): EXIT 0 -> publish EMA@6000 (+ collapse check) -> S-c floor calibration
# (freezes configs/score_v52.json) -> G1 / G2 fixed states (VALSEL B0, Dev-3 B0, Dev-3 B2; 4 parts) -> eval-side belief
# GT (magician) -> belief metrics -> G1 / G2 analysis.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5_2; SC=$R/experiments/exp6_v5_2/scripts
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
say() { echo "$(date '+%F %T') $*"; }
cd $R
until grep -q "^EXIT" $W/logs/train_v52.log; do sleep 30; done
code=$(grep "^EXIT" $W/logs/train_v52.log | tail -1 | awk '{print $2}'); say "training exit $code"
[ "$code" = "0" ] || { say "ABORT: training failed"; exit 1; }
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/select_final.py; c=$?
[ $c = 0 ] || { say "STOP: publish failed or Stage-J collapse (exit $c) -- see checkpoints/v52_final.json"; exit 1; }
CK=$W/checkpoints/v52_final.pt
CUDA_VISIBLE_DEVICES=0 $S2E/bin/python $SC/calibrate_floors.py --ckpt $CK > $W/logs/calibrate_floors.log 2>&1 || { say "ABORT: calibration failed"; exit 1; }
say "floors frozen: $(grep -A4 '"floors"' $W/logs/calibrate_floors.log | tr -d ' \n' | head -c 200)"
for sx in "valsel B0" "dev3 B0" "dev3 B2"; do
  set -- $sx
  for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g $S2E/bin/python $SC/eval_g1_g2.py --set $1 --stratum $2 --ckpt $CK --tag s_final --part $g/4 > $W/logs/eval_fixed_$1_$2_$g.log 2>&1 & done; wait
  say "G1/G2 $1 $2: $(ls $W/eval_fixed/$1_s_final/$2/*.jsonl 2>/dev/null | wc -l) episode files"
done
unset LD_PRELOAD LD_LIBRARY_PATH
i=0; for d in $W/eval_fixed/valsel_s_final/B0 $W/eval_fixed/dev3_s_final/B0 $W/eval_fixed/dev3_s_final/B2; do
  CUDA_VISIBLE_DEVICES=$i $MG $R/experiments/exp6_v5/scripts/eval_belief_gt.py --dir $d > $W/logs/belief_gt_$(basename $(dirname $d))_$(basename $d).log 2>&1 & i=$((i + 1)); done; wait
say "belief GT done"
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/eval_belief_metrics_v52.py --ckpt $CK --dirs $W/eval_fixed/valsel_s_final/B0 $W/eval_fixed/dev3_s_final/B0 $W/eval_fixed/dev3_s_final/B2 \
  --out $R/experiments/exp6_v5_2/reports/belief_metrics_eval.json > $W/logs/belief_metrics.log 2>&1; say "belief metrics exit $?"
CUDA_VISIBLE_DEVICES="" $S2E/bin/python $SC/analyze_g1_g2.py > $W/logs/analyze_g1_g2.log 2>&1; say "G1/G2 analysis exit $?"
say "POST MAIN DONE"
