#!/bin/bash
# G0-11: 2-move closed-loop planner smokes with a V5.2 checkpoint ($1) and lambda ($2): MP3D VALSEL floor (8194nk5LbLH L0,
# start 10 -- the first pre-registered smoke floor) and Macarons++ Test-3 (alhambra s0); then check the step logs: guard
# counts all zero, no flow / DistP1 call, finite scores.
CK=$1; LAM=$2; TAG=${3:-g0}
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; SC=$R/experiments/exp6_v5_2/scripts; E=$R/work_dirs/exp6_v5_2/eval_smoke/$TAG
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; rm -rf $E; mkdir -p $E/indoor $E/outdoor/V2M; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=0 $MG $SC/sim_in_v52.py --v4-ckpt $CK --lam $LAM --scan 8194nk5LbLH --level 0 --start 10 --group V2M --moves 2 --out $E/indoor > $E/smoke_in.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=1 $MG $SC/sim_out_v52.py --v4-ckpt $CK --lam $LAM --group V2M --scene alhambra --start 0 --moves 2 --root $E/outdoor/V2M > $E/smoke_out.log 2>&1 & p2=$!
wait $p1; e1=$?; wait $p2; e2=$?
echo "indoor exit $e1 outdoor exit $e2"
find $E -path "*v52_steps*" -name "*.jsonl" | while read f; do echo "== $f"; /bsuscratch/jiahuizhang/envs/seen2scene/bin/python -c "
import json, sys
for x in open('$f'):
    r = json.loads(x); print('n', r['n'], 'guards', r['guard_counts'], 'score', [round(s, 3) for s in r['distp1']], 'vis', [round(s, 1) for s in r['v52_s_vis']], 'pot_share', [round(s, 3) for s in r['v52_parts']['pot_share']], 't %.2fs' % r['timing']['v52_score_s'])
"; done
grep -h "V5.2 Core\|Traceback\|Error" $E/indoor/*/planner.err $E/outdoor/V2M/*/*.err 2>/dev/null | head -5
echo SMOKE-DONE
