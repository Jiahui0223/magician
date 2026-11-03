#!/bin/bash
# EXP Density A/B -- after the G0 replay: (1) STOP unless every replayed episode is bitwise identical to history;
# (2) restart the OOM-killed VALSEL part 1/3 (finished episodes are skipped); (3) start G3 batch 1 workers
# (1 per GPU while the fixed-state run is still active, 2 per GPU afterwards).
source /bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp_density_ab/scripts/env.sh
cd $R
L=work_dirs/exp_density_ab/logs; X=experiments/exp_density_ab
until [ "$(cat work_dirs/exp_density_ab/g0_replay/logs/queue_* 2>/dev/null | wc -l)" -ge 10 ]; do sleep 30; done
$PY $X/scripts/da_g0_compare.py > $L/g0_compare.log 2>&1
ok=$($PY -c "
import json; d=json.load(open('$X/results/g0_closed_loop.json'))['replay']
ok = len(d) == 5 and all(v.get('indoor', {}).get('executed_equal') and v['indoor'].get('coverage_equal') and v.get('outdoor', {}).get('executed_keys_equal')
         and v['outdoor'].get('coverage_official_equal') and v['outdoor'].get('cov_mono_equal') for v in d.values())
print('PASS' if ok else 'FAIL')")
echo "G0 $ok $(date '+%F %T')" >> $L/chain.log
[ "$ok" = PASS ] || exit 1
CUDA_VISIBLE_DEVICES=1 $PY $X/scripts/da_fixed.py --set valsel --part 1/3 > $L/fixed_valsel_1b.log 2>&1 &
VP=$!
Q=$X/scripts/da_g3_queue.sh; J=$X/configs/g3_batch1.txt
for g in 0 2 3; do nohup bash $Q $g $J > /dev/null 2>&1 & sleep 3; done
echo "batch1 workers x3 started $(date '+%F %T')" >> $L/chain.log
wait $VP
until ! pgrep -f "[d]a_fixed.py" > /dev/null; do sleep 30; done
echo "FIXED_DONE_ALL $(date '+%F %T')" >> $L/chain.log
for g in 0 1 1 2 3; do nohup bash $Q $g $J > /dev/null 2>&1 & sleep 3; done
echo "batch1 workers +5 started (total 8) $(date '+%F %T')" >> $L/chain.log
