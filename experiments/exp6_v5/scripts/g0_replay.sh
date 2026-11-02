#!/bin/bash
# G0-1 / G0-11: (a) V4 replay through the V5 evaluation harness (v5_core installed; V4 final.pt as the V4C arm) on the
# first VALSEL episode and the first Dev-3 B0 episode -> compared with the exp6_v4 V4 rows; (b) the same episodes with
# the untrained V5 smoke checkpoint (planner-core path, belief dump); (c) 2-move closed-loop planner smokes (MP3D and
# Macarons++) with the V5 smoke checkpoint. GPUs given as $1 $2.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene; SC=$R/experiments/exp6_v5/scripts
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; L=work_dirs/exp6_v5/logs; E=work_dirs/exp6_v5/eval
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
V4=work_dirs/exp6_v4/checkpoints/final.pt; V5I=work_dirs/exp6_v5/checkpoints/smoke_v5_init.pt
g1=$1; g2=$2
( CUDA_VISIBLE_DEVICES=$g1 $S2E/bin/python $SC/eval_g1_g2.py --set valsel --stratum B0 --v4c $V4 --tag g0_v4replay --part 0/9 > $L/g0_v4replay_valsel.log 2>&1; echo "v4replay valsel exit $?"
  CUDA_VISIBLE_DEVICES=$g1 $S2E/bin/python $SC/eval_g1_g2.py --set dev3 --stratum B0 --v4c $V4 --tag g0_v4replay --part 0/15 > $L/g0_v4replay_dev3.log 2>&1; echo "v4replay dev3 exit $?" ) &
( CUDA_VISIBLE_DEVICES=$g2 $S2E/bin/python $SC/eval_g1_g2.py --set valsel --stratum B0 --v5 $V5I --tag g0_v5init --part 0/9 > $L/g0_v5init_valsel.log 2>&1; echo "v5init valsel exit $?"
  CUDA_VISIBLE_DEVICES=$g2 $S2E/bin/python $SC/eval_g1_g2.py --set dev3 --stratum B0 --v5 $V5I --tag g0_v5init --part 0/15 > $L/g0_v5init_dev3.log 2>&1; echo "v5init dev3 exit $?" ) &
wait
mkdir -p $E/smoke/indoor $E/smoke/outdoor/V2M
unset LD_PRELOAD LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=$g1 $MG $SC/sim_in_v5.py --v4-ckpt $V5I --scan Z6MFQCViBuw --level 0 --start 6 --group V2M --moves 2 --out $E/smoke/indoor > $L/g0_smoke_in_v5.log 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=$g2 $MG $SC/sim_out_v5.py --v4-ckpt $V5I --group V2M --scene barts --start 0 --moves 2 --root $E/smoke/outdoor/V2M > $L/g0_smoke_out_v5.log 2>&1 & p2=$!
wait $p1; e1=$?; wait $p2; e2=$?
echo "planner smoke indoor exit $e1 outdoor exit $e2"
echo G0-REPLAY-DONE
