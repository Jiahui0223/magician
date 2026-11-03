#!/bin/bash
# EXP6 V2 (B2 / E2) -- G3 closed-loop worker. Tracked, parameterised form of the historical launchers
# work_dirs/exp5_future_view/task3_full_depth_p1/run_queue2.sh (B2) and .../e2/run_queue_g3.sh (E2-D100 / E2-S10):
# same simulators (task3 t33_in_sim.py / t33_out_sim.py -> task2 t23 / t24 + t32 planners), same P1 modes, same K / n_win /
# moves / seeds. Only differences: the checkpoint is read from the unified registry (work_dirs/checkpoints/exp6_v2, symlinks
# to the original bytes, sha verified by the planner core) and the output root defaults to a NEW folder, so historical
# results are never overwritten.
# Usage: [MOVES=n] g3_queue.sh GPU JOBFILE [OUT_ROOT] (MOVES: smoke only)     job lines: cl_in:<arm>:<scan>:<level>:<start> | cl_out:<arm>:<scene>:<start>
#        arm b2 = B2 (V2-Mix EMA@17.5k, P1 b2) | E2D = E2-D100 EMA@5k (P1 b2) | E2S = E2-S10 EMA@5k (P1 sparse, control)
# Coverage of outdoor episodes afterwards: task3 t35_out_cov.py --grp <B2|E2D|E2S> (reads the closed-loop root of Task 3;
# run it only when OUT_ROOT is the Task 3 root -- see B2/README.md).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp5_future_view/task3_full_depth_p1/scripts
CK=$R/work_dirs/checkpoints/exp6_v2
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); OUT=$(readlink -m ${3:-$R/work_dirs/exp6_v2/g3})
mkdir -p $OUT/claims $OUT/logs; cd $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind arm b c d <<< "$it"
  case $arm in
    b2|B2) grp=B2; p1=b2; ck=$CK/B2/v2mix_ema17500.pt ;;
    E2D) grp=E2D; p1=b2; ck=$CK/E2/e2d100_ema5000.pt ;;
    E2S) grp=E2S; p1=sparse; ck=$CK/E2/control_e2s10/e2s10_ema5000.pt ;;
    *) echo "unknown arm $arm"; exit 2 ;;
  esac
  case $kind in
    cl_in)  done_f=closed_loop/indoor/$grp/V2M/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  case $kind in
    cl_in)  T3_CKPT=$ck CUDA_VISIBLE_DEVICES=$g $MG $SC/t33_in_sim.py --p1 $p1 --scan $b --level $c --start $d --group V2M ${MOVES:+--moves $MOVES} --out $OUT/closed_loop/indoor/$grp > logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cl_out) T3_CKPT=$ck CUDA_VISIBLE_DEVICES=$g $MG $SC/t33_out_sim.py --p1 $p1 --group V2M --scene $b --start $c ${MOVES:+--moves $MOVES} --root $OUT/closed_loop/outdoor/$grp/V2M > logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) $(date '+%F %T')" >> logs/queue_$(hostname -s)_$g.log
done < $jf
