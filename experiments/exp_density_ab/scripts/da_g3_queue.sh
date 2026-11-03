#!/bin/bash
# EXP Density A/B -- G3 closed-loop worker (FROZEN_PROTOCOL.md §4). Same simulators / settings as the historical Task 3 and
# E2 runs (task3 t33 sims -> task2 t23 / t24 -> task1), planner child = da_in/out_planner.py (P1 mode via da_p1).
# Usage: [MOVES=n] da_g3_queue.sh GPU JOBFILE [OUT_ROOT]
#   job lines: cl_in:<CFG>:<scan>:<level>:<start> | cl_out:<CFG>:<scene>:<start>  (cl_out also runs the coverage replay)
#   CFG = <CKPT>_<INF>_<SCORE>, CKPT in B2 | E2S | E2D, INF / SCORE in S10 | D100
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp_density_ab/scripts
E2T=$R/work_dirs/exp5_future_view/task3_full_depth_p1/e2/training
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); OUT=$(readlink -m ${3:-$R/work_dirs/exp_density_ab/g3})
mkdir -p $OUT/claims $OUT/logs; cd $OUT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DA_G3_ROOT=$OUT
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind cfg b c d <<< "$it"
  IFS=_ read ck inf sco <<< "$cfg"
  case $ck in
    B2) ckenv="" ;;                                            # group V2M checkpoint = V2-Mix EMA@17.5k (historical B0/B1/B2)
    E2S) ckenv="T3_CKPT=$E2T/s10/ema_005000.pt" ;;
    E2D) ckenv="T3_CKPT=$E2T/d100/ema_005000.pt" ;;
    *) echo "unknown ckpt $ck"; exit 2 ;;
  esac
  mode=$(echo ${inf}${sco} | tr 'A-Z' 'a-z')
  case $mode in s10s10|d100s10|d100d100|s10d100) ;; *) echo "bad mode $mode"; exit 2 ;; esac
  case $kind in
    cl_in)  done_f=closed_loop/indoor/$cfg/V2M/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=metrics/outdoor/$cfg/V2M/V2M_${b}_s${c}_r0.json ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  t0=$(date +%s)
  case $kind in
    cl_in)  env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_in_sim.py --p1 $mode --scan $b --level $c --start $d --group V2M ${MOVES:+--moves $MOVES} --out $OUT/closed_loop/indoor/$cfg > logs/cl_in_${cfg}_${b}_L${c}_S${d}.log 2>&1; rc=$? ;;
    cl_out) env $ckenv CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_sim.py --p1 $mode --group V2M --scene $b --start $c ${MOVES:+--moves $MOVES} --root $OUT/closed_loop/outdoor/$cfg/V2M > logs/cl_out_${cfg}_${b}_s${c}.log 2>&1; rc=$?
            if [ $rc -eq 0 ] && [ -z "$MOVES" ]; then CUDA_VISIBLE_DEVICES=$g $MG $SC/da_out_cov.py --cfg $cfg --scene $b --start $c > logs/cov_${cfg}_${b}_s${c}.log 2>&1; rc=$?; fi ;;
  esac
  echo "exit $rc $it $(hostname -s) gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> logs/queue_$(hostname -s)_$g.log
done < $jf
