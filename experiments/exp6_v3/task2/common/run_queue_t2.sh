#!/bin/bash
# Task 2 G3 worker: run_queue_t2.sh GPU JOBFILE EVALROOT
# Job lines: in:<arm>:<model>:<scorer>:<frontier>:<lambda>:<theta>:<scan>:<level>:<start>
#            out:<arm>:<model>:<scorer>:<scene>:<start> | cov:<arm>:<scene>:<start>
# <arm> names the output group dir (<EVALROOT>/closed_loop/{indoor,outdoor}/<arm>/V2M/...); one process per GPU; shared claims.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp6_v3/task2/common
g=$1; jf=$2; WE=$3
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; mkdir -p $WE/claims $WE/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind arm f3 f4 f5 f6 f7 f8 f9 f10 <<< "$it"
  case $kind in
    in)  done_f=$WE/closed_loop/indoor/$arm/V2M/${f8}_L${f9}_S${f10}/episode.json ;;
    out) done_f=$WE/closed_loop/outdoor/$arm/V2M/episodes/ep_V2M_${f5}_s${f6}_r0.json ;;
    cov) done_f=$WE/metrics/outdoor/$arm/V2M/V2M_${f3}_s${f4}_r0.json; [ -f $WE/closed_loop/outdoor/$arm/V2M/episodes/ep_V2M_${f3}_s${f4}_r0.json ] || continue ;;
    *) continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$ $(date '+%F %T')" > $WE/claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  t0=$(date +%s)
  case $kind in
    in)  CUDA_VISIBLE_DEVICES=$g $MG $SC/t2_in_sim.py --model $f3 --scorer $f4 --frontier $f5 --lambda $f6 --theta $f7 --scan $f8 --level $f9 --start $f10 --out $WE/closed_loop/indoor/$arm > $WE/logs/in_${arm}_${f8}_L${f9}_S${f10}.log 2>&1 ;;
    out) CUDA_VISIBLE_DEVICES=$g $MG $SC/t2_out_sim.py --model $f3 --scorer $f4 --scene $f5 --start $f6 --root $WE/closed_loop/outdoor/$arm/V2M > $WE/logs/out_${arm}_${f5}_s${f6}.log 2>&1 ;;
    cov) CUDA_VISIBLE_DEVICES=$g $MG $SC/t2_out_cov.py --root $WE/closed_loop/outdoor/$arm/V2M --out $WE/metrics/outdoor/$arm --scene $f3 --start $f4 > $WE/logs/cov_${arm}_${f3}_s${f4}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WE/logs/queue_$(hostname -s)_$g.log
done < $jf
echo "QUEUE DONE $(hostname -s) gpu$g $(date '+%F %T')" >> $WE/logs/queue_$(hostname -s)_$g.log
