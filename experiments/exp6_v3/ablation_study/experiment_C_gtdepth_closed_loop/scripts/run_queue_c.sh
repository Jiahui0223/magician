#!/bin/bash
# Experiment C worker: run_queue_c.sh GPU JOBFILE [EVALROOT]
# Job lines: cl_in:<grp>:<scan>:<level>:<start> | cl_out:<grp>:<scene>:<start> | cov_out:<grp>:<scene>:<start>   (grp C1 | C2)
# One process per GPU; claims shared across nodes (atomic noclobber); finished jobs skipped (done file).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp6_v3/ablation_study/experiment_C_gtdepth_closed_loop/scripts
g=$1; jf=$2; WE=${3:-$R/work_dirs/exp6_v3/ablation_study/experiment_C/eval}
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; mkdir -p $WE/claims $WE/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind grp b c d <<< "$it"
  case $kind in
    cl_in)   done_f=$WE/closed_loop/indoor/$grp/${b}_L${c}_S${d}/episode.json ;;
    cl_out)  done_f=$WE/closed_loop/outdoor/$grp/episodes/ep_${grp}_${b}_s${c}_r0.json ;;
    cov_out) done_f=$WE/metrics/outdoor/$grp/${grp}_${b}_s${c}_r0.json; [ -f $WE/closed_loop/outdoor/$grp/episodes/ep_${grp}_${b}_s${c}_r0.json ] || continue ;;
    *) continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$ $(date '+%F %T')" > $WE/claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  t0=$(date +%s)
  case $kind in
    cl_in)   CUDA_VISIBLE_DEVICES=$g $MG $SC/c_in_sim.py --group $grp --scan $b --level $c --start $d --out $WE/closed_loop/indoor > $WE/logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cl_out)  CUDA_VISIBLE_DEVICES=$g $MG $SC/c_out_sim.py --group $grp --scene $b --start $c --root $WE/closed_loop/outdoor/$grp > $WE/logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
    cov_out) CUDA_VISIBLE_DEVICES=$g $MG $SC/c_out_cov.py --grp $grp --scene $b --start $c --root $WE/closed_loop/outdoor/$grp --out $WE/metrics/outdoor > $WE/logs/cov_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $WE/logs/queue_$(hostname -s)_$g.log
done < $jf
echo "QUEUE DONE $(hostname -s) gpu$g $(date '+%F %T')" >> $WE/logs/queue_$(hostname -s)_$g.log
