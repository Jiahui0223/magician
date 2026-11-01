#!/bin/bash
# Experiment A G3 worker: a_run_queue.sh GPU JOBFILE. Job lines: cl_in:A1:<scan>:<level>:<start> | cl_out:A1:<scene>:<start> |
# cov_out:A1:<scene>:<start>. A1 = the frozen V4 final.pt through the V2 planner path with T3_P1_MODE=full (D100 input AND
# D100 DistP1 band_dil); everything else = the V4 G3 run (A0). Outputs under work_dirs/exp6_v4/ablation_study/experiment_A/eval.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
W6=$R/work_dirs/exp6_v4/ablation_study/experiment_A/eval
SC=$R/experiments/exp6_v4/scripts
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
ck=$R/work_dirs/exp6_v4/checkpoints/final.pt
g=$1; jf=$2; cd $R; mkdir -p $W6/claims $W6/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V4_P1_MODE=full V4_EVAL_ROOT=$W6
while read -r it; do
  IFS=: read kind grp b c d <<< "$it"
  [ "$grp" = "A1" ] || continue
  case $kind in
    cl_in)  done_f=$W6/closed_loop/indoor/$grp/V2M/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=$W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
    cov_out) done_f=$W6/metrics/outdoor/$grp/V2M/V2M_${b}_s${c}_r0.json; [ -f $W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ] || continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > $W6/claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  case $kind in
    cl_in)  CUDA_VISIBLE_DEVICES=$g $MG $SC/sim_in_v4.py --v4-ckpt $ck --scan $b --level $c --start $d --group V2M --out $W6/closed_loop/indoor/$grp > $W6/logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cov_out) CUDA_VISIBLE_DEVICES=$g $MG $SC/out_cov_v4.py --grp $grp --scene $b --start $c > $W6/logs/cov_out_${grp}_${b}_s${c}.log 2>&1 ;;
    cl_out) CUDA_VISIBLE_DEVICES=$g $MG $SC/sim_out_v4.py --v4-ckpt $ck --group V2M --scene $b --start $c --root $W6/closed_loop/outdoor/$grp/V2M > $W6/logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) $(date '+%F %T')" >> $W6/logs/queue_$(hostname -s)_$g.log
done < $jf
