#!/bin/bash
# G3 worker for the Outdoor-only arms: run_queue_v4.sh GPU JOBFILE. Job lines: cl_in:<grp>:<scan>:<level>:<start> |
# cl_out:<grp>:<scene>:<start> | cov_out:<grp>:<scene>:<start>; grp V4 -> work_dirs/exp6_v4/checkpoints/final.pt via the V2
# planner path (v4 sims), grp V31 -> work_dirs/exp6_v3_1/checkpoints/final.pt via the V3 planner path (exp6_v3/shared sims);
# both B2-style (D100 input, sparse DistP1). Outputs under work_dirs/exp6_v4/eval/{closed_loop,metrics}/.../<grp>/V2M.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
W6=$R/work_dirs/exp6_v4/eval
SC=$R/experiments/exp6_v4/scripts; SC3=$R/experiments/exp6_v3/shared
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$2; cd $R; mkdir -p $W6/claims $W6/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r it; do
  IFS=: read kind grp b c d <<< "$it"
  case $grp in V4) ck=$R/work_dirs/exp6_v4/checkpoints/final.pt; SI="$SC/sim_in_v4.py --v4-ckpt"; SO="$SC/sim_out_v4.py --v4-ckpt" ;;
               V31) ck=$R/work_dirs/exp6_v3_1/checkpoints/final.pt; SI="$SC3/sim_in_v3.py --v3-ckpt"; SO="$SC3/sim_out_v3.py --v3-ckpt" ;; esac
  case $kind in
    cl_in)  done_f=$W6/closed_loop/indoor/$grp/V2M/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=$W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
    cov_out) done_f=$W6/metrics/outdoor/$grp/V2M/V2M_${b}_s${c}_r0.json; [ -f $W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ] || continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > $W6/claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  case $kind in
    cl_in)  CUDA_VISIBLE_DEVICES=$g $MG $SI $ck --scan $b --level $c --start $d --group V2M --out $W6/closed_loop/indoor/$grp > $W6/logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cov_out) CUDA_VISIBLE_DEVICES=$g $MG $SC/out_cov_v4.py --grp $grp --scene $b --start $c > $W6/logs/cov_out_${grp}_${b}_s${c}.log 2>&1 ;;
    cl_out) CUDA_VISIBLE_DEVICES=$g $MG $SO $ck --group V2M --scene $b --start $c --root $W6/closed_loop/outdoor/$grp/V2M > $W6/logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) $(date '+%F %T')" >> $W6/logs/queue_g3_$(hostname -s)_$g.log
done < $jf
