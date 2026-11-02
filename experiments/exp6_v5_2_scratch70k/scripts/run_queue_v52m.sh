#!/bin/bash
# G3 worker of the V5.2-M round: run_queue_v52m.sh GPU JOBFILE. Groups V52ML0 / V52ML01 = the evaluated V5.2-M checkpoint
# ($V52M_CKPT) with lambda 0 / 0.1 and its score config ($V52M_SCORE_CFG), V5.2-M planner children (sim_{in,out}_v52m.py);
# frozen B2 planner (D100 input, S10 novelty, K / n_win / R10 unchanged). Outputs $G3_ROOT/{closed_loop,metrics}/... (claims shared).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
W6=${G3_ROOT:-$R/work_dirs/exp6_v5_2_scratch70k/eval/g3}
SC=$R/experiments/exp6_v5_2_scratch70k/scripts; SC4=$R/experiments/exp6_v4/scripts
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$2; cd $R; mkdir -p $W6/claims $W6/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V4_EVAL_ROOT=$W6 V52_SCORE_CFG=$V52M_SCORE_CFG
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind grp b c d <<< "$it"
  case $grp in
    V52ML0)  SI="$SC/sim_in_v52m.py --lam 0 --v4-ckpt"; SO="$SC/sim_out_v52m.py --lam 0 --v4-ckpt" ;;
    V52ML01) SI="$SC/sim_in_v52m.py --lam 0.1 --v4-ckpt"; SO="$SC/sim_out_v52m.py --lam 0.1 --v4-ckpt" ;;
  esac
  ck=$V52M_CKPT
  case $kind in
    cl_in)  done_f=$W6/closed_loop/indoor/$grp/V2M/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=$W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
    cov_out) done_f=$W6/metrics/outdoor/$grp/V2M/V2M_${b}_s${c}_r0.json; [ -f $W6/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ] || continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > $W6/claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  case $kind in
    cl_in)  CUDA_VISIBLE_DEVICES=$g $MG $SI $ck --scan $b --level $c --start $d --group V2M --out $W6/closed_loop/indoor/$grp > $W6/logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cov_out) CUDA_VISIBLE_DEVICES=$g $MG $SC4/out_cov_v4.py --grp $grp --scene $b --start $c > $W6/logs/cov_out_${grp}_${b}_s${c}.log 2>&1 ;;
    cl_out) CUDA_VISIBLE_DEVICES=$g $MG $SO $ck --group V2M --scene $b --start $c --root $W6/closed_loop/outdoor/$grp/V2M > $W6/logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it $(hostname -s) $(date '+%F %T')" >> $W6/logs/queue_$(hostname -s)_$g.log
done < $jf
