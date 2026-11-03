#!/bin/bash
# Camera Pose Phase 1 -- closed-loop worker (C0 / C1). Usage: run_queue.sh GPU JOBFILE
# Job lines: cl_in:<arm>:<cond>:<scan>:<level>:<start> | cl_out:<arm>:<cond>:<scene>:<start> | cov_out:<arm>:<cond>:<scene>:<start>
#   arm  B2 (V2-Mix EMA@17.5k) | E2D (E2-D100 EMA@5k) | V4 (EMA@40k) | V5 (EMA@12k, CP_V5=1 -> v5_core.install as
#        v5_*_planner.py): registry checkpoints via T3_CKPT, P1 mode b2 (D100 input, S10 DistP1) = their frozen planners;
#        MAGB with kind mag_in = MAGICIAN official beam 10x10 (Task 2 t23_in_sim --group MAG-B, original grid)
#   cond C0 (lambda 0 = original DistP1, identical to the historical planner) | C1 (lambda from configs/c1_lambda.json) | ORIG (MAGB)
# Output root per (arm, cond): work_dirs/exp_camera_pose/closed_loop/<arm>_<cond>/ (own replay-frame dirs -> no races);
# done files skip finished episodes, noclobber claims share the queue between workers / nodes.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp_camera_pose/scripts; CK=$R/work_dirs/checkpoints/exp6_v2; W=$R/work_dirs/exp_camera_pose
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); mkdir -p $W/claims $W/logs; cd $W
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r it; do
  [ -z "$it" ] && continue
  IFS=: read kind arm cond b c d <<< "$it"
  v5=0
  case $arm in
    B2) ck=$CK/B2/v2mix_ema17500.pt ;; E2D) ck=$CK/E2/e2d100_ema5000.pt ;;
    V4) ck=$R/work_dirs/checkpoints/exp6_v4/V4/v4_ema40000.pt ;; V5) ck=$R/work_dirs/checkpoints/exp6_v5/V5/v5_ema12000.pt; v5=1 ;;
    MAGB) ck=none ;; *) echo "bad arm $arm"; exit 2 ;;
  esac
  case $cond in
    C0|ORIG) lam=0 ;;
    C1) lam=$(/bsuscratch/jiahuizhang/envs/seen2scene/bin/python -c "import json; print(json.load(open('$R/experiments/exp_camera_pose/configs/c1_lambda.json'))['$arm'])") || exit 3 ;;
    *) echo "bad cond $cond"; exit 2 ;;
  esac
  grp=${arm}_${cond}; root=$W/closed_loop/$grp
  case $kind in
    cl_in)  done_f=$root/closed_loop/indoor/$grp/V2M/${b}_L${c}_S${d}/episode.json ;;
    mag_in) done_f=$root/closed_loop/indoor/MAG-B/${b}_L${c}_S${d}/episode.json ;;
    cl_out) done_f=$root/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ;;
    cov_out) done_f=$root/metrics/outdoor/$grp/V2M/V2M_${b}_s${c}_r0.json; [ -f $root/closed_loop/outdoor/$grp/V2M/episodes/ep_V2M_${b}_s${c}_r0.json ] || continue ;;
  esac
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$(hostname -s):$g:$$" > claims/$(echo $it | tr ':' '_').claim ) 2>/dev/null || continue
  case $kind in
    mag_in) CUDA_VISIBLE_DEVICES=$g $MG $R/experiments/exp5_future_view/task2/scripts/t23_in_sim.py --scan $b --level $c --start $d --group MAG-B --out $root/closed_loop/indoor > logs/mag_in_${b}_L${c}_S${d}.log 2>&1 ;;
    cl_in)  CP_V5=$v5 CP_LAMBDA=$lam T3_CKPT=$ck CUDA_VISIBLE_DEVICES=$g $MG $SC/c1_in_sim.py --p1 b2 --scan $b --level $c --start $d --group V2M --out $root/closed_loop/indoor/$grp > logs/cl_in_${grp}_${b}_L${c}_S${d}.log 2>&1 ;;
    cl_out) CP_V5=$v5 CP_LAMBDA=$lam T3_CKPT=$ck CUDA_VISIBLE_DEVICES=$g $MG $SC/c1_out_sim.py --p1 b2 --group V2M --scene $b --start $c --root $root/closed_loop/outdoor/$grp/V2M > logs/cl_out_${grp}_${b}_s${c}.log 2>&1 ;;
    cov_out) V4_EVAL_ROOT=$root CUDA_VISIBLE_DEVICES=$g $MG $R/experiments/exp6_v4/scripts/out_cov_v4.py --grp $grp --scene $b --start $c > logs/cov_out_${grp}_${b}_s${c}.log 2>&1 ;;
  esac
  echo "exit $? $it lam=$lam $(hostname -s) $(date '+%F %T')" >> logs/queue_$(hostname -s)_$g.log
done < $jf
