#!/bin/bash
# CVPR Table 1 -- closed-loop worker for missing episodes (same simulators / planners / settings as the historical runs;
# launch chains from MODEL_REGISTRY.yaml). Job lines: <MODEL>:<scene>:<start>. Per job: closed loop -> coverage replay
# (t1_out_cov.py; official curve must equal the stored one) -> canonical record (import_episodes.py --new). Claims are
# noclobber files; a failed job releases its partial outputs to g3/failed/<model>_<tag>_tryN (<= 3 tries, nothing deleted).
# At most ONE V2-family planner per GPU is not needed outdoors (planners 2-7 GB). Usage: t1_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
EXP5=$R/experiments/exp5_future_view; X=$R/experiments; SC=$R/experiments/cvpr_results/table1/scripts
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
T1W=$R/work_dirs/cvpr_results/table1/g3
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
mkdir -p $T1W/claims $T1W/logs $T1W/failed; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
E2T=$R/work_dirs/exp5_future_view/task3_full_depth_p1/e2/training
group_of() { case $1 in MAG-B|MAG-G|V2O) echo $1 ;; *) echo V2M ;; esac; }
run_sim() {   # model scene start root
  local m=$1 s=$2 k=$3 root=$4
  case $m in
    MAG-B|MAG-G|V2O) $MG $EXP5/task2/scripts/t24_out_sim.py --group $m --scene $s --start $k --root $root ;;
    B0)  $MG $EXP5/task2/scripts/t24_out_sim.py --group V2M --scene $s --start $k --root $root ;;
    B2)  $MG $EXP5/task3_full_depth_p1/scripts/t33_out_sim.py --p1 b2 --group V2M --scene $s --start $k --root $root ;;
    B1)  $MG $EXP5/task3_full_depth_p1/scripts/t33_out_sim.py --p1 full --group V2M --scene $s --start $k --root $root ;;
    E2D) env -u LD_PRELOAD -u PYTHONPATH -u V4_P1_MODE -u V3_CKPT -u V4_CKPT T3_CKPT=$E2T/d100/ema_005000.pt /bsuscratch/jiahuizhang/envs/magician/bin/python $EXP5/task3_full_depth_p1/scripts/t33_out_sim.py --p1 b2 --group V2M --scene $s --start $k --root $root ;;
    E2S) env -u LD_PRELOAD -u PYTHONPATH -u V4_P1_MODE -u V3_CKPT -u V4_CKPT T3_CKPT=$E2T/s10/ema_005000.pt /bsuscratch/jiahuizhang/envs/magician/bin/python $EXP5/task3_full_depth_p1/scripts/t33_out_sim.py --p1 sparse --group V2M --scene $s --start $k --root $root ;;
    V3)  $MG $X/exp6_v3/shared/sim_out_v3.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --group V2M --scene $s --start $k --root $root ;;
    V3.1) $MG $X/exp6_v3/shared/sim_out_v3.py --v3-ckpt $R/work_dirs/exp6_v3_1/checkpoints/final.pt --group V2M --scene $s --start $k --root $root ;;
    V4)  $MG $X/exp6_v4/scripts/sim_out_v4.py --v4-ckpt $R/work_dirs/exp6_v4/checkpoints/final.pt --group V2M --scene $s --start $k --root $root ;;
    V4C) $MG $X/exp6_v4/scripts/sim_out_v4.py --v4-ckpt $R/work_dirs/exp6_v5/checkpoints/v4c_final.pt --group V2M --scene $s --start $k --root $root ;;
    V5)  $MG $X/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $R/work_dirs/exp6_v5/checkpoints/v5_final.pt --group V2M --scene $s --start $k --root $root ;;
    F1-CONT-v1@2k) env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V3_CKPT -u V4_CKPT V4_P1_MODE=sparse /bsuscratch/jiahuizhang/envs/magician/bin/python $X/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $R/work_dirs/exp6_v5/F1/runs/A/ema_002000.pt --group V2M --scene $s --start $k --root $root ;;
    F1-GQ-v1@5k) env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V3_CKPT -u V4_CKPT V4_P1_MODE=sparse /bsuscratch/jiahuizhang/envs/magician/bin/python $X/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $R/work_dirs/exp6_v5/F1/runs/B/ema_005000.pt --group V2M --scene $s --start $k --root $root ;;
    V3-H1) $MG $X/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy h1 --group V2M --scene $s --start $k --root $root ;;
    V3-H2) $MG $X/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy h2 --group V2M --scene $s --start $k --root $root ;;
    V3-H3) $MG $X/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy h3 --group V2M --scene $s --start $k --root $root ;;
    V3-D1-H3-NORISK) $MG $X/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy d1 --group V2M --scene $s --start $k --root $root ;;
    V3-D2-DEPTH3-B3) $MG $X/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy d2 --group V2M --scene $s --start $k --root $root ;;
    V3-D3-DEPTH3-B5) $MG $X/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy d3 --group V2M --scene $s --start $k --root $root ;;
    V3-D4-DEPTH4-B3) $MG $X/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --v3-ckpt $R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt --policy d4 --group V2M --scene $s --start $k --root $root ;;
    V3-Y45K-H0) $MG $X/exp6_v3/shared/sim_out_v3.py --v3-ckpt $R/work_dirs/exp6_v3/Y/runs/main/ema_045000.pt --group V2M --scene $s --start $k --root $root ;;
    V3-Y50K-H0) $MG $X/exp6_v3/shared/sim_out_v3.py --v3-ckpt $R/work_dirs/exp6_v3/Y/runs/main/ema_050000.pt --group V2M --scene $s --start $k --root $root ;;
    V3-YBEST-H3) $MG $X/exp6_v3/H/scripts/sim_out_h.py --v3-ckpt $R/work_dirs/exp6_v3/Y/runs/main/ema_050000.pt --policy h3 --group V2M --scene $s --start $k --root $root ;;
    V3-YD-COMBINED-Y50K-D1) $MG $X/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/sim_out_d.py --v3-ckpt $R/work_dirs/exp6_v3/Y/runs/main/ema_050000.pt --policy d1 --group V2M --scene $s --start $k --root $root ;;
    *) echo "unknown model $m"; return 2 ;;
  esac
}
release() {   # model scene start reason
  local m=$1 s=$2 k=$3 G=$(group_of $1) n dst
  local tag=${G}_${s}_s${k}_r0 root=$T1W/closed_loop/outdoor/$m/$G
  n=$(ls -d $T1W/failed/${m}_${tag}_try* 2>/dev/null | wc -l)
  if [ $n -ge 3 ]; then echo "GAVE_UP $m:$s:$k $4 $(date '+%F %T')" >> $T1W/logs/gave_up.log; return; fi
  dst=$T1W/failed/${m}_${tag}_try$((n+1)); mkdir -p $dst
  for sub in frames svc ckpt planner_logs episodes; do
    for f in $root/$sub/*${tag}*; do [ -e "$f" ] && mkdir -p $dst/$sub && mv "$f" $dst/$sub/; done
  done
  [ -f $T1W/metrics/outdoor/$m/$G/$tag.json ] && mv $T1W/metrics/outdoor/$m/$G/$tag.json $dst/
  mv $T1W/claims/${m}_${s}_${k}.claim $dst/ 2>/dev/null
  echo "released $m:$s:$k ($4) -> $dst $(date '+%F %T')" >> $T1W/logs/requeue.log
}
while true; do
  progress=0
  while IFS=: read -r m s k; do
    [ -z "$m" ] && continue
    [ -f $R/work_dirs/cvpr_results/table1/episodes/$m/${s}_s${k}.json ] && continue
    ( set -o noclobber; echo "$host:$g:$$" > $T1W/claims/${m}_${s}_${k}.claim ) 2>/dev/null || continue
    G=$(group_of $m); root=$T1W/closed_loop/outdoor/$m/$G; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g run_sim $m $s $k $root > $T1W/logs/sim_${m}_${s}_s${k}.log 2>&1 < /dev/null; rc=$?
    [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g $MG $SC/t1_out_cov.py --model $m --group $G --scene $s --start $k > $T1W/logs/cov_${m}_${s}_s${k}.log 2>&1 < /dev/null; rc=$?; }
    [ $rc -eq 0 ] && { $PY $SC/import_episodes.py --new $m $s $k >> $T1W/logs/import.log 2>&1; rc=$?; }
    echo "exit $rc $m:$s:$k $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $T1W/logs/queue_${host}_$g.log
    [ $rc -ne 0 ] && release $m $s $k "rc=$rc"
    progress=1
  done < $jf
  [ $progress = 0 ] && break
done
echo "worker end gpu$g $(date '+%F %T')" >> $T1W/logs/queue_${host}_$g.log
