#!/bin/bash
# Experiment XA -- closed-loop worker. Job lines:  out RUN CKPT SCENE START [MOVES]  |  in RUN CKPT SCAN LEVEL START [MOVES]
# RUN = name under work_dirs/.../experiment_XA_multiscale_ray_scene/eval/ (one per model). Outdoor: xa_sim_out.py (D1, frozen simulator; the simulator resumes
# a partial episode bitwise) + xa_out_cov.py (official replay check). Indoor: xa_sim_in.py (H0). noclobber claims; a failed job is retried up to 3 times
# (outdoor: resume; indoor: partial folder moved to failed/ and restarted with the same seeds); nothing is deleted. Usage: xa_eval_queue.sh GPU JOBFILE
#
# v2 (2026-11-06, scheduling only -- the commands that run an episode are unchanged). Round 1 with 4 workers per GPU hit the per-user thread limit (4096;
# one episode = about 200 threads) and filled a GPU with four large-scene planners. Now:
#   - at most 3 workers per GPU (the others exit at once);
#   - a job starts only when the GPU's reservations allow it: per-scene planner + simulator peak (measured in round 1) within a 42 GB budget;
#     dunnottar and indoor jobs stay alone / nearly alone on a GPU;
#   - an episode running longer than 90 min is terminated (a hung simulator blocked both chains for 2.7 h) and counted as a failed try;
#   - after a failed outdoor try an orphaned planner of that episode is removed.
R=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; cd $R
E=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene; W=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT -u H_POLICY -u D_POLICY -u D_CONFIG -u E3_DV2 -u E3_DV2_CONFIG /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MAXW=3; BUDGET=42; TMO=5400
RES=$W/locks/${host}_g$g; mkdir -p $RES
live() {   # under the lock: remove entries of dead workers, print "<workers> <reserved GB>"
  local n=0 tot=0 f p
  for f in $RES/*.wrk; do [ -e "$f" ] || continue; p=$(basename $f .wrk); if kill -0 $p 2>/dev/null; then n=$((n+1)); else rm -f $f $RES/$p.res; fi; done
  for f in $RES/*.res; do [ -e "$f" ] || continue; p=$(basename $f .res); if kill -0 $p 2>/dev/null; then tot=$((tot + $(cat $f))); else rm -f $f; fi; done
  echo "$n $tot"
}
ok=$( ( flock 8; set -- $(live); if [ $1 -lt $MAXW ]; then echo $(date +%FT%T) > $RES/$$.wrk; echo yes; else echo no; fi ) 8>$RES/.lock )
[ "$ok" = yes ] || exit 0
trap 'rm -f $RES/$$.wrk $RES/$$.res' EXIT
need_gb() { case $1 in dunnottar) echo 40;; bridge) echo 18;; colosseum|sestino_museum) echo 14;; pisa) echo 11;; alhambra) echo 10;; *) echo 9;; esac; }
admit() {  # $1 = GB; waits until the reservations on this GPU allow the job
  local a
  while true; do
    a=$( ( flock 8; set -- $(live) $1; if [ $(( $2 + $3 )) -le $BUDGET ]; then echo $3 > $RES/$$.res; echo yes; else echo no; fi ) 8>$RES/.lock )
    [ "$a" = yes ] && return 0
    sleep 15
  done
}
while read -r kind run ck a b c d; do
  [ -z "$kind" ] && continue; [[ $kind == \#* ]] && continue
  root=$W/eval/$run; mkdir -p $root/claims $root/logs $root/failed
  if [ $kind = out ]; then id=out_${a}_s${b}; done_f=$root/metrics/V2M/V2M_${a}_s${b}_r0.json; mv=${c:-100}; need=$(need_gb $a)
  else id=in_${a}_L${b}_S${c}; done_f=$root/indoor/V2M/${a}_L${b}_S${c}/episode.json; mv=${d:-150}; need=24; fi
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $root/claims/$id.claim ) 2>/dev/null || continue
  admit $need
  t0=$(date +%s); rc=1
  for try in 1 2 3; do
    if [ $kind = out ]; then
      rc=0
      [ -f $root/outdoor/V2M/episodes/ep_V2M_${a}_s${b}_r0.json ] || { CUDA_VISIBLE_DEVICES=$g timeout -k 60 $TMO $MG $E/scripts/xa_sim_out.py --ckpt $R/$ck --policy d1 --group V2M --scene $a --start $b --root $root/outdoor/V2M --moves $mv >> $root/logs/$id.log 2>&1; rc=$?; }
      [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g timeout -k 60 $TMO $MG $E/scripts/xa_out_cov.py --root $root --scene $a --start $b >> $root/logs/$id.log 2>&1; rc=$?; }
      [ $rc -ne 0 ] && pkill -u $(id -u) -f "xa_out_planner.py --exp V2M --scene $a --start $b --rep 0 --root $root/outdoor/V2M " 2>/dev/null
    else
      CUDA_VISIBLE_DEVICES=$g timeout -k 60 $TMO $MG $E/scripts/xa_sim_in.py --ckpt $R/$ck --scan $a --level $b --start $c --group V2M --out $root/indoor --moves $mv >> $root/logs/$id.log 2>&1; rc=$?
      [ $rc -ne 0 ] && [ -d $root/indoor/V2M/${a}_L${b}_S${c} ] && mv $root/indoor/V2M/${a}_L${b}_S${c} $root/failed/${id}_try${try}_$(date +%s)
    fi
    echo "try $try exit $rc $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $root/logs/queue.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
  rm -f $RES/$$.res
  [ $rc -ne 0 ] && mv $root/claims/$id.claim $root/failed/$id.claim.$(date +%s)
done < $jf
