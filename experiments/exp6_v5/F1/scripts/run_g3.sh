#!/bin/bash
# EXP6 V5 / F1 -- G3 closed-loop worker (resumable). Job lines <MODEL>:<scene>:<start>, MODEL in A2k A5k B2k B5k (A1k B1k
# optional). Per job: closed loop (exp6_v5 sim_out_v5.py, V4_P1_MODE=sparse = S10 inference + S10 DistP1, V4_CKPT = the
# F1 EMA checkpoint; frozen M-protocol simulator, settings, starts, seeds) -> coverage replay (g3_cov.py) -> record
# (g3_import.py). A job waits for its checkpoint file. Claims are noclobber files; a failed job moves its partial outputs to
# g3/failed/<model>_<tag>_tryN (<= 3 tries, then logs/gave_up.log). Usage: run_g3.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S=$R/experiments/exp6_v5/F1/scripts; W=$R/work_dirs/exp6_v5/F1; G3=$W/g3
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V3_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
mkdir -p $G3/claims $G3/logs $G3/failed; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ck() { case $1 in A1k) echo $W/runs/A/ema_001000.pt;; A2k) echo $W/runs/A/ema_002000.pt;; A5k) echo $W/runs/A/ema_005000.pt;;
                  B1k) echo $W/runs/B/ema_001000.pt;; B2k) echo $W/runs/B/ema_002000.pt;; B5k) echo $W/runs/B/ema_005000.pt;; esac; }
release() {
  local m=$1 s=$2 k=$3 tag=V2M_${2}_s${3}_r0 root=$G3/closed_loop/outdoor/$1/V2M n dst
  n=$(ls -d $G3/failed/${m}_${tag}_try* 2>/dev/null | wc -l)
  if [ $n -ge 3 ]; then echo "GAVE_UP $m:$s:$k $4 $(date '+%F %T')" >> $G3/logs/gave_up.log; return; fi
  dst=$G3/failed/${m}_${tag}_try$((n+1)); mkdir -p $dst
  for sub in frames svc ckpt planner_logs episodes; do for f in $root/$sub/*${tag}*; do [ -e "$f" ] && mkdir -p $dst/$sub && mv "$f" $dst/$sub/; done; done
  [ -f $G3/metrics/outdoor/$m/V2M/$tag.json ] && mv $G3/metrics/outdoor/$m/V2M/$tag.json $dst/
  mv $G3/claims/${m}_${s}_${k}.claim $dst/ 2>/dev/null
  echo "released $m:$s:$k ($4) -> $dst $(date '+%F %T')" >> $G3/logs/requeue.log
}
while true; do
  progress=0; waiting=0
  while IFS=: read -r m s k; do
    [ -z "$m" ] && continue
    [ -f $G3/episodes/$m/${s}_s${k}.json ] && continue
    c=$(ck $m); [ -f "$c" ] || { waiting=1; continue; }
    ( set -o noclobber; echo "$host:$g:$$" > $G3/claims/${m}_${s}_${k}.claim ) 2>/dev/null || continue
    root=$G3/closed_loop/outdoor/$m/V2M; t0=$(date +%s)
    CUDA_VISIBLE_DEVICES=$g V4_P1_MODE=sparse $MG $R/experiments/exp6_v5/scripts/sim_out_v5.py --v4-ckpt $c --group V2M --scene $s --start $k --root $root > $G3/logs/sim_${m}_${s}_s${k}.log 2>&1 < /dev/null; rc=$?
    [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g $MG $S/g3_cov.py --model $m --scene $s --start $k > $G3/logs/cov_${m}_${s}_s${k}.log 2>&1 < /dev/null; rc=$?; }
    [ $rc -eq 0 ] && { $PY $S/g3_import.py $m $s $k >> $G3/logs/import.log 2>&1; rc=$?; }
    echo "exit $rc $m:$s:$k $host gpu$g $(( $(date +%s) - t0 ))s $(date '+%F %T')" >> $G3/logs/queue_${host}_$g.log
    [ $rc -ne 0 ] && release $m $s $k "rc=$rc"
    progress=1
  done < $jf
  [ $progress = 0 ] && [ $waiting = 0 ] && break
  [ $progress = 0 ] && sleep 60
done
echo "worker end gpu$g $(date '+%F %T')" >> $G3/logs/queue_${host}_$g.log
