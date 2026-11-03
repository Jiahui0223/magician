#!/bin/bash
# EXP Density A/B -- release failed G3 jobs for a clean rerun (FROZEN_PROTOCOL §9): for every job logged with a non-zero
# exit whose done file is missing and whose episode is not running, move its partial outputs and claim to g3/failed/
# (nothing is deleted) so that a queue worker reruns it from scratch. At most 2 releases per job; then it stays failed
# and is reported. Usage: requeue_failed.sh [loop]
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
G=$R/work_dirs/exp_density_ab/g3; cd $G; mkdir -p failed
one_pass() {
  grep -h "^exit [^0]" logs/queue_*.log logs/queue2_*.log 2>/dev/null | awk '{print $3}' | grep -v cov-only | sort -u | while read it; do
    IFS=: read kind cfg b c d <<< "$it"
    case $kind in
      cl_in)  done_f=closed_loop/indoor/$cfg/V2M/${b}_L${c}_S${d}/episode.json; tag=${b}_L${c}_S${d}
              pat="da_in_sim.py --p1 [a-z0-9]* --scan $b --level $c --start $d " ;;
      cl_out) done_f=metrics/outdoor/$cfg/V2M/V2M_${b}_s${c}_r0.json; tag=V2M_${b}_s${c}_r0
              pat="da_out_sim.py --p1 [a-z0-9]* --group V2M --scene $b --start $c " ;;
      *) continue ;;
    esac
    [ -f $done_f ] && continue
    claim=claims/$(echo $it | tr ':' '_').claim
    [ -f $claim ] || continue
    pgrep -f "$pat--.*/$cfg" > /dev/null && continue
    pgrep -f "$pat" | while read p; do tr '\0' ' ' < /proc/$p/cmdline; echo; done | grep -q "/$cfg" && continue
    n=$(ls -d failed/${cfg}_${tag}_try* 2>/dev/null | wc -l)
    if [ $n -ge 2 ]; then echo "GAVE_UP $it" >> logs/requeue.log; continue; fi
    dst=failed/${cfg}_${tag}_try$((n+1)); mkdir -p $dst
    if [ $kind = cl_in ]; then
      [ -d closed_loop/indoor/$cfg/V2M/$tag ] && mv closed_loop/indoor/$cfg/V2M/$tag $dst/
    else
      for sub in frames svc ckpt planner_logs episodes; do
        for f in closed_loop/outdoor/$cfg/V2M/$sub/*${tag}*; do [ -e "$f" ] && mkdir -p $dst/$sub && mv "$f" $dst/$sub/; done
      done
    fi
    mv $claim $dst/
    echo "released $it -> $dst $(date '+%F %T')" >> logs/requeue.log
  done
}
if [ "$1" = loop ]; then
  while pgrep -f "[d]a_g3_queue2.sh" > /dev/null; do one_pass; sleep 300; done
else
  one_pass
fi
