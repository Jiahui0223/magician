#!/bin/bash
# EXP6 V3 / ablation E -- closed-loop worker (diagnostic runs only; never Table 1). Job line (whitespace separated):
#   ID FAMILY POLICY CKPT SCENE START REP ROOT [SRC_EPISODE T]
# SRC_EPISODE/T given -> e_branch.py prepares ROOT from the source prefix first (skipped if ROOT/branch.json exists).
# Claims are noclobber files (one try per job; a failed job keeps its outputs and log for inspection). Usage: e_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
E=$R/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/scripts
W=$R/work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT -u H_POLICY -u D_POLICY /bsuscratch/jiahuizhang/envs/magician/bin/python"
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p $W/jobs/claims $W/logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r id fam pol ck s k rep root src T; do
  [ -z "$id" ] && continue; [[ $id == \#* ]] && continue
  [ -f $root/episodes/ep_V2M_${s}_s${k}_r${rep}.json ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/jobs/claims/$id.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  if [ -n "$src" ] && [ ! -f $root/branch.json ]; then $PY $E/e_branch.py --src-episode $src --T $T --root $root >> $W/logs/$id.log 2>&1; fi
  CUDA_VISIBLE_DEVICES=$g $MG $E/e_sim.py --family $fam --policy $pol --v3-ckpt $ck --scene $s --start $k --rep $rep --root $root >> $W/logs/$id.log 2>&1
  echo "exit $? $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $W/logs/queue.log
done < $jf
