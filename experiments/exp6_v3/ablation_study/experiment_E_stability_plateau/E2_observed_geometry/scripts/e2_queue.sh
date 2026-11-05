#!/bin/bash
# E2 closed-loop worker. Job line: ID KSV(0/1) SCENE START REP ROOT. noclobber claims, one try. Usage: e2_queue.sh GPU JOBFILE
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
E=$R/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/E2_observed_geometry
W=$R/work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau/E2_observed_geometry
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT -u H_POLICY -u D_POLICY /bsuscratch/jiahuizhang/envs/magician/bin/python"
V3=$R/work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p $W/claims $E/logs/closed_loop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r id ksv s k rep root; do
  [ -z "$id" ] && continue; [[ $id == \#* ]] && continue
  [ -f $root/episodes/ep_V2M_${s}_s${k}_r${rep}.json ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/claims/$id.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g $MG $E/scripts/e2_sim.py --v3-ckpt $V3 --ksv $ksv --scene $s --start $k --rep $rep --root $root >> $E/logs/closed_loop/$id.log 2>&1
  echo "exit $? $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $E/logs/closed_loop/queue.log
done < $jf
