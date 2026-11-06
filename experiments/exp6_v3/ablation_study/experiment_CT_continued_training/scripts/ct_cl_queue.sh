#!/bin/bash
# Experiment CT -- closed-loop worker. Job lines: out MODEL CKPT SCENE START | in MODEL CKPT SCAN LEVEL START. Outdoor: e3_sim.py --dv2 full --rep 0 (frozen M-protocol simulator, official coverage) + coverage replay
# (t25_out_cov: official replay must equal the stored curve). Indoor: exp6_v3/shared/sim_in_v3.py (frozen Task 2 planner path). Outputs only under CT's work dir; noclobber claims. Usage: ct_cl_queue.sh GPU JOBFILE
R=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; cd $R
E3=$R/experiments/exp6_v3/ablation_study/experiment_E_stability_plateau/E3_dv2_closed_loop/scripts
SH=$R/experiments/exp6_v3/shared
E=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_CT_continued_training; W=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT -u H_POLICY -u D_POLICY -u E3_DV2 -u E3_DV2_CONFIG /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); host=$(hostname -s); mkdir -p $W/cl_claims $E/logs/closed_loop
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r kind model ck a b c; do
  [ -z "$kind" ] && continue; [[ $kind == \#* ]] && continue
  if [ $kind = out ]; then id=${model}_${a}_s${b}; root=$W/closed_loop/outdoor/$model/V2M; done_f=$W/closed_loop/metrics/outdoor/$model/V2M/V2M_${a}_s${b}_r0.json
  else id=${model}_${a}_L${b}_S${c}; root=$W/closed_loop/indoor/$model; done_f=$root/V2M/${a}_L${b}_S${c}/episode.json; fi
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $W/cl_claims/$id.claim ) 2>/dev/null || continue
  t0=$(date +%s)
  if [ $kind = out ]; then
    CUDA_VISIBLE_DEVICES=$g $MG $E3/e3_sim.py --v3-ckpt $R/$ck --dv2 full --scene $a --start $b --rep 0 --root $root >> $E/logs/closed_loop/$id.log 2>&1; rc=$?
    [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g $MG $E/scripts/ct_out_cov.py --model $model --scene $a --start $b >> $E/logs/closed_loop/$id.log 2>&1; rc=$?; }
  else
    CUDA_VISIBLE_DEVICES=$g $MG $SH/sim_in_v3.py --v3-ckpt $R/$ck --scan $a --level $b --start $c --group V2M --out $root >> $E/logs/closed_loop/$id.log 2>&1; rc=$?
  fi
  echo "exit $rc $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $E/logs/closed_loop_queue.log
done < $jf
