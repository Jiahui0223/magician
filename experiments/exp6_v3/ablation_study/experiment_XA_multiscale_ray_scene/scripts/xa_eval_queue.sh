#!/bin/bash
# Experiment XA -- closed-loop worker. Job lines:  out RUN CKPT SCENE START [MOVES]  |  in RUN CKPT SCAN LEVEL START [MOVES]
# RUN = name under work_dirs/.../experiment_XA_multiscale_ray_scene/eval/ (one per model). Outdoor: xa_sim_out.py (D1, frozen simulator; the simulator resumes
# a partial episode bitwise) + xa_out_cov.py (official replay check). Indoor: xa_sim_in.py (H0). noclobber claims; a failed job is retried up to 3 times
# (outdoor: resume; indoor: partial folder moved to failed/ and restarted with the same seeds); nothing is deleted. Usage: xa_eval_queue.sh GPU JOBFILE
R=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene; cd $R
E=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene; W=/bsuhome/jiahuizhang/scratch/projects/occupancy/seen2scene/work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT -u H_POLICY -u D_POLICY -u D_CONFIG -u E3_DV2 -u E3_DV2_CONFIG /bsuscratch/jiahuizhang/envs/magician/bin/python"
g=$1; jf=$(readlink -f $2); host=$(hostname -s)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while read -r kind run ck a b c d; do
  [ -z "$kind" ] && continue; [[ $kind == \#* ]] && continue
  root=$W/eval/$run; mkdir -p $root/claims $root/logs $root/failed
  if [ $kind = out ]; then id=out_${a}_s${b}; done_f=$root/metrics/V2M/V2M_${a}_s${b}_r0.json; mv=${c:-100}
  else id=in_${a}_L${b}_S${c}; done_f=$root/indoor/V2M/${a}_L${b}_S${c}/episode.json; mv=${d:-150}; fi
  [ -f $done_f ] && continue
  ( set -o noclobber; echo "$host g$g $(date +%FT%T)" > $root/claims/$id.claim ) 2>/dev/null || continue
  t0=$(date +%s); rc=1
  for try in 1 2 3; do
    if [ $kind = out ]; then
      rc=0
      [ -f $root/outdoor/V2M/episodes/ep_V2M_${a}_s${b}_r0.json ] || { CUDA_VISIBLE_DEVICES=$g $MG $E/scripts/xa_sim_out.py --ckpt $R/$ck --policy d1 --group V2M --scene $a --start $b --root $root/outdoor/V2M --moves $mv >> $root/logs/$id.log 2>&1; rc=$?; }
      [ $rc -eq 0 ] && { CUDA_VISIBLE_DEVICES=$g $MG $E/scripts/xa_out_cov.py --root $root --scene $a --start $b >> $root/logs/$id.log 2>&1; rc=$?; }
    else
      CUDA_VISIBLE_DEVICES=$g $MG $E/scripts/xa_sim_in.py --ckpt $R/$ck --scan $a --level $b --start $c --group V2M --out $root/indoor --moves $mv >> $root/logs/$id.log 2>&1; rc=$?
      [ $rc -ne 0 ] && [ -d $root/indoor/V2M/${a}_L${b}_S${c} ] && mv $root/indoor/V2M/${a}_L${b}_S${c} $root/failed/${id}_try${try}_$(date +%s)
    fi
    echo "try $try exit $rc $id $host g$g $(( $(date +%s) - t0 ))s $(date +%FT%T)" >> $root/logs/queue.log
    [ $rc -eq 0 ] && break
    sleep 20
  done
  [ $rc -ne 0 ] && mv $root/claims/$id.claim $root/failed/$id.claim.$(date +%s)
done < $jf
