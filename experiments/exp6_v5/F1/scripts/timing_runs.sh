#!/bin/bash
# EXP6 V5 / F1 -- controlled cost measurement: barts_s0 re-run for B0 / A5k / B5k (and A2k / B2k), ONE episode per idle GPU,
# same chain as G3 (sim_out_v5.py, V4_P1_MODE=sparse); outputs under g3/timing/<model> (never the G3 records). The re-runs
# also check reproducibility (executed keys / official coverage vs the G3 records; B0 vs Table 1). Usage: timing_runs.sh
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5/F1; cd $R
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V3_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
declare -A CK=([B0]=$R/work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt [A5k]=$W/runs/A/ema_005000.pt [B5k]=$W/runs/B/ema_005000.pt [A2k]=$W/runs/A/ema_002000.pt)
g=0
for m in B0 A5k B5k A2k; do
  mkdir -p $W/g3/timing/$m/V2M
  ( CUDA_VISIBLE_DEVICES=$g V4_P1_MODE=sparse $MG experiments/exp6_v5/scripts/sim_out_v5.py --v4-ckpt ${CK[$m]} --group V2M --scene barts --start 0 --root $W/g3/timing/$m/V2M > $W/logs/timing_$m.log 2>&1; echo "exit $?" >> $W/logs/timing_$m.log ) &
  g=$((g+1))
done
wait
echo TIMING_DONE $(date '+%F %T') >> $W/logs/timing_done.log
