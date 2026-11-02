#!/bin/bash
# run_labels_rev.sh G1 G2 : extra label workers (one per scene) that process the scene's REMAINING split trajectories in
# reverse order, alongside the forward workers of run_labels.sh (outputs are written tmp + os.replace and skipped when
# present, so a trajectory done twice only costs time). Remaining list: work_dirs/exp6_v5/labels/remaining.json.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; SC=$R/experiments/exp6_v5/scripts; L=$R/work_dirs/exp6_v5/labels/v1
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
cd $R; i=0; gs=($1 $2)
for s in bannerman dunnottar fushimi liberty pisa redeemer; do
  tr=$($MG -c "import json; print(','.join(t for sc, t in reversed(json.load(open('$R/work_dirs/exp6_v5/labels/remaining.json'))) if sc == '$s'))")
  [ -z "$tr" ] && continue
  CUDA_VISIBLE_DEVICES=${gs[$((i % 2))]} $MG $SC/build_multisurface_labels.py run --scene $s --trajs $tr --K 16 --k-store 8 --n-sel 1024 --out $L > $L/logs/${s}_rev.log 2>&1 &
  i=$((i + 1))
done
wait; echo "REV LABELS DONE $(date '+%F %T')"
