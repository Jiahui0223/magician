#!/bin/bash
# G1 multi-surface belief of one evaluated checkpoint (evaluation side only): mesh depth-peeling GT of the dumped rays
# (exp6_v5/scripts/eval_belief_gt.py, magician env) -> interval-belief metrics (eval_belief_metrics_v52m.py).
# Usage: belief_eval_v52m.sh EVAL_DIR CKPT GPU   (EVAL_DIR = work_dirs/exp6_v5_2_scratch70k/eval/<name>)
OD=$1; CK=$2; G=${3:-3}
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
cd $R; echo "$(date '+%F %T') belief GT $OD on $(hostname -s) GPU $G"
D=""
for d in $OD/fixed/valsel/B0 $OD/fixed/dev3/B0 $OD/fixed/dev3/B2; do
  [ -d $d ] || continue; D="$D $d"
  env -u LD_PRELOAD -u PYTHONPATH CUDA_VISIBLE_DEVICES=$G /bsuscratch/jiahuizhang/envs/magician/bin/python experiments/exp6_v5/scripts/eval_belief_gt.py --dir $d
done
LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R CUDA_VISIBLE_DEVICES="" $S2E/bin/python experiments/exp6_v5_2_scratch70k/scripts/eval_belief_metrics_v52m.py --ckpt $CK --dirs $D --out $OD/belief_metrics.json
echo "$(date '+%F %T') belief done exit $?"; touch $OD/BELIEF_DONE
