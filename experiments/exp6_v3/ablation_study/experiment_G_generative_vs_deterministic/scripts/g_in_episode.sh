#!/bin/bash
# Experiment G2 item 5 -- ONE queue task = one MP3D-150 episode (protocol G2-v1 section 5). Arms:
#   G1IN = V3-T0 + D1, K = 1 (noise map 0)    D0IN = D0 + D1, K = 1    G4R = V3-T0 + D1, K = 4 through the wrapper (regression smoke only)
# Idempotent (finished episode.json -> skip), atomic mkdir claim, <= 3 tries (failed outputs moved to _failed/, never deleted).
# Usage: g_in_episode.sh ARM SCAN LEVEL START     (GPU = CUDA_VISIBLE_DEVICES of the queue worker; test overrides G_MOVES / G_WORKDIR)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
A=$1; scan=$2; lv=$3; st=$4; MOVES=${G_MOVES:-150}
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
WI=${G_WORKDIR:-$R/work_dirs/exp6_v3/experiment_G/g2/indoor}
case $A in
  G1IN) MODE=flow; K=1; CK=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt ;;
  D0IN) MODE=d0;   K=1; CK=$R/work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt ;;
  G4R)  MODE=flow; K=4; CK=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt ;;
  *) echo "unknown arm $A"; exit 9 ;;
esac
tag=${scan}_L${lv}_S${st}; od=$WI/raw/$A/V2M/$tag; CL=$WI/claims/${A}_$tag; L=$WI/logs; mkdir -p $WI/claims $WI/raw/_failed/$A $L
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f $od/episode.json ] && { echo "SKIP done $A $tag"; exit 0; }
mkdir $CL 2>/dev/null || { echo "CLAIMED by $(cat $CL/owner 2>/dev/null)"; exit 4; }
echo "${WORKER_ID:-?} $(hostname -s) gpu $CUDA_VISIBLE_DEVICES $(date '+%F %T')" > $CL/owner
trap 'rm -rf $CL' EXIT
for try in 1 2 3; do
  if [ -d $od ] && [ ! -f $od/episode.json ]; then mv $od $WI/raw/_failed/$A/${tag}_try${try}_$(hostname -s)_$(date +%s); fi
  t0=$(date +%s); echo "start try $try $(date '+%F %T') $(hostname -s) gpu $CUDA_VISIBLE_DEVICES: $A ($MODE, K $K) $tag moves $MOVES"
  $MG $GD/scripts/sim_in_g.py --v3-ckpt $CK --g-mode $MODE --g-k $K --scan $scan --level $lv --start $st --group V2M --moves $MOVES --out $WI/raw/$A >> $L/${A}_$tag.log 2>&1
  rc=$?; echo "exit $rc wall $(( $(date +%s) - t0 ))s"
  [ -f $od/episode.json ] && { tail -1 $L/${A}_$tag.log; exit 0; }
done
echo "FAILED_3 $A $tag"; exit 1
