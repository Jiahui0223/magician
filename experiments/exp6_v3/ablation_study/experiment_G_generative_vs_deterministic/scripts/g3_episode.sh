#!/bin/bash
# Experiment G3 -- ONE queue task = one Macarons++ episode of arm ARM (protocol G3-v1) + the official coverage replay (t25_out_cov unchanged;
# must equal the stored curve bitwise). Idempotent: a finished episode (replay metrics with official_equal_stored) is skipped; claim-safe
# (atomic mkdir claim, removed on exit); a partial episode without replay metrics is moved to _failed/<arm>_<scene>_s<k>_try<n> and
# restarted (<= 3 tries); a finished closed loop without replay only replays.  (Same mechanism as experiment_U u_episode.sh.)
#   ARM  MRAW = V3-T0, 4 samples averaged in (b, y) then decoded, D1 via K = 1   MDEPTH = 4 samples decoded, >= 2 of 4 hit, mean t, D1 via K = 1
#        G4R = hooks off, K = 4 (regression only: must reproduce the archived V3-T0-D1 episode)
# Usage: g3_episode.sh ARM SCENE START      (GPU = CUDA_VISIBLE_DEVICES set by the queue worker; test overrides G_MOVES / G_WORKDIR)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
A=$1; S=$2; K=$3; MOVES=${G_MOVES:-100}
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
WG=${G_WORKDIR:-$R/work_dirs/exp6_v3/experiment_G/g3}
T0CK=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt
case $A in
  MRAW)   MODE=raw;   CK=$T0CK ;;
  MDEPTH) MODE=depth; CK=$T0CK ;;
  G4R)    MODE=off;   CK=$T0CK ;;
  *) echo "unknown arm $A"; exit 9 ;;
esac
[ -f $CK ] || { echo "missing checkpoint $CK"; exit 8; }
O=$WG/closed_loop/$A/${S}_s$K; TAG=V2M_${S}_s${K}_r0; MET=$O/metrics/V2M/$TAG.json; EP=$O/episodes/ep_$TAG.json
CL=$WG/claims/${A}_${S}_s$K; L=$WG/logs/episodes; mkdir -p $WG/claims $WG/_failed $L
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
done_ok() { [ -f $MET ] && $PY -c "import json,sys; m=json.load(open('$MET')); sys.exit(0 if m.get('official_equal_stored') and m.get('moves')==$MOVES and len(m['cov_official'])==$MOVES+1 else 1)"; }
if done_ok; then echo "SKIP done $A $S s$K"; exit 0; fi
mkdir $CL 2>/dev/null || { echo "CLAIMED by $(cat $CL/owner 2>/dev/null)"; exit 4; }
echo "${WORKER_ID:-?} $(hostname -s) gpu $CUDA_VISIBLE_DEVICES $(date '+%F %T')" > $CL/owner
trap 'rm -rf $CL' EXIT
echo "start $(date '+%F %T') $(hostname -s) gpu $CUDA_VISIBLE_DEVICES worker ${WORKER_ID:-?}: $A ($MODE, $CK) $S s$K moves $MOVES"
if [ ! -f $EP ] || ! $PY -c "import json,sys; e=json.load(open('$EP')); sys.exit(0 if e.get('moves')==$MOVES and len(e['coverage'])==$MOVES+1 else 1)" 2>/dev/null; then
  if [ -d $O ]; then
    n=$(ls -d $WG/_failed/${A}_${S}_s${K}_try* 2>/dev/null | wc -l)
    [ $n -ge 3 ] && { echo "GAVE_UP $A $S s$K after $n tries" | tee -a $WG/logs/gave_up.log; exit 5; }
    mv $O $WG/_failed/${A}_${S}_s${K}_try$((n+1))
  fi
  mkdir -p $O; t0=$(date +%s)
  $MG $GD/scripts/sim_out_g3.py --v3-ckpt $CK --g3-mode $MODE --group V2M --scene $S --start $K --moves $MOVES --root $O > $L/sim_${A}_${S}_s$K.log 2>&1
  rc=$?; echo "closed loop rc=$rc $(( $(date +%s) - t0 ))s"; [ $rc = 0 ] || { tail -20 $L/sim_${A}_${S}_s$K.log; exit 1; }
fi
t1=$(date +%s)
$MG $GD/scripts/g_out_cov.py --root $O --out $O/metrics --scene $S --start $K > $L/cov_${A}_${S}_s$K.log 2>&1
rc=$?; echo "replay rc=$rc $(( $(date +%s) - t1 ))s"; [ $rc = 0 ] || { tail -20 $L/cov_${A}_${S}_s$K.log; exit 2; }
done_ok || { echo "replay metrics not valid"; exit 3; }
tail -1 $L/cov_${A}_${S}_s$K.log; echo "end $(date '+%F %T')"
