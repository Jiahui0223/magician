#!/bin/bash
# Experiment U Phase E -- ONE queue task = one 100-move Macarons++ episode of policy POLICY (V3-T0, protocol U-v1) + the official
# coverage replay (t25_out_cov unchanged; must equal the stored curve bitwise). Idempotent: a finished episode (replay metrics with
# official_equal_stored) is skipped; claim-safe (atomic mkdir claim, removed on exit); a partial episode without replay metrics is
# moved to _failed/<policy>_<scene>_s<k>_try<n> and restarted (<= 3 tries); a finished closed loop without replay only replays.
# Usage: u_episode.sh POLICY SCENE START      (GPU = CUDA_VISIBLE_DEVICES set by the queue worker)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
P=$1; S=$2; K=$3; MOVES=${U_MOVES:-100}            # U_MOVES / U_WORKDIR: test overrides only (Phase E uses the defaults)
U=$R/experiments/exp6_v3/ablation_study/experiment_U_generative_disagreement
WU=${U_WORKDIR:-$R/work_dirs/exp6_v3/experiment_U}
O=$WU/closed_loop/$P/${S}_s$K; TAG=V2M_${S}_s${K}_r0; MET=$O/metrics/V2M/$TAG.json; EP=$O/episodes/ep_$TAG.json
CL=$WU/claims/${P}_${S}_s$K; L=$WU/logs/episodes; mkdir -p $WU/claims $WU/_failed $L
MG="env -u LD_PRELOAD -u PYTHONPATH -u T3_CKPT -u V4_P1_MODE -u V3_CKPT -u V4_CKPT /bsuscratch/jiahuizhang/envs/magician/bin/python"
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
done_ok() { [ -f $MET ] && $PY -c "import json,sys; m=json.load(open('$MET')); sys.exit(0 if m.get('official_equal_stored') and m.get('moves')==$MOVES and len(m['cov_official'])==$MOVES+1 else 1)"; }
if done_ok; then echo "SKIP done $P $S s$K"; exit 0; fi
mkdir $CL 2>/dev/null || { echo "CLAIMED by $(cat $CL/owner 2>/dev/null)"; exit 4; }
echo "${WORKER_ID:-?} $(hostname -s) gpu $CUDA_VISIBLE_DEVICES $(date '+%F %T')" > $CL/owner
trap 'rm -rf $CL' EXIT
echo "start $(date '+%F %T') $(hostname -s) gpu $CUDA_VISIBLE_DEVICES worker ${WORKER_ID:-?}: $P $S s$K"
if [ ! -f $EP ] || ! $PY -c "import json,sys; e=json.load(open('$EP')); sys.exit(0 if e.get('moves')==$MOVES and len(e['coverage'])==$MOVES+1 else 1)" 2>/dev/null; then
  if [ -d $O ]; then
    n=$(ls -d $WU/_failed/${P}_${S}_s${K}_try* 2>/dev/null | wc -l)
    [ $n -ge 3 ] && { echo "GAVE_UP $P $S s$K after $n tries" | tee -a $WU/logs/gave_up.log; exit 5; }
    mv $O $WU/_failed/${P}_${S}_s${K}_try$((n+1))
  fi
  mkdir -p $O; t0=$(date +%s)
  $MG $U/scripts/sim_out_u.py --v3-ckpt $R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt --policy $P --group V2M --scene $S --start $K --moves $MOVES --root $O > $L/sim_${P}_${S}_s$K.log 2>&1
  rc=$?; echo "closed loop rc=$rc $(( $(date +%s) - t0 ))s"; [ $rc = 0 ] || { tail -20 $L/sim_${P}_${S}_s$K.log; exit 1; }
fi
t1=$(date +%s)
$MG $U/scripts/u_out_cov.py --root $O --out $O/metrics --scene $S --start $K > $L/cov_${P}_${S}_s$K.log 2>&1
rc=$?; echo "replay rc=$rc $(( $(date +%s) - t1 ))s"; [ $rc = 0 ] || { tail -20 $L/cov_${P}_${S}_s$K.log; exit 2; }
done_ok || { echo "replay metrics not valid"; exit 3; }
tail -1 $L/cov_${P}_${S}_s$K.log; echo "end $(date '+%F %T')"
