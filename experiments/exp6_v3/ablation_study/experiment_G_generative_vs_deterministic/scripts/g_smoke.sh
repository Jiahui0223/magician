#!/bin/bash
# Experiment G -- smoke / regression of one short closed loop through the shared queue (protocol G-v1 section 6).
# Usage: g_smoke.sh ARM SCENE START MOVES NAME [REF_MODEL] [CKPT]      (ARM D0 | G0 | G1R; G1R must reproduce REF_MODEL = V3-T0-K1D2)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
A=$1; S=$2; K=$3; MV=$4; NAME=$5; REF=$6; CK=$7
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic
SW=$R/work_dirs/exp6_v3/experiment_G/smoke/$NAME
[ -d $SW ] && mv $SW ${SW}_old_$(date +%s)
mkdir -p $SW
export G_MOVES=$MV G_WORKDIR=$SW
[ -n "$CK" ] && export G_CKPT=$CK
bash $GD/scripts/g_episode.sh $A $S $K || exit $?
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python $GD/scripts/g_smoke_check.py --root $SW/closed_loop/$A/${S}_s$K --scene $S --start $K --moves $MV ${REF:+--ref $REF} --out $SW/result.json
/bsuscratch/jiahuizhang/envs/seen2scene/bin/python -c "import json,sys; sys.exit(0 if json.load(open('$SW/result.json'))['pass'] else 7)"
