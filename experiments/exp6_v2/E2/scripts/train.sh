#!/bin/bash
# E2: data build + fine-tuning exactly as the E2 pilot (protocol e2_pilot/protocol.md; configs e2_{d100,s10}_v1.json:
# init V2-Mix EMA@17.5k, 5k steps, lr 2e-5, 64 views = 32 MP3D + 32 Macarons++, bf16, EMA 0.999). Tracked form of
# work_dirs/.../e2/{build_out.sh, build_mp3d_node.sh, train_arm.sh}. Prints the commands; RUN=1 executes them with NEW
# output folders (the historical training dirs are never written). Not run during the 2026-11-02 organisation.
# Usage: [RUN=1] train.sh [d100|s10]
source $(dirname $0)/../../shared/env.sh
ARM=${1:-d100}; O=${OUT:-$R/work_dirs/exp6_v2/E2/retrain}
B1="for s in bannerman dunnottar fushimi liberty pisa redeemer; do $PY $E2P/e2_build.py out \$s; done   # Macarons++ D100 shards (only if missing)"
B2="$PY $E2P/e2_build.py mp3d 0 1   # MP3D D100 shards (claim-based; only if missing)"
T="cd $E2P && CUDA_VISIBLE_DEVICES=0,1,2,3 $S2E/bin/torchrun --standalone --nproc_per_node 4 e2_train.py $ARM --out $O/$ARM"
echo "$B1"; echo "$B2"; echo "$T"
[ "${RUN:-0}" = 1 ] && eval "$T"
