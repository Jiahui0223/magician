#!/bin/bash
# EXP7 MP3D Round-1 smoke: 1 level x 1 start x 3 methods (Ours V3-T0+D1, MAGICIAN beam, Random), N moves, one GPU each.
# Usage: bash smoke_mp3d.sh SCAN LEVEL START MOVES OUTROOT
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
cd $R
SCAN=$1; LV=$2; ST=$3; MV=$4; O=$5
mkdir -p $O/logs
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=$R/work_dirs/exp6_v3/task0_vae/checkpoints/final.pt
( /usr/bin/time -v env CUDA_VISIBLE_DEVICES=0 $MG $R/experiments/cvpr_results/exp7/scripts/sim_in_d.py --v3-ckpt $CK --scan $SCAN --level $LV --start $ST --group V2M --moves $MV --out $O/ours > $O/logs/ours.log 2>&1; echo "ours rc=$?" >> $O/logs/rc.txt ) &
( /usr/bin/time -v env CUDA_VISIBLE_DEVICES=1 $MG $R/experiments/exp5_future_view/task2/scripts/t23_in_sim.py --scan $SCAN --level $LV --start $ST --group MAG-B --moves $MV --out $O/mag > $O/logs/mag.log 2>&1; echo "mag rc=$?" >> $O/logs/rc.txt ) &
( /usr/bin/time -v env CUDA_VISIBLE_DEVICES=2 $MG $R/experiments/exp5_future_view/task0/magician_comparison/scripts/t12_sim.py --scan $SCAN --level $LV --start $ST --method RAND-NV --moves $MV --out $O/rand > $O/logs/rand.log 2>&1; echo "rand rc=$?" >> $O/logs/rc.txt ) &
( for i in $(seq 1 400); do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ' >> $O/logs/gpumem.txt; echo >> $O/logs/gpumem.txt; sleep 3; [ $(wc -l < $O/logs/rc.txt 2>/dev/null || echo 0) -ge 3 ] && break; done ) &
wait
