#!/bin/bash
# G0 smokes of V4 on one node (4 x L40S): tiny overfit (GPU 0, 200 steps) || single-GPU exact-RNG resume test (GPU 1:
# A = 40 steps straight, B = 20 + resume to 40; the trainer restores numpy / torch / cuda RNG on 1 GPU) -> 4-GPU
# short run (1,000 steps of the real recipe: throughput, VRAM, loss). Tracked copy; run as: bash g0_v4.sh
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp6_v4
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
PY=$S2E/bin/python; TR="$S2E/bin/torchrun --standalone --nproc_per_node 4"
SC=experiments/exp6_v4/scripts/train_v4.py
mkdir -p $W/runs $W/logs
echo "G0 V4 start $(hostname -s) job ${SLURM_JOB_ID:-ssh} $(date '+%F %T')" >> $W/logs/g0.log
rm -rf $W/runs/smoke_overfit $W/runs/resume_A $W/runs/resume_B $W/runs/smoke_short
CUDA_VISIBLE_DEVICES=0 $PY $SC --out $W/runs/smoke_overfit --smoke overfit > $W/logs/smoke_overfit.log 2>&1 &
p1=$!
( OV="ckpt_every=20 eval_every=40 sample_eval_every=1000000"
  CUDA_VISIBLE_DEVICES=1 $PY $SC --out $W/runs/resume_A --override steps=40 $OV > $W/logs/resume_A.log 2>&1; echo "resume A exit $? $(date '+%T')" >> $W/logs/g0.log
  CUDA_VISIBLE_DEVICES=1 $PY $SC --out $W/runs/resume_B --override steps=20 $OV > $W/logs/resume_B1.log 2>&1; echo "resume B1 exit $? $(date '+%T')" >> $W/logs/g0.log
  CUDA_VISIBLE_DEVICES=1 $PY $SC --out $W/runs/resume_B --override steps=40 $OV > $W/logs/resume_B2.log 2>&1; echo "resume B2 exit $? $(date '+%T')" >> $W/logs/g0.log ) &
p2=$!
wait $p1; echo "overfit exit $? $(date '+%T')" >> $W/logs/g0.log
wait $p2
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/smoke_short --smoke short > $W/logs/smoke_short.log 2>&1
echo "short exit $? $(date '+%T')" >> $W/logs/g0.log
echo "G0 V4 DONE $(date '+%F %T')" >> $W/logs/g0.log
