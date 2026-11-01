#!/bin/bash
# G0 smokes of V3.1 on one node (4 x L40S): tiny overfit (GPU 0, 200 steps) -> 4-GPU profile (100 steps) -> 4-GPU
# resume test (A = 60 steps straight, B = 30 + resume; model / EMA / optimizer restored, re-seeded streams = the V3
# rule) -> 4-GPU short run (1,000 steps of the real recipe). Tracked copy; run as: bash g0_v31.sh
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp6_v3_1
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
PY=$S2E/bin/python; TR="$S2E/bin/torchrun --standalone --nproc_per_node 4"
SC=experiments/exp6_v3.1/scripts/train.py
mkdir -p $W/runs $W/logs
echo "G0 V3.1 start $(hostname -s) job ${SLURM_JOB_ID:-ssh} $(date '+%F %T')" >> $W/logs/g0.log
rm -rf $W/runs/smoke_overfit $W/runs/smoke_profile $W/runs/resume_A $W/runs/resume_B $W/runs/smoke_short
CUDA_VISIBLE_DEVICES=0 $PY $SC --out $W/runs/smoke_overfit --smoke overfit > $W/logs/smoke_overfit.log 2>&1
echo "overfit exit $? $(date '+%T')" >> $W/logs/g0.log
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/smoke_profile --smoke profile > $W/logs/smoke_profile.log 2>&1
echo "profile exit $? $(date '+%T')" >> $W/logs/g0.log
OV="ckpt_every=30 eval_every=60 eval_views=64"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/resume_A --override steps=60 $OV > $W/logs/resume_A.log 2>&1; echo "resume A exit $? $(date '+%T')" >> $W/logs/g0.log
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/resume_B --override steps=30 $OV > $W/logs/resume_B1.log 2>&1; echo "resume B1 exit $? $(date '+%T')" >> $W/logs/g0.log
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/resume_B --override steps=60 $OV > $W/logs/resume_B2.log 2>&1; echo "resume B2 exit $? $(date '+%T')" >> $W/logs/g0.log
CUDA_VISIBLE_DEVICES=0,1,2,3 $TR $SC --out $W/runs/smoke_short --smoke short > $W/logs/smoke_short.log 2>&1
echo "short exit $? $(date '+%T')" >> $W/logs/g0.log
echo "G0 V3.1 DONE $(date '+%F %T')" >> $W/logs/g0.log
