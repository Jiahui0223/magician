#!/bin/bash
# EXP6 V3 / Y -- formal continuation run (or a test run with extra trainer overrides) on the 4 GPUs of this node.
# Seeds the run dir with a copy of the parent 40K ckpt_last.pt (sha256 checked) if it has no ckpt_last.pt yet; then
# torchrun train_y.py (resumes from ckpt_last.pt). Resumable: re-run the same command. Usage: train_y.sh RUN_DIR [override ...]
set -u
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
OUT=$1; shift; mkdir -p $OUT; cd $R
PARENT=$R/work_dirs/exp6_v3/task1_xattn/runs/main/ckpt_last.pt; PSHA=3d81978206c88068c5dd5eda9cb221c8023948854549184b8aa35274c5941ce9
if [ ! -f $OUT/ckpt_last.pt ]; then
  s=$(sha256sum $PARENT | cut -d' ' -f1); [ "$s" = "$PSHA" ] || { echo "PARENT SHA MISMATCH $s" >> $OUT/train.log; exit 3; }
  cp $PARENT $OUT/ckpt_last.pt.tmp && mv $OUT/ckpt_last.pt.tmp $OUT/ckpt_last.pt
  echo "seeded ckpt_last.pt from parent (sha $PSHA) $(date '+%F %T')" >> $OUT/train.log
fi
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
PORT=${Y_PORT:-29611}
echo "start $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-alloc} git $(git rev-parse --short HEAD) $(date '+%F %T')" >> $OUT/train.log
OV=""; [ $# -gt 0 ] && OV="--override $*"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT --nnodes 1 --nproc_per_node 4 experiments/exp6_v3/Y/scripts/train_y.py --out $OUT $OV >> $OUT/train.log 2>&1
rc=$?; echo "EXIT $rc $(hostname -s) $(date '+%F %T')" >> $OUT/train.log; exit $rc
