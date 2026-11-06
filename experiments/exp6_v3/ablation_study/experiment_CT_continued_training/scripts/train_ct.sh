#!/bin/bash
# EXP6 V3 / CT -- CT-M continuation on the 4 GPUs of this node (trainer unchanged, hooks in train_ct.py). Seeds the run dir with a copy of the parent
# 40K ckpt_last.pt (sha256 checked); automatic restart from ckpt_last.pt at most 2 times. Usage: train_ct.sh RUN_DIR PORT [--config FILE] [--override k=v ...]
set -u
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; S2E=/bsuscratch/jiahuizhang/envs/seen2scene
OUT=$1; PORT=$2; shift 2; mkdir -p $OUT; cd $R
E=$R/experiments/exp6_v3/ablation_study/experiment_CT_continued_training
PARENT=$R/work_dirs/exp6_v3/task1_xattn/runs/main/ckpt_last.pt; PSHA=3d81978206c88068c5dd5eda9cb221c8023948854549184b8aa35274c5941ce9
if [ ! -f $OUT/ckpt_last.pt ]; then
  s=$(sha256sum $PARENT | cut -d' ' -f1); [ "$s" = "$PSHA" ] || { echo "PARENT SHA MISMATCH $s" >> $OUT/train.log; exit 3; }
  cp $PARENT $OUT/ckpt_last.pt.tmp && mv $OUT/ckpt_last.pt.tmp $OUT/ckpt_last.pt
  echo "seeded ckpt_last.pt from parent (sha $PSHA) $(date '+%F %T')" >> $OUT/train.log
fi
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1
for try in 0 1 2; do
  echo "start try $try $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} alloc ${SLURM_JOB_ID:-3461041} git $(git rev-parse --short HEAD) $(date '+%F %T')" >> $OUT/train.log
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$PORT --nnodes 1 --nproc_per_node 4 $E/scripts/train_ct.py --out $OUT "$@" >> $OUT/train.log 2>&1
  rc=$?; echo "EXIT $rc $(hostname -s) $(date '+%F %T')" >> $OUT/train.log
  [ $rc -eq 0 ] && break
  sleep 60
done
exit $rc
