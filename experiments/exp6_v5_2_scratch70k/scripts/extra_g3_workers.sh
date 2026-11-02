#!/bin/bash
# Extra G3 queue workers (efficiency only; same job lists / claims / outputs as the driver's workers, so finished or
# claimed items are skipped and results are identical in kind). Usage: extra_g3_workers.sh smoke|full N_PER_GPU
# Reads the checkpoint / score config of the running G3 stage from RUN_STATUS.json. On node B run it through
# srun --jobid=<nodeB job> --overlap -N1 -n1.
KIND=$1; N=${2:-1}
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; W=$R/work_dirs/exp6_v5_2_scratch70k; SC=$R/experiments/exp6_v5_2_scratch70k/scripts
cd $R
eval $(/bsuscratch/jiahuizhang/envs/seen2scene/bin/python -c "
import json; g = json.load(open('$R/experiments/exp6_v5_2_scratch70k/RUN_STATUS.json'))['g3']['$KIND']
print('export V52M_CKPT=%s V52M_SCORE_CFG=%s' % (g['ckpt'], g['score_cfg']))")
export G3_ROOT=$W/eval/g3
echo "$(date '+%F %T') extra G3 $KIND workers on $(hostname -s): $N / GPU, ckpt $V52M_CKPT"
for g in 0 1 2 3; do for i in $(seq $N); do bash $SC/run_queue_v52m.sh $g $W/g3_${KIND}_jobs.txt & sleep 2; done; done; wait
for g in 0 1 2 3; do for i in $(seq $N); do bash $SC/run_queue_v52m.sh $g $W/g3_${KIND}_cov.txt & sleep 2; done; done; wait
echo "$(date '+%F %T') extra G3 $KIND workers done on $(hostname -s)"
