#!/bin/bash
# B2: G3 closed loops (MP3D VALSEL 10 x 150, Macarons++ Dev-3 + Test-3: 6 scenes, 30 x 100) with the frozen B2 planner (V2-Mix EMA@17.5k,
# D100 input, S10 DistP1, K 4, n_win 5, R10). Tracked form of work_dirs/.../task3_full_depth_p1/chain_b2.sh.
# Usage: g3.sh [OUT_ROOT] [WORKERS_PER_GPU]   (OUT_ROOT default work_dirs/exp6_v2/g3: historical results are not touched)
source $(dirname $0)/../../shared/env.sh
OUT=${1:-$R/work_dirs/exp6_v2/g3}; N=${2:-2}; C=$X2/B2/configs
for g in 0 1 2 3; do for i in $(seq $N); do ( bash $X2/shared/g3_queue.sh $g $C/g3_jobs_in.txt $OUT; bash $X2/shared/g3_queue.sh $g $C/g3_jobs_out.txt $OUT ) & sleep 2; done; done; wait
echo "B2 closed loops done; outdoor coverage: task3 t35_out_cov.py --grp B2 (reads the Task 3 root; see README)"
