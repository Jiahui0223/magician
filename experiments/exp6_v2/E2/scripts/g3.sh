#!/bin/bash
# E2: G3 closed loops of E2-D100 (B2-style planner; job lists configs/g3_jobs_*.txt) and, with CONTROL=1, of the E2-S10 control
# arm (sparse planner; configs/control_e2s10_g3_jobs_*.txt). Tracked form of work_dirs/.../e2/chain_g3.sh + run_queue_g3.sh.
# Usage: [CONTROL=1] g3.sh [OUT_ROOT] [WORKERS_PER_GPU]   (OUT_ROOT default work_dirs/exp6_v2/g3)
source $(dirname $0)/../../shared/env.sh
OUT=${1:-$R/work_dirs/exp6_v2/g3}; N=${2:-2}; C=$X2/E2/configs
L="g3_jobs_in.txt g3_jobs_out.txt"; [ "${CONTROL:-0}" = 1 ] && L="$L control_e2s10_g3_jobs_in.txt control_e2s10_g3_jobs_out.txt"
for g in 0 1 2 3; do for i in $(seq $N); do ( for f in $L; do bash $X2/shared/g3_queue.sh $g $C/$f $OUT; done ) & sleep 2; done; done; wait
echo "E2 closed loops done; outdoor coverage: task3 t35_out_cov.py --grp E2D|E2S; analysis e2_pilot/scripts/e2_g3_analyze.py"
