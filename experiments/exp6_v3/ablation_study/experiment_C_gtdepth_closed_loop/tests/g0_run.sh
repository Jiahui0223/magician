#!/bin/bash
# Experiment C -- G0 smoke runs (magician env; 4 GPUs of this node). Outputs under work_dirs/.../experiment_C/g0/.
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
SC=$R/experiments/exp6_v3/ablation_study/experiment_C_gtdepth_closed_loop/scripts
G0=$R/work_dirs/exp6_v3/ablation_study/experiment_C/g0
MG="env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
cd $R; rm -rf $G0; mkdir -p $G0/logs
say() { echo "$(date '+%F %T') $*"; }
run() { local g=$1 name=$2; shift 2; ( "$@" > $G0/logs/$name.log 2>&1; echo "exit $? $name $(date '+%T')" >> $G0/logs/status.txt ) & }
say "G0 start"
# GPU 0: indoor C1 (a) then indoor C1 (b: determinism)
( CUDA_VISIBLE_DEVICES=0 $MG $SC/c_in_sim.py --group C1 --scan Z6MFQCViBuw --level 0 --start 6 --moves 5 --out $G0/indoor_a > $G0/logs/in_C1_a.log 2>&1; echo "exit $? in_C1_a $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=0 $MG $SC/c_in_sim.py --group C1 --scan Z6MFQCViBuw --level 0 --start 6 --moves 5 --out $G0/indoor_b > $G0/logs/in_C1_b.log 2>&1; echo "exit $? in_C1_b $(date '+%T')" >> $G0/logs/status.txt ) &
# GPU 1: indoor C2 then indoor B2 replay
( CUDA_VISIBLE_DEVICES=1 $MG $SC/c_in_sim.py --group C2 --scan Z6MFQCViBuw --level 0 --start 6 --moves 5 --out $G0/indoor_a > $G0/logs/in_C2.log 2>&1; echo "exit $? in_C2 $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=1 $MG $SC/c_in_sim.py --group B2replay --scan Z6MFQCViBuw --level 0 --start 6 --moves 5 --out $G0/indoor_a > $G0/logs/in_B2replay.log 2>&1; echo "exit $? in_B2replay $(date '+%T')" >> $G0/logs/status.txt ) &
# GPU 2: outdoor C2 (a) then outdoor C2 (b: determinism) then its coverage replay
( CUDA_VISIBLE_DEVICES=2 $MG $SC/c_out_sim.py --group C2 --scene barts --start 0 --moves 3 --root $G0/outdoor_a/C2 > $G0/logs/out_C2_a.log 2>&1; echo "exit $? out_C2_a $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=2 $MG $SC/c_out_sim.py --group C2 --scene barts --start 0 --moves 3 --root $G0/outdoor_b/C2 > $G0/logs/out_C2_b.log 2>&1; echo "exit $? out_C2_b $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=2 $MG $SC/c_out_cov.py --grp C2 --scene barts --start 0 --root $G0/outdoor_a/C2 --out $G0/outdoor_a/metrics > $G0/logs/out_C2_cov.log 2>&1; echo "exit $? out_C2_cov $(date '+%T')" >> $G0/logs/status.txt ) &
# GPU 3: outdoor C1 then outdoor B2 replay (group V2M = Task 3 B2 planner) then C1 coverage replay
( CUDA_VISIBLE_DEVICES=3 $MG $SC/c_out_sim.py --group C1 --scene barts --start 0 --moves 3 --root $G0/outdoor_a/C1 > $G0/logs/out_C1.log 2>&1; echo "exit $? out_C1 $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=3 $MG $SC/c_out_sim.py --group V2M --scene barts --start 0 --moves 3 --root $G0/outdoor_rep/V2M > $G0/logs/out_B2replay.log 2>&1; echo "exit $? out_B2replay $(date '+%T')" >> $G0/logs/status.txt
  CUDA_VISIBLE_DEVICES=3 $MG $SC/c_out_cov.py --grp C1 --scene barts --start 0 --root $G0/outdoor_a/C1 --out $G0/outdoor_a/metrics > $G0/logs/out_C1_cov.log 2>&1; echo "exit $? out_C1_cov $(date '+%T')" >> $G0/logs/status.txt ) &
wait
say "G0 runs finished"; cat $G0/logs/status.txt
