#!/bin/bash
# Experiment G -- post-training steps of D0 (protocol G-v1 sections 2.6, 7, 8). PREPARED, run by hand after the D0 training log shows "EXIT 0":
#   1. frozen V3 selection rule -> work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt (+ final.json)
#   2. full-run check vs V3-T0 (step-1 rec / kl bitwise, manifests, finite loss) -> checks/d0_main_vs_t0.json
#   3. enqueue the fixed-state workers (expG_c_fixed_w1..8: D0 / G0 / G1 / G4 on the 755-state pool)
#   4. enqueue the closed loops (expG_d_{D0,G0}_*: All-14 x 5 x 100 moves + official replay; 140 tasks, dunnottar first)
# After the queue is empty: g_analyze_fixed.py (fixed-state summary) and g_collect.py (Table 3 csv / md / tex with paired CIs vs G4 and G1).
# Usage: post_train.sh [select|check|fixed|closed|all]       (default all; CPU only, the GPU work goes through the shared queue)
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; cd $R
S2E=/bsuscratch/jiahuizhang/envs/seen2scene; PY="env CUDA_VISIBLE_DEVICES= PYTHONPATH=$R LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib $S2E/bin/python"
GD=$R/experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic; W=$R/work_dirs/exp6_v3/experiment_G
what=${1:-all}
[ -f $W/d0/runs/main/run_none.json ] || { echo "D0 training not finished (no $W/d0/runs/main/run_none.json)"; exit 1; }
if [ $what = select ] || [ $what = all ]; then $PY $GD/scripts/g_select_final.py || [ -f $W/d0/checkpoints/final.pt ] || exit 2; fi
if [ $what = check ] || [ $what = all ]; then $PY $GD/scripts/check_train_vs_t0.py --run $W/d0/runs/main --mode det --out $W/checks/d0_main_vs_t0.json; fi
if [ $what = fixed ] || [ $what = all ]; then $PY $GD/scripts/make_queue_tasks.py fixed --workers 8 --write; fi
if [ $what = closed ] || [ $what = all ]; then $PY $GD/scripts/make_queue_tasks.py closed --arms D0,G0 --write; fi
echo "then: $PY $GD/scripts/g_analyze_fixed.py ; $PY $GD/scripts/g_collect.py"
