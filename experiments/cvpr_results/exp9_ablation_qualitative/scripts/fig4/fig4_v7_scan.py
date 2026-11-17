"""EXP9 Fig. 4 v7 -- extend the outdoor state pool to the 4 Unseen-6 scenes the v4-v6 pool lacked (eiffel, bridge, barts, sestino_museum).
Runs experiments/cvpr_results/scripts/fig4/fig4_scan.py UNCHANGED (frozen V3-T0 + D1 planner-state replay of the stored Table 1 episodes,
grid-hash / value guards against the planner log, t in {0, 10, ..., 50}, 5 official starts) with its output directory redirected to
work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v7/states/ (the overnight intermediates are not touched).
Usage: bash work_dirs/exp6_v3/run_py.sh GPU <this file> --scene S"""
import sys

sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/cvpr_results/scripts/fig4")
import fig4_scan as M  # noqa: E402

M.OUT = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/exp9_ablation_qualitative/fig4_v7/states"
M.main()
