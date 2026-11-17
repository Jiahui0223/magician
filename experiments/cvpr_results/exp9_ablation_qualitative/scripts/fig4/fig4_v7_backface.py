"""EXP9 Fig. 4 v7 -- back-face (inside-the-mesh) check of the new pool states: experiments/cvpr_results/scripts/fig4/fig4_backface.py run
UNCHANGED except its output directory (the OUT line is replaced before execution, asserted present exactly once).
Usage: CUDA_VISIBLE_DEVICES=g $MAGPY -I -B <this file> --scene S  ->  work_dirs/.../fig4_v7/states/backface_<scene>.jsonl"""
import sys

SRC = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/cvpr_results/scripts/fig4/fig4_backface.py"
OLD = 'OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "overnight_20261110", "intermediates", "fig4")'
NEW = 'OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "exp9_ablation_qualitative", "fig4_v7", "states")'
src = open(SRC).read()
assert src.count(OLD) == 1
sys.argv = [SRC] + sys.argv[1:]
exec(compile(src.replace(OLD, NEW), SRC, "exec"), {"__name__": "__main__", "__file__": SRC})
