"""Experiment U -- official coverage replay of one closed-loop episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py run
UNCHANGED (official MAGICIAN coverage replayed from the executed poses must equal the stored episode curve bitwise, else STOP),
as cvpr_results/table1/scripts/t1_out_cov.py does, with the temporary replay frames under work_dirs/exp6_v3/experiment_U/cov_tmp.
Usage: python u_out_cov.py --root EPISODE_ROOT --out METRICS_DIR --scene S --start K [--group V2M]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    g = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    root, out, s, k, group = g("--root"), g("--out"), g("--scene"), g("--start"), g("--group", "V2M")
    T25.C.W2 = os.path.join(REPO, "work_dirs", "exp6_v3", "experiment_U", "cov_tmp", os.path.basename(os.path.normpath(root)))
    sys.argv = [sys.argv[0], "--group", group, "--tag", f"{group}_{s}_s{k}_r0", "--root", root, "--out", out]
    T25.main()
