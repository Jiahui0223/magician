"""Experiment XA -- coverage replay of one outdoor closed-loop episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py unchanged (the official
MAGICIAN coverage replay must equal the stored episode curve bitwise, else STOP; the monotone 10 % coverage is stored as a secondary curve).
Usage: python xa_out_cov.py --root RUN_ROOT --scene S --start K      (RUN_ROOT/outdoor/V2M = closed-loop root, RUN_ROOT/metrics = output)"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    root, s, k = os.path.abspath(g("--root")), g("--scene"), g("--start")
    T25.C.W2 = os.path.join(root, "cov_tmp")
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0", "--root", os.path.join(root, "outdoor", "V2M"), "--out", os.path.join(root, "metrics")]
    T25.main()
