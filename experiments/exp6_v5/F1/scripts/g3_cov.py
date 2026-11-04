"""EXP6 V5 / F1 -- coverage replay of one G3 episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py unchanged (official
MAGICIAN coverage must equal the stored episode curve bitwise, else STOP; monotone 10 % coverage), roots under
work_dirs/exp6_v5/F1/g3 (temporary replay frames in g3/cov_tmp/<MODEL>, removed by t25). Same as the Table 1 wrapper.
Usage: python g3_cov.py --model MODEL --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

G3 = os.path.join(REPO, "work_dirs", "exp6_v5", "F1", "g3")

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    model, s, k = g("--model"), g("--scene"), g("--start")
    T25.C.W2 = os.path.join(G3, "cov_tmp", model)
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0",
                "--root", os.path.join(G3, "closed_loop", "outdoor", model, "V2M"), "--out", os.path.join(G3, "metrics", "outdoor", model)]
    T25.main()
