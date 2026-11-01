"""EXP6 V4 / V3.1 -- Macarons++ coverage of a closed loop under work_dirs/exp6_v4/eval (magician env):
task2/scripts/t25_out_cov.py unchanged. Usage: python out_cov_v4.py --grp V4|V31 --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

W = os.path.join(T25.C.REPO, "work_dirs", "exp6_v4", "eval")

if __name__ == "__main__":
    grp = sys.argv[sys.argv.index("--grp") + 1]
    s = sys.argv[sys.argv.index("--scene") + 1]
    k = sys.argv[sys.argv.index("--start") + 1]
    T25.C.W2 = W
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0",
                "--root", os.path.join(W, "closed_loop", "outdoor", grp, "V2M"), "--out", os.path.join(W, "metrics", "outdoor", grp)]
    T25.main()
