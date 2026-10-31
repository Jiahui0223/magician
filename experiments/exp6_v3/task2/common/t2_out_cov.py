"""Task 2 -- Macarons++ coverage of a Task 2 closed loop (magician env): t25_out_cov.py unchanged.
Usage: python t2_out_cov.py --root DIR --out DIR --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(E6))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    root = sys.argv[sys.argv.index("--root") + 1]
    out = sys.argv[sys.argv.index("--out") + 1]
    s = sys.argv[sys.argv.index("--scene") + 1]
    k = sys.argv[sys.argv.index("--start") + 1]
    T25.C.W2 = os.path.join(REPO, "work_dirs", "exp6_v3", "task2")
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0", "--root", root, "--out", out]
    T25.main()
