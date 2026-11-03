"""CVPR Table 1 -- coverage replay of one closed-loop episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py run
unchanged (official MAGICIAN coverage must equal the stored episode curve bitwise, else STOP; monotone 10 % coverage), with
the temporary replay frames under work_dirs/cvpr_results/table1/g3/cov_tmp/<MODEL> instead of the Task 2 work dir.
Usage: python t1_out_cov.py --model MODEL --group GROUP --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

T1W = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3")

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    model, group, s, k = g("--model"), g("--group"), g("--scene"), g("--start")
    T25.C.W2 = os.path.join(T1W, "cov_tmp", model)
    sys.argv = [sys.argv[0], "--group", group, "--tag", f"{group}_{s}_s{k}_r0",
                "--root", os.path.join(T1W, "closed_loop", "outdoor", model, group), "--out", os.path.join(T1W, "metrics", "outdoor", model)]
    T25.main()
