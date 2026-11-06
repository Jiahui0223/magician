"""Experiment CT -- coverage replay of one outdoor closed-loop episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py unchanged (official coverage must equal the stored
curve bitwise, else STOP; monotone 10 % coverage), temporary frames and output under CT's work dir. Usage: python ct_out_cov.py --model MODEL --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

W = os.path.join(REPO, "work_dirs", "exp6_v3", "ablation_study", "experiment_CT_continued_training", "closed_loop")

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    model, s, k = g("--model"), g("--scene"), g("--start")
    T25.C.W2 = os.path.join(W, "cov_tmp", model)
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0", "--root", os.path.join(W, "outdoor", model, "V2M"), "--out", os.path.join(W, "metrics", "outdoor", model)]
    T25.main()
