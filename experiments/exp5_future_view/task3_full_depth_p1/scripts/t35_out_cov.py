"""EXP5 Task 3 / E1 -- Macarons++ coverage of the B0 / B1 closed loops (magician env, 1 GPU; evaluation side):
task2/scripts/t25_out_cov.py run unchanged (official coverage must equal the stored curve, else STOP; monotone coverage =
the Task 2 primary evaluator), with its temporary replay frames redirected to W3 instead of the Task 2 work dir.
Usage: python t35_out_cov.py --grp B0|B1 --scene S --start K"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

W3 = os.path.join(T25.C.REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")

if __name__ == "__main__":
    i = sys.argv.index("--grp")
    grp = sys.argv[i + 1]
    s = sys.argv[sys.argv.index("--scene") + 1]
    k = sys.argv[sys.argv.index("--start") + 1]
    T25.C.W2 = W3                                  # replay_frames temp dir -> W3/eval/outdoor/replay_frames
    sys.argv = [sys.argv[0], "--group", "V2M", "--tag", f"V2M_{s}_s{k}_r0",
                "--root", os.path.join(W3, "closed_loop", "outdoor", grp, "V2M"), "--out", os.path.join(W3, "metrics", "outdoor", grp)]
    T25.main()
