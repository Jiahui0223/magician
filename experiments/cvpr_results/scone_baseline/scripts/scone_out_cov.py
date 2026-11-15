"""EXP9-SCONE Phase D -- official-coverage replay of one outdoor SCONE episode (magician env, 1 GPU): task2/scripts/t25_out_cov.py
run UNCHANGED (as table1/scripts/t1_out_cov.py does for the Table 1 rows): the official MAGICIAN coverage recomputed from the
executed keys must equal the stored episode curve bitwise (else STOP), monotone 10 % coverage alongside. Replay frames go to
--work (deleted by t25 after the replay).
Usage: python scone_out_cov.py --tag scone_<scene>_s<k>_r0 --root CLOSED_LOOP_ROOT --out METRICS_DIR --work TMP_DIR"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    g = lambda k: sys.argv[sys.argv.index(k) + 1]  # noqa: E731
    tag, root, out, work = g("--tag"), g("--root"), g("--out"), g("--work")
    T25.C.W2 = work
    sys.argv = [sys.argv[0], "--group", "scone", "--tag", tag, "--root", root, "--out", out]
    T25.main()
