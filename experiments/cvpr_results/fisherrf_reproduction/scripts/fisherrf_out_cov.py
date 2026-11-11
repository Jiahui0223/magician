"""FisherRF reproduction -- coverage replay of one Macarons++ closed-loop episode (magician env, 1 GPU), the CVPR Table 1
procedure (table1/scripts/t1_out_cov.py): task2/scripts/t25_out_cov.py run UNCHANGED (the official MAGICIAN coverage of the
replay must equal the stored episode curve bitwise, else STOP; monotone 10 % coverage), with the temporary replay frames
and the metrics under the FisherRF work dir (nothing is written to the table1 / task2 folders).
Usage: python fisherrf_out_cov.py --root CLOSED_LOOP_ROOT --out METRICS_DIR --scene S --start K [--group FisherRF]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    g = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d  # noqa: E731
    root, out, s, k, group = os.path.abspath(g("--root")), os.path.abspath(g("--out")), g("--scene"), g("--start"), g("--group", "FisherRF")
    T25.C.W2 = os.path.join(os.path.dirname(out), "cov_tmp")      # replay frames: <cov_tmp>/eval/outdoor/replay_frames/
    sys.argv = [sys.argv[0], "--group", group, "--tag", f"{group}_{s}_s{k}_r0", "--root", root, "--out", out]
    T25.main()
