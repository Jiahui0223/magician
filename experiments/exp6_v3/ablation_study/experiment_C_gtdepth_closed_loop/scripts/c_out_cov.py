"""EXP6 V3 Ablation Experiment C -- Macarons++ coverage of a C1 / C2 closed loop (magician env): task2/scripts/t25_out_cov.py
unchanged (official coverage replay must equal the stored curve, else STOP; monotone coverage = the primary metric),
roots under the Experiment C work dir. Usage: python c_out_cov.py --grp C1|C2 --scene S --start K [--root DIR --out DIR]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
import t25_out_cov as T25  # noqa: E402

if __name__ == "__main__":
    grp = sys.argv[sys.argv.index("--grp") + 1]
    s = sys.argv[sys.argv.index("--scene") + 1]
    k = sys.argv[sys.argv.index("--start") + 1]
    root = sys.argv[sys.argv.index("--root") + 1] if "--root" in sys.argv else CC.outdoor_root(grp)
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(CC.WEVAL, "metrics", "outdoor")
    T25.C.W2 = CC.WC                                   # replay_frames temp dir -> WC/eval/outdoor/replay_frames
    sys.argv = [sys.argv[0], "--group", grp, "--tag", CC.tag_out(grp, s, k), "--root", root, "--out", out]
    T25.main()
