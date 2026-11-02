"""EXP6 V5.2 -- full G3 analysis (seen2scene env, CPU): exp6_v5/scripts/eval_g3.py main (exp6_v3 readers, paired
scene / floor bootstrap, Task 3 §4 criteria) with the groups V52L0 / V52L01 (work_dirs/exp6_v5_2/eval) added to V5 / V4C
(exp6_v5), V4 / V31 / A1 (exp6_v4) and the history; output written to experiments/exp6_v5_2/reports/g3_v52_full.{json,md}
and figures/g3_v52_full_coverage_curves.png (exp6_v5 files untouched). Usage: python eval_g3_full.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v52_common as VC  # noqa: E402

sys.path.insert(0, os.path.join(VC.REPO, "experiments", "exp6_v5", "scripts"))
import eval_g3 as G5  # noqa: E402

W52 = os.path.join(VC.W52, "eval")
G5.AG.ROOTS.update({g: (os.path.join(W52, "closed_loop"), os.path.join(W52, "metrics", "outdoor"), "V2M") for g in ("V52L0", "V52L01")})
G5.GRP = ("MAG-G", "MAG-B", "V2O", "B0", "B2", "E2D", "T1", "V31", "V4", "A1", "V4C", "V5", "V52L0", "V52L01")
G5.PAIRS = (("V52L0", "V5"), ("V52L01", "V5"), ("V52L01", "V52L0"), ("V52L0", "V4C"), ("V52L01", "V4C"), ("V52L0", "B2"), ("V52L01", "B2"),
            ("V52L0", "E2D"), ("V52L01", "E2D"), ("V52L0", "MAG-B"), ("V52L01", "MAG-B"), ("V52L0", "MAG-G"), ("V52L01", "MAG-G"), ("V52L0", "V4"))
G5.LAB.update({"V52L0": "V5.2 λ=0 (visible only)", "V52L01": "V5.2 λ=0.1 (visible + potential)"})
G5.SRC.update({"V52L0": "V5@12k + hazard 6k (no flow), direct visible score", "V52L01": "same checkpoint, + 0.1 potential"})
G5.COL.update({"V52L0": "#2f9e44", "V52L01": "#e8590c"})
G5.OUT, G5.FIG = os.path.join(VC.E52, "reports"), os.path.join(VC.E52, "figures")

if __name__ == "__main__":
    sys.argv = [sys.argv[0], "--suffix", "2_full"]
    G5.main()
