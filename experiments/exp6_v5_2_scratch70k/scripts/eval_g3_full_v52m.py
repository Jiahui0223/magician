"""EXP6 V5.2-M -- full G3 analysis (seen2scene env, CPU): the shared exp6_v5/scripts/eval_g3.py main (exp6_v3 readers,
paired scene / floor bootstrap) with the groups V52ML0 / V52ML01 (work_dirs/exp6_v5_2_scratch70k/eval/g3) added to the
old V5.2 pilot (V52L0 / V52L01), V5 / V4C, V4 / V31 / A1, B2 / E2-D100 / MAGICIAN and the history (same episodes).
Output experiments/exp6_v5_2_scratch70k/reports/g3_v52m_full.{json,md} + figures/g3_v52m_full_coverage_curves.png;
other experiments' files untouched. Runs only when the V5.2-M full G3 has outputs. Usage: python eval_g3_full_v52m.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5_2", "scripts"))
import v52_common as VC  # noqa: E402

sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5", "scripts"))
import eval_g3 as G5  # noqa: E402

W52 = os.path.join(VC.W52, "eval")
WM = os.path.join(R, "work_dirs", "exp6_v5_2_scratch70k", "eval", "g3")
G5.AG.ROOTS.update({g: (os.path.join(W52, "closed_loop"), os.path.join(W52, "metrics", "outdoor"), "V2M") for g in ("V52L0", "V52L01")})
G5.AG.ROOTS.update({g: (os.path.join(WM, "closed_loop"), os.path.join(WM, "metrics", "outdoor"), "V2M") for g in ("V52ML0", "V52ML01")})
G5.GRP = ("MAG-G", "MAG-B", "V2O", "B0", "B2", "E2D", "T1", "V31", "V4", "A1", "V4C", "V5", "V52L0", "V52L01", "V52ML0", "V52ML01")
G5.PAIRS = (("V52ML0", "V52L0"), ("V52ML01", "V52L01"), ("V52ML0", "V5"), ("V52ML01", "V5"), ("V52ML01", "V52ML0"), ("V52ML0", "V4C"), ("V52ML01", "V4C"),
            ("V52ML0", "B2"), ("V52ML01", "B2"), ("V52ML0", "E2D"), ("V52ML01", "E2D"), ("V52ML0", "MAG-B"), ("V52ML01", "MAG-B"), ("V52ML0", "MAG-G"), ("V52ML0", "V4"))
G5.LAB.update({"V52L0": "old V5.2 λ=0", "V52L01": "old V5.2 λ=0.1", "V52ML0": "V5.2-M λ=0 (visible only)", "V52ML01": "V5.2-M λ=0.1 (visible + potential)"})
G5.SRC.update({"V52L0": "V5@12k + hazard 6k (no flow)", "V52L01": "same, + 0.1 potential",
               "V52ML0": "V5.2-M scratch 70k (27.0M), direct visible score", "V52ML01": "same checkpoint, + 0.1 potential"})
G5.COL.update({"V52L0": "#94d82d", "V52L01": "#ffa94d", "V52ML0": "#2b8a3e", "V52ML01": "#d9480f"})
G5.OUT, G5.FIG = os.path.join(E, "reports"), os.path.join(E, "figures")

if __name__ == "__main__":
    if not os.path.isdir(os.path.join(WM, "closed_loop", "outdoor", "V52ML0")) or not os.path.exists(os.path.join(E, "AUTO_DECISION.json")) \
            or "full" not in open(os.path.join(E, "AUTO_DECISION.json")).read():
        sys.exit(0)
    os.makedirs(G5.OUT, exist_ok=True); os.makedirs(G5.FIG, exist_ok=True)
    sys.argv = [sys.argv[0], "--suffix", "2m_full"]
    G5.main()
