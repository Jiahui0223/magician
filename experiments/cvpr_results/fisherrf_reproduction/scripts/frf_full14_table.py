"""FisherRF-full (f1c4) -- All-14 table in the CVPR Table 1 format + paired scene bootstrap (CPU, seen2scene env).
Reads full/outdoor/closed_loop_f1c4/<scene>_s<k>/ (episode curve must equal the official replay: official_equal_stored), computes
AUC@20/50/100, Cov@20/50, Final on the official coverage, scene-equal means, and paired differences against the Table 1 canonical
records (V3-T0-D1, MAG-B, MACARONS, RANDOM) and the D3a ablation. Writes results/full14_f1c4_table.{json,md}."""
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
ROOT = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction", "full", "outdoor", "closed_loop_f1c4")
D3A = os.path.join(R, "work_dirs", "cvpr_results", "fisherrf_reproduction", "outdoor", "closed_loop")
OUT = os.path.join(R, "experiments", "cvpr_results", "fisherrf_reproduction", "results")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402

K = ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Cov@50", "Final"]
met = lambda c: dict(zip(K, [np.trapz(c[:21]) / 20, np.trapz(c[:51]) / 50, np.trapz(c[:101]) / 100, c[20], c[50], c[100]]))


def load_root(root):
    sc = {}
    for s in L.SCENE_ORDER:
        rows = []
        for k in range(5):
            u = f"{s}_s{k}"
            ep = json.load(open(os.path.join(root, u, "episodes", f"ep_FisherRF_{u}_r0.json")))
            mt = json.load(open(os.path.join(root, u, "metrics", "FisherRF", f"FisherRF_{u}_r0.json")))
            c = np.asarray(ep["coverage"], float)
            assert len(c) == 101 and mt.get("official_equal_stored") is True, u
            rows.append(met(c))
        sc[s] = {kk: float(np.mean([r[kk] for r in rows])) for kk in K}
    return sc


def boot(d):
    d = np.asarray(d, float)
    b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5)), int((d > 0).sum())]


def main():
    sc = {"FisherRF-full": load_root(ROOT), "FisherRF-score-1step (D3a)": load_root(D3A)}
    for m in ("RANDOM", "MACARONS", "MAG-B", "V3-T0-D1"):
        E = L.load(m)
        sc[m] = {s: {kk: float(np.mean([met(np.asarray(E[(s, k)]["coverage_official"], float))[kk] for k in range(5)])) for kk in K} for s in L.SCENE_ORDER}
    rows = {m: {kk: float(np.mean([v[s][kk] for s in L.SCENE_ORDER])) for kk in K} for m, v in sc.items()}
    pairs = {f"FisherRF-full - {m}": {kk: boot([sc["FisherRF-full"][s][kk] - sc[m][s][kk] for s in L.SCENE_ORDER]) for kk in K}
             for m in sc if m != "FisherRF-full"}
    json.dump({"rows": rows, "pairs": pairs, "per_scene": sc}, open(os.path.join(OUT, "full14_f1c4_table.json"), "w"), indent=1)
    md = ["| Method (All-14, scene-equal) | " + " | ".join(K) + " |", "|---|" + "---|" * len(K)]
    md += [f"| {m} | " + " | ".join(f"{rows[m][kk]:.3f}" for kk in K) + " |" for m in rows]
    md.append("")
    md += [f"{p}: " + "; ".join(f"{kk} {v[kk][0]:+.3f} [{v[kk][1]:+.3f}, {v[kk][2]:+.3f}] ({v[kk][3]}/14)" for kk in ("AUC@20", "AUC@100", "Final")) for p, v in pairs.items()]
    open(os.path.join(OUT, "full14_f1c4_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
