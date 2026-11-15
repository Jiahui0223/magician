"""Experiment G -- Table 3 restricted to unseen scenes (CPU). Reads results/table3/per_scene.csv (D0 / G1 / G4 / G0 scene means) and the
Table 1 manifest for MAGICIAN (context only). Groups: Dev+Test-8 = scenes not in our Train-6 (unseen by D0 and G; Dev-5 was used for
validation monitoring / D1 selection), Unseen-6 = scenes no method trained on (alhambra, neuschwanstein, eiffel, bridge, barts,
sestino_museum; definition of fig3_v3). Paired scene bootstrap 10,000, seed 0. Writes results/table3/table3_G_unseen.{csv,md}."""
import csv
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
G = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_G_generative_vs_deterministic", "results", "table3")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402

K = ["AUC@20", "AUC@50", "AUC@100", "Final"]
ps = {}
for r in csv.DictReader(open(os.path.join(G, "per_scene.csv"))):
    ps.setdefault(r["arm"], {})[r["scene"]] = {k: float(r[k]) for k in K}
E = L.load_manifest("MAG-B")
met = lambda c: dict(zip(K, [np.trapz(c[:21]) / 20, np.trapz(c[:51]) / 50, np.trapz(c[:101]) / 100, c[100]]))
ps["MAGICIAN"] = {s: {k: float(np.mean([met(np.asarray(E[(s, i)]["coverage_official"]))[k] for i in range(5)])) for k in K} for s in L.SCENE_ORDER}
split = {s: L.SPLIT[s] for s in L.SCENE_ORDER}
groups = {"All-14": list(L.SCENE_ORDER), "Dev+Test-8 (unseen by D0 / G)": [s for s in L.SCENE_ORDER if split[s] != "train"],
          "Unseen-6 (no method trained on)": ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]}
arms = ["D0", "G1", "G4", "G0", "MAGICIAN"]


def boot(d):
    d = np.asarray(d)
    b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5), int((d > 0).sum())


rows, md = [], []
for gname, sc in groups.items():
    md.append(f"### {gname}: {', '.join(sc)}\n\n| Arm | " + " | ".join(K) + " |\n|---|" + "---|" * len(K))
    for a in arms:
        v = {k: np.mean([ps[a][s][k] for s in sc]) for k in K}
        rows.append([gname, a] + [f"{v[k]:.4f}" for k in K])
        md.append(f"| {a} | " + " | ".join(f"{v[k]:.3f}" for k in K) + " |")
    md.append("\n| Paired | " + " | ".join(K) + " |\n|---|" + "---|" * len(K))
    for a, b in (("D0", "G4"), ("D0", "G1"), ("G4", "G1"), ("D0", "MAGICIAN"), ("G4", "MAGICIAN")):
        cells = []
        for k in K:
            m, lo, hi, n = boot([ps[a][s][k] - ps[b][s][k] for s in sc])
            cells.append(f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}] ({n}/{len(sc)})")
        md.append(f"| {a} − {b} | " + " | ".join(cells) + " |")
    md.append("")
with open(os.path.join(G, "table3_G_unseen.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["group", "arm"] + K); w.writerows(rows)
open(os.path.join(G, "table3_G_unseen.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
