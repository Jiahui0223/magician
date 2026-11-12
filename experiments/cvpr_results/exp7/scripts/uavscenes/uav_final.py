"""EXP7 / UAVScenes -- final tables (CPU). Every episode must have its official coverage replay equal to the stored curve
(metrics/<METHOD>/<group>/<group>_<tile>_s<k>_r0.json, official_equal_stored). Sets:
  UAV-final  = AMtown_11, AMtown_12, HKairport_00, HKairport_01 (UAV-v0; no method visits under-surface positions, except Random
               0.4 % on AMtown_12) + AMvalley_01_v1, HKisland_00_v1 (UAV-v1: holes sealed, 0 under-surface positions reachable)
  UAV-v0-all = the six v0 tiles (record of the first run; AMvalley_01 / HKisland_00 are affected by the mesh-hole problem)
Metrics on the official coverage C[0..100]: AUC@20 / 50 / 100, Cov@20 / 50, Final = C[100]; tile-equal means of the 10-start means;
paired tile bootstrap (10,000, seed 0) and per-episode win counts (same tile / start). Outputs results/uav_final_*.csv|md|json."""
import csv
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "episodes", "uavscenes")
OUT = os.path.join(R, "experiments", "cvpr_results", "exp7", "results")
M = {"RANDOM": ("random", "random", "Random Walk"), "MAG-B": ("MAG-B", "MAG-B", "MAGICIAN"), "OURS": ("V2M", "V2M", "Ours")}
K = ["AUC@20", "AUC@50", "AUC@100", "Cov@20", "Cov@50", "Final"]
FINAL = ["uav_AMtown_11", "uav_AMtown_12", "uav_HKairport_00", "uav_HKairport_01", "uav_AMvalley_01_v1", "uav_HKisland_00_v1"]
V0 = ["uav_AMtown_11", "uav_AMtown_12", "uav_AMvalley_01", "uav_HKairport_00", "uav_HKairport_01", "uav_HKisland_00"]


def met(c):
    c = np.asarray(c, float)
    return dict(zip(K, [np.trapz(c[:21]) / 20, np.trapz(c[:51]) / 50, np.trapz(c[:101]) / 100, c[20], c[50], c[100]]))


def load(tile, m):
    grp, pre, _ = M[m]
    out, bad = {}, []
    for k in range(10):
        ep = os.path.join(W7, "raw", m, grp, "episodes", f"ep_{pre}_{tile}_s{k}_r0.json")
        mt = os.path.join(W7, "metrics", m, grp, f"{grp}_{tile}_s{k}_r0.json")
        if not (os.path.exists(ep) and os.path.exists(mt)):
            bad.append((k, "missing"))
            continue
        c = json.load(open(ep))["coverage"]
        if not (json.load(open(mt)).get("official_equal_stored") is True and len(c) == 101):
            bad.append((k, "replay"))
            continue
        out[k] = met(c)
    return out, bad


def table(tiles, name):
    data, bad = {}, []
    for t in tiles:
        for m in M:
            e, b = load(t, m)
            data[(t, m)] = e
            bad += [(t, m) + x for x in b]
    tm = {(t, m): {k: float(np.mean([v[k] for v in data[(t, m)].values()])) for k in K} for t in tiles for m in M if len(data[(t, m)]) == 10}
    rows = []
    for m in M:
        done = [t for t in tiles if (t, m) in tm]
        rows.append({"set": name, "method": M[m][2], "tiles": f"{len(done)}/{len(tiles)}", **{k: round(float(np.mean([tm[(t, m)][k] for t in done])), 4) for k in K}})
    pairs = {}
    for a, b in (("OURS", "MAG-B"), ("OURS", "RANDOM"), ("MAG-B", "RANDOM")):
        ts = [t for t in tiles if (t, a) in tm and (t, b) in tm]
        d = np.array([[tm[(t, a)][k] - tm[(t, b)][k] for k in K] for t in ts])
        bs = d[np.random.default_rng(0).integers(0, len(ts), (10000, len(ts)))].mean(1)
        wins = {k: int(sum(data[(t, a)][s][k] > data[(t, b)][s][k] for t in ts for s in range(10))) for k in K}
        pairs[f"{M[a][2]} - {M[b][2]}"] = {k: {"mean": float(d[:, i].mean()), "ci95": [float(np.percentile(bs[:, i], 2.5)), float(np.percentile(bs[:, i], 97.5))],
                                                "tiles_higher": int((d[:, i] > 0).sum()), "episodes_higher": wins[k], "episodes": 10 * len(ts)} for i, k in enumerate(K)}
    per_tile = [{"set": name, "tile": t, "method": M[m][2], **{k: round(tm[(t, m)][k], 4) for k in K}} for t in tiles for m in M if (t, m) in tm]
    return rows, pairs, per_tile, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    allres = {}
    md = []
    for tiles, name in ((FINAL, "UAV-final"), (V0, "UAV-v0-all")):
        rows, pairs, per_tile, bad = table(tiles, name)
        allres[name] = {"rows": rows, "pairs": pairs, "invalid_or_missing": bad}
        with open(os.path.join(OUT, f"uav_final_{name}_per_tile.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_tile[0])); w.writeheader(); w.writerows(per_tile)
        md.append(f"### {name}\n| Method | tiles | " + " | ".join(K) + " |\n|---|---|" + "---|" * len(K))
        md += [f"| {r['method']} | {r['tiles']} | " + " | ".join(f"{r[k]:.3f}" for k in K) + " |" for r in rows]
        for p, v in pairs.items():
            md.append(f"\n{p}: " + "; ".join(f"{k} {v[k]['mean']:+.3f} [{v[k]['ci95'][0]:+.3f}, {v[k]['ci95'][1]:+.3f}] (tiles {v[k]['tiles_higher']}/{len(tiles)}, episodes {v[k]['episodes_higher']}/{v[k]['episodes']})" for k in ("AUC@20", "AUC@100", "Final")))
        md.append(f"\ninvalid / missing: {bad}\n")
    json.dump(allres, open(os.path.join(OUT, "uav_final_summary.json"), "w"), indent=1)
    open(os.path.join(OUT, "uav_final_summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
