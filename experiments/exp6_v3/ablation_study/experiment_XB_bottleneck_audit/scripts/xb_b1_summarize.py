"""Experiment XB / B1 -- summary of results/B1_fidelity/*.json (CPU): observed-point retention per representation level and distance stratum, pooled over
trajectories and states (point-weighted) and per trajectory; map-side thin-cell and UNKNOWN-boundary statistics. Output results/B1_fidelity/summary.json."""
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
D = f"{R}/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/results/B1_fidelity"
KEYS = ("s10_vox", "d100_vox", "s10_4x", "d100_4x", "s10_8x_obs", "d100_8x_obs", "d100_4x_thin")
out = {"per_trajectory": {}, "pooled_by_distance": {}, "pooled_overall": {}, "map_4x": {}, "unknown_boundary": {}}
acc = {}; accd = {}; mp = {}; ub = [0, 0]
for f in sorted(glob.glob(D + "/*.json")):
    if f.endswith("summary.json"):
        continue
    j = json.load(open(f)); tag = j["tag"]
    tr = {"scene": j["scene"], "states": len(j["states"]), "overall_last": None, "by_distance_last": None}
    for t, st in sorted(j["states"].items(), key=lambda kv: int(kv[0])):
        n = st["n_points"]
        for k in KEYS:
            v = st["overall"].get(k)
            if v is not None:
                acc.setdefault(k, [0.0, 0.0]); acc[k][0] += v * n; acc[k][1] += n
        for sname, row in st["by_distance"].items():
            for k in KEYS:
                v = row.get(k)
                if v is not None and row["n_dom"]:
                    a = accd.setdefault(sname, {}).setdefault(k, [0.0, 0.0]); a[0] += v * row["n_dom"]; a[1] += row["n_dom"]
        for k, v in st["map_4x"].items():
            mp.setdefault(k, []).append(v)
        ub[0] += st["unknown_boundary_4x"]["cells"]; ub[1] += st["unknown_boundary_4x"]["parent_8x_observed"]
        tr["overall_last"], tr["by_distance_last"], tr["map_last"], tr["unknown_boundary_last"] = st["overall"], st["by_distance"], st["map_4x"], st["unknown_boundary_4x"]
    out["per_trajectory"][tag] = tr
out["pooled_overall"] = {k: v[0] / v[1] for k, v in acc.items() if v[1]}
out["pooled_by_distance"] = {s: {k: v[0] / v[1] for k, v in d.items() if v[1]} for s, d in accd.items()}
out["map_4x"] = {k: {"mean": float(np.mean(v)), "min": int(min(v)), "max": int(max(v))} for k, v in mp.items()}
out["unknown_boundary"] = {"cells": ub[0], "parent_8x_observed": ub[1], "share": ub[1] / max(1, ub[0])}
json.dump(out, open(D + "/summary.json", "w"), indent=1)
print("trajectories:", list(out["per_trajectory"]))
print("pooled overall:", {k: round(v, 4) for k, v in out["pooled_overall"].items()})
for s, d in out["pooled_by_distance"].items():
    print("  dist", s, {k: round(v, 4) for k, v in d.items()})
print("map 4x:", {k: round(v["mean"], 1) for k, v in out["map_4x"].items()})
print("unknown boundary 4x cells with an observed 8x parent:", round(out["unknown_boundary"]["share"], 3))
