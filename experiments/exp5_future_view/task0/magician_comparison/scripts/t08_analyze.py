"""EXP5 Task 0 / T0.8 (and T0.11) -- aggregation of the unified per-episode metrics (metric_replay/episodes/<group>/*.json):
cluster bootstrap over houses (10,000, seed 0) per group; paired differences per episode; per-house table; curves.
Usage: python t08_analyze.py --groups ref,frozen,C,V2K4,V2K1 --ref V2K4 --out t08     (CPU)
Output W/magician_comparison/analysis/<out>.json + figures/<out>_curves.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as M  # noqa: E402

KEYS = ("cov10", "cov20", "cov50", "cov100", "auc", "comp5_10", "comp5_20", "comp5_50", "comp5_100", "comp5vis_100", "comp5s_100",
        "dist_100", "distmed_100", "dists_100", "n_valid_frames", "distinct_stations", "station_revisit_ratio", "pose_revisit_ratio",
        "pose_revisit_steps", "path_m", "zero_gain_steps", "longest_zero_run", "gain_on_revisit_steps", "distinct_regions",
        "region_changes", "decision_s_mean", "wall_s")


def flat(e):
    c = {r["t"]: r for r in e["curve"]}
    out = {**e["cov"], "auc": e["auc"]}
    for t in (10, 20, 50, 100):
        out[f"comp5_{t}"] = c[t]["comp5"]
    out.update({"comp5vis_100": c[100]["comp5_vis"], "comp5s_100": c[100]["comp5_s"], "dist_100": c[100]["dist_mean_cm"],
                "distmed_100": c[100]["dist_median_cm"], "dists_100": c[100]["dist_mean_s_cm"]})
    for k in KEYS:
        if k not in out:
            out[k] = e.get(k)
    return out


def cboot(vals, groups, n=10000, seed=0):
    by = {}
    for v, g in zip(vals, groups):
        if v is not None and np.isfinite(v):
            by.setdefault(g, []).append(v)
    if not by:
        return None
    cm = np.asarray([np.mean(v) for v in by.values()])
    bs = cm[np.random.default_rng(seed).integers(0, len(cm), (n, len(cm)))].mean(1)
    allv = [x for v in by.values() for x in v]
    return {"mean": float(cm.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "n": len(allv), "clusters": int(len(cm)), "n_pos": int(sum(x > 0 for x in allv))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="ref,frozen,C,V2K4,V2K1")
    ap.add_argument("--pairs", default="frozen-V2K4,C-V2K4,V2K1-V2K4,ref-V2K4,frozen-ref,C-ref,V2K4-ref")
    ap.add_argument("--out", default="t08")
    a = ap.parse_args()
    groups = a.groups.split(",")
    eps = M.episodes()
    rows = {}
    for g in groups:
        for tag, scan, lv, st in eps:
            p = os.path.join(M.MC, "metric_replay", "episodes", g, f"{tag}.json")
            if os.path.exists(p):
                e = json.load(open(p))
                rows.setdefault(tag, {"house": scan})[g] = {"flat": flat(e), "curve": e["curve"],
                                                            "coverage_reproduction": e["coverage_reproduction_max_abs_diff"]}
    tags = [t for t, *_ in eps]
    houses = [rows.get(t, {}).get("house") for t in tags]
    summ = {g: {"n": sum(g in rows.get(t, {}) for t in tags)} for g in groups}
    for g in groups:
        for k in KEYS:
            summ[g][k] = cboot([rows[t][g]["flat"].get(k) if g in rows.get(t, {}) else None for t in tags], houses)
    paired = {}
    for pr in a.pairs.split(","):
        x, y = pr.split(":") if ":" in pr else pr.split("-")      # ":" for group names that contain "-" (MAG-B)
        paired[pr] = {k: cboot([(rows[t][x]["flat"][k] - rows[t][y]["flat"][k]) if x in rows.get(t, {}) and y in rows.get(t, {})
                                and rows[t][x]["flat"].get(k) is not None and rows[t][y]["flat"].get(k) is not None else None
                                for t in tags], houses)
                      for k in ("cov20", "cov100", "auc", "comp5_100", "comp5vis_100", "dist_100", "pose_revisit_ratio", "longest_zero_run")}
    per_house = {}
    for g in groups:
        for h in sorted(set(houses)):
            sel = [t for t, hh in zip(tags, houses) if hh == h and g in rows.get(t, {})]
            if sel:
                per_house.setdefault(h, {})[g] = {k: float(np.mean([rows[t][g]["flat"][k] for t in sel])) for k in ("cov100", "auc", "comp5_100", "dist_100")}
                per_house[h]["n_levels"] = len(sel)
    curves = {g: {k: np.mean([[r[k] for r in rows[t][g]["curve"]] for t in tags if g in rows.get(t, {})], 0).tolist()
                  for k in ("comp5", "comp5_s")} for g in groups}
    out = {"doc": __doc__, "groups": groups, "summary": summ, "paired": paired, "per_house": per_house, "mean_curves": curves,
           "per_episode": {t: {g: rows[t][g]["flat"] for g in groups if g in rows.get(t, {})} for t in tags if t in rows},
           "coverage_reproduction_max": max(rows[t][g]["coverage_reproduction"] for t in rows for g in rows[t] if g != "house")}
    od = os.path.join(M.MC, "analysis")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"{a.out}.json"), "w"), indent=1, default=float)
    for g in groups:
        s = summ[g]
        print(g, s["n"], {k: round(s[k]["mean"], 3) for k in ("cov20", "cov100", "auc", "comp5_100", "comp5vis_100", "comp5s_100", "dist_100",
                                                              "distmed_100", "pose_revisit_ratio", "longest_zero_run", "path_m") if s.get(k)})
    figure(out, a.out)


def figure(out, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    for g in out["groups"]:
        cs = [M.stored_curve(g, t)["coverage"] for t in out["per_episode"] if g in out["per_episode"][t]]
        if cs:
            ax[0].plot(np.mean(cs, 0), label=g)
        ax[1].plot(out["mean_curves"][g]["comp5"], label=g)
    ax[0].set_title("C_obs (level scale)")
    ax[1].set_title("Comp@5cm (all in-box GT vertices, full-depth cloud)")
    for x in ax:
        x.set_xlabel("step")
        x.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(M.MC, "figures"), exist_ok=True)
    fig.savefig(os.path.join(M.MC, "figures", f"{name}_curves.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
