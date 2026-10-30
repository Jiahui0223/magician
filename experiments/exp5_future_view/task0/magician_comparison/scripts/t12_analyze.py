"""EXP5 Task 0 / T0.12 -- aggregation of the grid-protocol episodes (protocol v1.3 §22-§23; CPU, seen2scene env).
Reads metric_replay/episodes/G-*/<tag>.json (t08_metrics.py episodes --grid) and the closed-loop planner logs.
  summary   per group: house-cluster mean (5 houses x 2 starts) + cluster bootstrap 95 % CI (10,000, seed 0)
  paired    per-episode differences, same bootstrap, n_pos = episodes with a positive difference
  station   descriptive reference: the station-graph groups (T0.8 / T0.10 metric replays) on the 5 first starts, @100 only
  beam      MAGICIAN: share of steps where the executed pick differs from the layer-0 greedy argmax
Output W/magician_comparison/analysis/t12.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_common as M  # noqa: E402
from t08_analyze import cboot  # noqa: E402

GG = M.GRID_GROUPS
ST_GROUPS = ("ref", "V2K4", "V2K4-NV", "MAG-G", "MAG-B")
FIRST = ("Z6MFQCViBuw_L0_S6", "oLBMNvg9in8_L1_S36", "QUCTc6BB5sX_L1_S34", "2azQ1b91cZZ_L0_S60", "TbHJrupSAjP_L1_S112")
KEYS = ("cov10", "cov20", "cov50", "cov100", "cov150", "auc", "auc150", "comp5_20", "comp5_50", "comp5_100", "comp5_150",
        "comp5vis_100", "comp5vis_150", "dist_100", "dist_150", "distinct_stations", "station_revisit_ratio", "pose_revisit_ratio",
        "path_m", "step_m", "n_moves", "zero_gain_steps", "longest_zero_run", "distinct_regions", "decision_s_mean")
PAIRS = ("G-V2K4:G-MAG-B", "G-V2K4:G-MAG-G", "G-MAG-B:G-MAG-G", "G-V2K4-NV:G-MAG-G", "G-V2K4-NV:G-MAG-B", "G-V2K4-NV:G-V2K4",
         "G-MAG-G:G-RAND", "G-V2K4:G-RAND")
PKEYS = ("cov20", "cov50", "cov100", "cov150", "auc", "auc150", "comp5_100", "comp5_150", "dist_150", "pose_revisit_ratio", "longest_zero_run")


def flat(e):
    c = {r["t"]: r for r in e["curve"]}
    out = {**e["cov"], "auc": e["auc"], "auc150": e.get("auc150")}
    for t in (20, 50, 100, 150):
        if t in c:
            out[f"comp5_{t}"] = c[t]["comp5"]
    for t in (100, 150):
        if t in c:
            out[f"comp5vis_{t}"] = c[t]["comp5_vis"]
            out[f"dist_{t}"] = c[t]["dist_mean_cm"]
    out["step_m"] = e["path_m"] / e["n_moves"] if e["n_moves"] else None
    for k in KEYS:
        if k not in out:
            out[k] = e.get(k)
    return out


def beam_override(g, tag):
    L = [json.loads(x) for x in open(os.path.join(M.MC, "grid_protocol", "closed_loop", g, tag, "planner.jsonl"))]
    n = d = 0
    for x in L:
        s = [v if v is not None else -np.inf for v in x["scores"]]
        if x["pick"] is None or not s or x["fallback"]:
            continue
        n += 1
        d += int(x["pick"] != int(np.argmax(s)))
    return d / n if n else None


def main():
    eps = M.grid_episodes()
    tags = [t for t, *_ in eps]
    houses = [s for _, s, _, _ in eps]
    rows = {t: {} for t in tags}
    for g in GG:
        for t in tags:
            p = os.path.join(M.MC, "metric_replay", "episodes", g, f"{t}.json")
            if os.path.exists(p):
                e = json.load(open(p))
                rows[t][g] = flat(e)
                rows[t][g]["coverage_reproduction"] = e["coverage_reproduction_max_abs_diff"]
                if g.startswith("G-MAG"):
                    rows[t][g]["beam_override"] = beam_override(g, t)
    summ = {g: {"n": sum(g in rows[t] for t in tags)} for g in GG}
    for g in GG:
        for k in KEYS + ("beam_override",):
            summ[g][k] = cboot([rows[t][g].get(k) if g in rows[t] else None for t in tags], houses)
    paired = {}
    for pr in PAIRS:
        x, y = pr.split(":")
        paired[pr] = {k: cboot([rows[t][x][k] - rows[t][y][k] if x in rows[t] and y in rows[t] and rows[t][x].get(k) is not None
                                and rows[t][y].get(k) is not None else None for t in tags], houses) for k in PKEYS}
    curves = {}
    for g in GG:
        by = {}
        for t, h in zip(tags, houses):
            if g in rows[t]:
                by.setdefault(h, []).append(np.asarray(M.stored_curve(g, t)["coverage"]))
        if by:
            curves[g] = np.mean([np.mean(v, 0) for v in by.values()], 0).tolist()
    station = {}
    for g in ST_GROUPS:
        for t in FIRST:
            p = os.path.join(M.MC, "metric_replay", "episodes", g, f"{t}.json")
            if os.path.exists(p):
                e = json.load(open(p))
                f = {**e["cov"], "auc": e["auc"], "comp5_100": e["curve"][100]["comp5"], "path_m": e["path_m"],
                     "step_m": e["path_m"] / e["n_moves"] if e["n_moves"] else None, "distinct_stations": e["distinct_stations"],
                     "pose_revisit_ratio": e["pose_revisit_ratio"]}
                station.setdefault(g, {})[t] = f
    st_mean = {g: {k: float(np.mean([v[k] for v in d.values() if v.get(k) is not None])) for k in
                   ("cov20", "cov50", "cov100", "auc", "comp5_100", "step_m", "distinct_stations", "pose_revisit_ratio")}
               for g, d in station.items()}
    grid_first = {g: {k: float(np.mean([rows[t][g][k] for t in FIRST if g in rows[t]])) for k in
                      ("cov20", "cov50", "cov100", "auc", "comp5_100", "step_m", "distinct_stations", "pose_revisit_ratio")}
                  for g in GG if all(g in rows[t] for t in FIRST)}
    out = {"doc": __doc__, "groups": GG, "summary": summ, "paired": paired, "mean_curves": curves, "per_episode": rows,
           "station_first_starts": {"per_episode": station, "mean": st_mean, "grid_same_episodes_mean": grid_first},
           "coverage_reproduction_max": max([r["coverage_reproduction"] for t in tags for r in rows[t].values()] or [None])}
    od = os.path.join(M.MC, "analysis")
    json.dump(out, open(os.path.join(od, "t12.json"), "w"), indent=1, default=float)
    for g in GG:
        s = summ[g]
        print(g, s["n"], {k: round(s[k]["mean"], 3) for k in ("cov20", "cov100", "cov150", "auc", "auc150", "comp5_150", "dist_150",
                                                              "step_m", "pose_revisit_ratio", "beam_override", "decision_s_mean") if s.get(k)})
    for pr, d in paired.items():
        print(pr, {k: (round(d[k]["mean"], 3), [round(x, 3) for x in d[k]["ci95"]], d[k]["n_pos"]) for k in ("auc", "auc150", "cov150", "comp5_150") if d.get(k)})
    print("station-graph (first starts, @100):", {g: {k: round(v, 3) for k, v in m.items()} for g, m in st_mean.items()})
    print("grid (same first starts, @100):", {g: {k: round(v, 3) for k, v in m.items()} for g, m in grid_first.items()})


if __name__ == "__main__":
    main()
