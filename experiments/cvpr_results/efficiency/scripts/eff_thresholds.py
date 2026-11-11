"""EXP8 Task 6 -- steps to coverage thresholds from the REAL Table 1 trajectories (CPU; canonical records of
work_dirs/cvpr_results/table1/episodes/<method>/<scene>_s<k>.json, the official coverage C[0..100]).

Per episode: first step t with C[t] >= thr (thr in 0.5 / 0.7 / 0.8 / 0.9); not reached within 100 steps -> censored (> 100).
Per scene (5 starts): reached fraction and the median step with censored episodes counted as 101 (an upper-censored value; the
median is reported as '>100' when 3 or more starts never reach the threshold). All-14: the median of the scene medians and the
number of scenes whose median is defined; plus, over the episodes where BOTH methods reach the threshold (same scene / start),
the paired difference of steps. Nothing is averaged over reached episodes only.
Output: experiments/cvpr_results/efficiency/tables/steps_to_threshold{,_episodes}.csv"""
import csv
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
OUT = os.path.join(R, "experiments", "cvpr_results", "efficiency", "tables")
SCENES = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein",
          "eiffel", "bridge", "barts", "sestino_museum"]
METHODS = {"V3-T0-D1": "RAYSCOUT", "MAG-B": "MAGICIAN"}
THR = (0.5, 0.7, 0.8, 0.9)


def first_reach(c, thr):
    idx = np.flatnonzero(np.asarray(c) >= thr)
    return int(idx[0]) if len(idx) else None


def main():
    os.makedirs(OUT, exist_ok=True)
    eps = []
    for m in METHODS:
        for s in SCENES:
            for k in range(5):
                r = json.load(open(os.path.join(EP, m, f"{s}_s{k}.json")))
                c = r["coverage_official"]
                assert len(c) == 101, (m, s, k, len(c))
                eps.append({"method": m, "scene": s, "start": k, **{f"t@{thr}": first_reach(c, thr) for thr in THR}})
    with open(os.path.join(OUT, "steps_to_threshold_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(eps[0]))
        w.writeheader()
        w.writerows(eps)
    rows = []
    for thr in THR:
        key = f"t@{thr}"
        for m in METHODS:
            meds, fr = [], []
            for s in SCENES:
                v = [e[key] for e in eps if e["method"] == m and e["scene"] == s]
                reached = [x for x in v if x is not None]
                fr.append(len(reached) / len(v))
                capped = sorted(x if x is not None else 101 for x in v)
                meds.append(float(np.median(capped)) if len(reached) >= 3 else None)
            defined = [x for x in meds if x is not None]
            rows.append({"threshold": thr, "method": METHODS[m], "episodes_reached": sum(1 for e in eps if e["method"] == m and e[key] is not None),
                         "episodes_total": sum(1 for e in eps if e["method"] == m), "scenes_with_median": len(defined),
                         "median_of_scene_medians": float(np.median([x if x is not None else 101 for x in meds])),
                         "scene_medians": ";".join("NA" if x is None else f"{x:g}" for x in meds)})
        both = [(a[key], b[key]) for a in eps if a["method"] == "V3-T0-D1" for b in eps
                if b["method"] == "MAG-B" and b["scene"] == a["scene"] and b["start"] == a["start"] and a[key] is not None and b[key] is not None]
        d = np.asarray([x - y for x, y in both], float)
        only = {m: sum(1 for a in eps if a["method"] == m and a[key] is not None and not any(
            b["method"] != m and b["scene"] == a["scene"] and b["start"] == a["start"] and b[key] is not None for b in eps)) for m in METHODS}
        rows.append({"threshold": thr, "method": "paired (RAYSCOUT - MAGICIAN), both reached", "episodes_reached": len(both), "episodes_total": 70,
                     "scenes_with_median": None, "median_of_scene_medians": float(np.median(d)) if len(d) else None,
                     "scene_medians": f"mean {d.mean():+.2f} steps; RAYSCOUT fewer in {int((d < 0).sum())}, equal {int((d == 0).sum())}, more {int((d > 0).sum())}; "
                                      f"reached by RAYSCOUT only {only['V3-T0-D1']}, by MAGICIAN only {only['MAG-B']}" if len(d) else ""})
    with open(os.path.join(OUT, "steps_to_threshold.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
