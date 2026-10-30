"""EXP5 Task 0.4 -- fixed-state selection (CPU; pre-registered rule, before any model output):
for every indoor reference trajectory (W/preprocessing/traj/<level>_S<start>/traj.json): for each coverage stratum
[1,5), [5,10), [10,20), [20,40) % take the FIRST step t whose level-scale C_obs lies in the stratum (none -> the stratum
is missing for that trajectory, reported, never filled). Writes W/manifests/indoor_states.json (+ support table).
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

W = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0"
STRATA = [("c01-05", 0.01, 0.05), ("c05-10", 0.05, 0.10), ("c10-20", 0.10, 0.20), ("c20-40", 0.20, 0.40)]


def main():
    states, support = [], {s[0]: {"states": 0, "houses": set(), "levels": set()} for s in STRATA}
    missing = []
    for p in sorted(glob.glob(os.path.join(W, "preprocessing", "traj", "*", "traj.json"))):
        tr = json.load(open(p))
        name = os.path.basename(os.path.dirname(p))
        key = tr["scene"]["key"]
        house = key.split("_L")[0]
        for nm, lo, hi in STRATA:
            t = next((r["t"] for r in tr["rows"] if lo <= r["cov"] < hi), None)
            if t is None:
                missing.append({"traj": name, "stratum": nm,
                                "cov_t0": tr["rows"][0]["cov"], "cov_t100": tr["rows"][-1]["cov"]})
                continue
            states.append({"id": f"{name}_t{t}", "traj": name, "level_key": key, "house": house, "start": tr["start_pano"],
                           "t": t, "stratum": nm, "cov": tr["rows"][t]["cov"]})
            support[nm]["states"] += 1
            support[nm]["houses"].add(house)
            support[nm]["levels"].add(key)
    sup = {k: {"states": v["states"], "houses": len(v["houses"]), "levels": len(v["levels"])} for k, v in support.items()}
    out = {"rule": __doc__, "states": states, "support": sup, "missing": missing, "n_states": len(states)}
    json.dump(out, open(os.path.join(W, "manifests", "indoor_states.json"), "w"), indent=1)
    print(json.dumps(sup, indent=1), len(states), "states;", len(missing), "missing trajectory x stratum")


if __name__ == "__main__":
    main()
