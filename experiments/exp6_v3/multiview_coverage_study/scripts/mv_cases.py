"""Multi-View Coverage Reasoning Study -- choose the case states for the figures (CPU, read-only) -> results/cases.json.
Rules (fixed before looking at any case figure):
  correct  E3 or G2 state, D-v2 Top-1 on the official one-step gain, >= 5 legal cameras, best true gain >= 1 pp, largest spread of the
           true one-step gains among such states (a decision that matters)
  wrong    D-v2 not Top-1, regret >= 1 pp, the same value function on TRUE depth would have picked the best camera (the error comes from
           the predictions), largest regret; preferably a different scene than 'correct'
Shown cameras per case: the D-v2 pick, the true best camera, and the camera with the largest predicted one-step value among the rest."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(MV)))
ST = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "states")


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(ST, "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    good, bad = [], []
    for r in rows:
        g = np.asarray(r["oracle_one_step_pp"], float)
        if not np.isfinite(g).any():
            continue
        x = r["rank"]["dv2"]["official"]
        pd_pick = int(np.argmax(np.where(np.isfinite(g), r["gtvalue_one_step"], -np.inf)))
        if x.get("top1") and r["n_legal"] >= 5 and np.nanmax(g) >= 1.0:
            good.append((float(np.nanmax(g) - np.nanmin(g)), r))
        if x.get("top1") is False and x["regret"] >= 1.0 and g[pd_pick] >= np.nanmax(g) - 1e-4:
            bad.append((x["regret"], r))
    good.sort(key=lambda z: -z[0])
    bad.sort(key=lambda z: -z[0])
    cases = []
    for name, lst in (("correct", good), ("wrong", bad)):
        if not lst:
            continue
        pick_r = lst[0][1]
        if name == "wrong" and cases:
            other = [r for _, r in lst if r["scene"] != cases[0]["scene"]]
            pick_r = other[0] if other else pick_r
        r = pick_r
        keys = [tuple(k) for k in r["keys1"]]
        g = np.asarray(r["oracle_one_step_pp"], float)
        pk, best = r["rank"]["dv2"]["official"]["pick"], int(np.nanargmax(g))
        pred = np.asarray(r["K_study"]["scores"]["one_step_first4"], float)
        rest = [j for j in np.argsort(-pred) if j not in (pk, best)]
        show = [pk] + ([best] if best != pk else []) + rest[:1 if best != pk else 2]
        cases.append({"name": name, "set": r["set"], "tag": r["tag"], "scene": r["scene"], "start": r["start"], "t": r["t"],
                      "show_keys": [list(keys[j]) for j in show], "render_keys": [list(keys[j]) for j in show],
                      "title": (f"{'Correct' if name == 'correct' else 'Wrong'} D-v2 decision — {r['scene']} ({r['set'].upper()} {r['tag']}), step {r['t']}: "
                                f"D-v2 picks camera {pk} (true gain {g[pk]:.2f} pp), best is camera {best} ({np.nanmax(g):.2f} pp)")})
    json.dump({"rules": __doc__, "cases": cases}, open(os.path.join(MV, "results", "cases.json"), "w"), indent=1)
    for c in cases:
        print(c["name"], c["set"], c["tag"], c["t"], c["show_keys"], c["title"])


if __name__ == "__main__":
    main()
