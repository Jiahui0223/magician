"""EXP5 Task 1.3 S-CAL (CPU; Train-6 S-c only, BEFORE any Dev 100-step run; protocol §4.2-§4.4):
  delta  = smallest of {2, 4, 8, 16} voxels >= the median depth difference of both-new-hit sample pairs on the S-c
           calibration states (V2 predictions only, no GT), else 16; saturation P(D >= delta) reported per grid value
  U valid: finite everywhere and NOT (all candidates equal in > 50% of the states), separately for U_raw and DistU
  lambda = argmin over {0.25, 0.5, 1, 2} of the mean regret of GU against the true one-step gain (S-GAIN-C), ties ->
           smaller; the regret of lambda = 0 (pure DistP1) is reported for reference
Inputs: W/eval_sc/<name>/states_*.jsonl (t12_eval_sc.py), W/fixed_state/sc_gain/gain_*.json (t13_sc_gain.py).
Writes frozen_values.json["scoring"] (refuses if it already exists) and W/fixed_state/calibration/calibration.json.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402

import m_common as MC  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402

SC = VF.load_cfg("scoring_v1.json")


def load_states(name):
    rows = []
    for p in sorted(glob.glob(os.path.join(VF.WORK, "eval_sc", name, "states_*.jsonl"))):
        rows += [json.loads(x) for x in open(p)]
    return rows


def pick_of(scores, scene, start, t):
    s = np.asarray(scores, float)
    if s.max() == 0.0:
        return MC.fallback_choice(scene, start, t, len(s))
    return MC.argmax_lowest(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="main")
    ap.add_argument("--dry", action="store_true", help="print only, do not freeze")
    a = ap.parse_args()
    rows = load_states(a.name)
    gains = {}
    for p in glob.glob(os.path.join(VF.WORK, "fixed_state", "sc_gain", "gain_*.json")):
        g = json.load(open(p))
        for st in g["states"]:
            gains[(g["scene"], g["traj"], st["t"])] = (st["keys"], st["g"], g["rec_num"])
    grid = SC["U"]["delta_grid_vox"]
    D = np.concatenate([np.asarray(r["deltas_both_new_vox"], float) for r in rows]) if rows else np.zeros(0)
    med = float(np.median(D)) if len(D) else None
    delta = next((g for g in grid if med is not None and g >= med), grid[-1])
    sat = {str(g): float((D >= g).mean()) if len(D) else None for g in grid}
    # U validity at the chosen delta
    valid = {}
    for u in ("u_raw", "dist_u"):
        vals = [r[f"{u}@{delta}"] for r in rows if r["n"] > 0]
        finite = all(np.all(np.isfinite(v)) for v in vals)
        const = float(np.mean([len(set(v)) == 1 for v in vals])) if vals else 1.0
        valid[u] = {"finite": finite, "const_state_frac": const, "valid": finite and const <= SC["U"]["validity_const_frac_max"]}
    # lambda
    res, n_used, missing = {}, 0, 0
    lams = [0.0] + list(SC["GU"]["lambda_grid"])
    reg = {lam: [] for lam in lams}
    for r in rows:
        key = (r["scene"], r["traj"], r["t"])
        if key not in gains or r["n"] == 0:
            missing += key not in gains
            continue
        keys, g, num = gains[key]
        assert keys == r["keys"], key
        g = np.asarray(g, float)
        n_used += 1
        for lam in lams:
            if lam == 0.0:
                pk = pick_of(r["distp1"], r["scene"], num, r["t"])
            else:
                du = r[f"dist_u@{delta}"]
                if max(r["distp1"]) == 0.0 and max(du) == 0.0:
                    pk = MC.fallback_choice(r["scene"], num, r["t"], len(g))
                else:
                    pk = MC.argmax_lowest(SCO.gu_scores(r["distp1"], du, lam))
            reg[lam].append(float(g.max() - g[pk]))
    mean_reg = {str(lam): (float(np.mean(v)) if v else None) for lam, v in reg.items()}
    cand = [(mean_reg[str(lam)], lam) for lam in SC["GU"]["lambda_grid"] if mean_reg[str(lam)] is not None]
    best = min(cand, key=lambda x: (round(x[0], 12), x[1]))[1] if cand else None
    out = {"rule": "protocol.md v1.0 §4.2-§4.4", "model": a.name, "n_states_scored": len(rows), "n_states_with_gain": n_used,
           "n_states_missing_gain": missing, "both_new_pairs": int(len(D)), "median_delta_vox": med, "delta_vox": delta,
           "saturation_by_delta": sat, "u_validity": valid, "u_valid": valid["u_raw"]["valid"] and valid["dist_u"]["valid"],
           "u_raw_valid": valid["u_raw"]["valid"], "dist_u_valid": valid["dist_u"]["valid"],
           "mean_regret_by_lambda": mean_reg, "lambda": best}
    print(json.dumps(out, indent=1))
    od = os.path.join(VF.WORK, "fixed_state", "calibration")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, "calibration.json" if not a.dry else "calibration_dry.json"), "w"), indent=1)
    if not a.dry:
        p = os.path.join(VF.T1, "frozen_values.json")
        fv = json.load(open(p))
        assert "scoring" not in fv, "scoring already frozen"
        fv["scoring"] = out
        json.dump(fv, open(p + ".tmp", "w"), indent=1)
        os.replace(p + ".tmp", p)


if __name__ == "__main__":
    main()
