"""EXP5 Task 0 -- aggregation of the POST-HOC phantom-gain diagnostic (t04_phantom.py; CPU, evaluation side).
Per protocol (M, O) x model (frozen, C): phantom_share (candidate mean / argmax candidate) per stratum with cluster
bootstrap over houses / scenes; fidelity check of the reconstruction (Spearman of the recomputed DistP1 with the
generation-time distp1_K4, per state); counterfactual ranking: Spearman / regret vs the true one-step gain of
DistP1 and of DistP1 - phantom weight ("observed-EMPTY voxels removed"), same ties / fallback rule as t04_metrics.
Output W/analysis/phantom.json
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t04_metrics as TM  # noqa: E402
import t045_analyze as TA  # noqa: E402

import numpy as np  # noqa: E402

ROOTS = {"M": "fixed_state", "O": "fixed_state_out"}


def main():
    res = {"doc": __doc__}
    for proto, root in ROOTS.items():
        for model in ("frozen", "C"):
            rows = []
            for p in sorted(glob.glob(os.path.join(TM.W, "analysis", f"phantom_{proto}_{model}_part*.json"))):
                rows += json.load(open(p))["rows"]
            per = []
            for r in rows:
                gen = json.load(open(os.path.join(TM.W, root, model, f"{r['id']}.json")))
                gain = np.load(os.path.join(TM.W, root, "labels", f"{r['id']}.npz"))["gain"]
                d = np.array([c["distp1"] for c in r["candidates"]])
                ph = np.array([c["phantom"] for c in r["candidates"]])
                assert len(d) == len(gain) == len(gen["distp1_K4"])
                orig = TM.one_step(d, gain, r["id"])
                cf = TM.one_step(np.clip(d - ph, 0, None), gain, r["id"])
                per.append({"id": r["id"], "stratum": r["stratum"], "house": r["house"],
                            "share_mean": r["phantom_share_mean"], "share_pick": r["phantom_share_pick"],
                            "fidelity_rho": TM.spearman(d, gen["distp1_K4"]),
                            "fidelity_rel": float(np.abs(d - np.asarray(gen["distp1_K4"])).sum() / max(1e-9, np.abs(gen["distp1_K4"]).sum())),
                            "spearman": orig["spearman"], "regret": orig["regret"], "top1": orig["top1"],
                            "cf_spearman": cf["spearman"], "cf_regret": cf["regret"], "cf_top1": cf["top1"]})
            agg = {}
            for grp in ["all"] + TA.STRATA:
                sel = [x for x in per if grp == "all" or x["stratum"] == grp]
                g = [x["house"] for x in sel]
                agg[grp] = {"n": len(sel), **{k: TA.cboot([x[k] for x in sel], g) for k in
                                               ("share_mean", "share_pick", "fidelity_rho", "fidelity_rel", "spearman", "regret",
                                                "top1", "cf_spearman", "cf_regret", "cf_top1")},
                            "cf_minus_orig_spearman": TA.cboot([x["cf_spearman"] - x["spearman"] if x["cf_spearman"] is not None
                                                                and x["spearman"] is not None else None for x in sel], g),
                            "cf_minus_orig_regret": TA.cboot([x["cf_regret"] - x["regret"] for x in sel], g)}
            res[f"{proto}_{model}"] = {"aggregate": agg, "per_state": per}
            a = agg["all"]
            print(proto, model, len(per), {k: round(v["mean"], 3) if isinstance(v, dict) else v for k, v in a.items() if v})
    json.dump(res, open(os.path.join(TM.W, "analysis", "phantom.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
