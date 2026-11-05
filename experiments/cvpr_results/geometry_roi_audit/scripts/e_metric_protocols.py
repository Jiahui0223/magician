"""Geometry / ROI audit E (CPU, read-only): official coverage vs the stored monotone ('mono') coverage of the same Table 1 episodes.
official = MAGICIAN scene_coverage on the capacity-limited covered scene (1000 points per coarse cell, random replacement);
mono     = share of the same GT target points within eps of ANY 10 % sample point so far (no capacity) -- reproduced independently by
           d_roi_coverage.py. Per model: mean final official / mono; paired D-v2 - D1 and D-v2 - MAG-B differences under both protocols
(scene-level paired bootstrap, 10k, seed 0). Nothing is written outside results/e_metric_protocols.json."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
EP = os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "episodes")
ALL14 = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer", "barts", "colosseum", "eiffel", "pantheon", "sestino_museum",
         "alhambra", "bridge", "neuschwanstein"]
MODELS = ["MAG-B", "B2", "V3", "V3-D1-H3-NORISK", "V3-DV2-KSV"]


def met(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "auc100": float(np.trapz(c) / 100), "final": float(c[100])}


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def main():
    R = {}
    for m in MODELS:
        for s in ALL14:
            for k in range(5):
                p = os.path.join(EP, m, f"{s}_s{k}.json")
                if os.path.exists(p):
                    r = json.load(open(p))
                    R[(m, s, k)] = {"official": met(r["coverage_official"]), "mono": met(r["coverage_mono"]) if r.get("coverage_mono") else None,
                                    "nonmono_steps": int((np.diff(np.asarray(r["coverage_official"])) < -1e-12).sum()),
                                    "max_drop": float(max(0.0, -np.diff(np.asarray(r["coverage_official"])).min()))}
    out = {"models": {}, "pairs": {}}
    for m in MODELS:
        rows = [R[(m, s, k)] for s in ALL14 for k in range(5) if (m, s, k) in R and R[(m, s, k)]["mono"]]
        sc = {s: {pr: {q: float(np.mean([R[(m, s, k)][pr][q] for k in range(5)])) for q in ("auc20", "auc100", "final")} for pr in ("official", "mono")} for s in ALL14
              if all((m, s, k) in R and R[(m, s, k)]["mono"] for k in range(5))}
        out["models"][m] = {"episodes": len(rows), "scenes": len(sc),
                            **{f"{pr}_{q}": float(np.mean([sc[s][pr][q] for s in sc])) for pr in ("official", "mono") for q in ("auc20", "auc100", "final")},
                            "final_mono_minus_official_mean": float(np.mean([sc[s]["mono"]["final"] - sc[s]["official"]["final"] for s in sc])),
                            "final_mono_minus_official_by_scene": {s: sc[s]["mono"]["final"] - sc[s]["official"]["final"] for s in sc},
                            "episodes_with_official_decrease": sum(r["nonmono_steps"] > 0 for r in rows), "mean_decreasing_steps": float(np.mean([r["nonmono_steps"] for r in rows])),
                            "max_single_step_drop": float(max(r["max_drop"] for r in rows))}
        out["models"][m]["_sc"] = sc
    for a, b in (("V3-DV2-KSV", "V3-D1-H3-NORISK"), ("V3-DV2-KSV", "MAG-B"), ("V3-D1-H3-NORISK", "MAG-B")):
        sa, sb = out["models"][a]["_sc"], out["models"][b]["_sc"]
        common = [s for s in ALL14 if s in sa and s in sb]
        o = {"scenes": len(common)}
        for pr in ("official", "mono"):
            for q in ("auc20", "auc100", "final"):
                d = [sa[s][pr][q] - sb[s][pr][q] for s in common]
                o[f"{pr}_{q}"] = {"d": float(np.mean(d)), "ci": boot(d), "wins": int(sum(x > 1e-9 for x in d)), "losses": int(sum(x < -1e-9 for x in d)),
                                  "per_scene": {s: float(x) for s, x in zip(common, d)}}
        out["pairs"][f"{a} - {b}"] = o
    for m in MODELS:
        out["models"][m].pop("_sc")
    json.dump(out, open(os.path.join(AUD, "results", "e_metric_protocols.json"), "w"), indent=1)
    for m, o in out["models"].items():
        print(f"{m:18s} ep {o['episodes']} official AUC20 {o['official_auc20']:.3f} AUC100 {o['official_auc100']:.3f} Final {o['official_final']:.3f} | mono {o['mono_auc20']:.3f} {o['mono_auc100']:.3f} {o['mono_final']:.3f} "
              f"| mono-official Final {o['final_mono_minus_official_mean']:+.3f} | episodes with a decrease {o['episodes_with_official_decrease']} mean steps {o['mean_decreasing_steps']:.1f} max drop {o['max_single_step_drop']:.3f}")
    for k, o in out["pairs"].items():
        print(k, " | ".join(f"{pr} {q} {o[f'{pr}_{q}']['d']:+.3f} [{o[f'{pr}_{q}']['ci'][0]:+.3f},{o[f'{pr}_{q}']['ci'][1]:+.3f}] W{o[f'{pr}_{q}']['wins']}/L{o[f'{pr}_{q}']['losses']}" for pr in ("official", "mono") for q in ("auc20", "auc100", "final")))
    d = out["pairs"]["V3-DV2-KSV - V3-D1-H3-NORISK"]
    print("per scene Final official:", {s: round(v, 3) for s, v in d["official_final"]["per_scene"].items()})
    print("per scene Final mono    :", {s: round(v, 3) for s, v in d["mono_final"]["per_scene"].items()})
    print("mono-official by scene (D-v2):", {s: round(v, 3) for s, v in out["models"]["V3-DV2-KSV"]["final_mono_minus_official_by_scene"].items()})


if __name__ == "__main__":
    main()
