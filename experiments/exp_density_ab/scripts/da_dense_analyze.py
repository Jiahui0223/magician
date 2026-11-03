"""EXP Density A/B -- analysis of the dense-coverage sensitivity check (Macarons++; CPU).
Per config: scene means (mean over the 5 starts of each of the 6 scenes, then over scenes; as the G3 analysis) of AUC / Final
under the primary 10 %-sample monotone coverage (mono) and the dense coverage (all valid pixels). Per pre-registered pair
(scoring D100 - S10, inference D100 - S10, training E2D - E2S): the paired difference under each metric and the
difference-in-differences (dense - mono) with a scene-level bootstrap (10k, seed 0) and wins / losses over scenes.
Output results/dense_cov_summary.json, results/dense_cov_episodes.csv
"""
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
D = os.path.join(REPO, "work_dirs", "exp_density_ab", "dense_cov")
CK, DN = ("B2", "E2S", "E2D"), ("S10", "D100")
CFGS = [f"{m}_{i}_{s}" for m in CK for i in DN for s in DN]


def boot(vals, n=10000, seed=0):
    g = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n, len(g)))].mean(1)
    return {"mean": float(g.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "wins": int(sum(x > 1e-12 for x in g)), "losses": int(sum(x < -1e-12 for x in g)), "per_scene": [float(x) for x in g]}


def main():
    E = {}
    rows = []
    for c in CFGS:
        for f in glob.glob(os.path.join(D, c, "V2M_*.json")):
            r = json.load(open(f))
            assert r["official_equal_stored"] and r["mono_equal_metrics"]
            E.setdefault(c, {}).setdefault(r["scene"], []).append(r)
            rows.append([c, r["scene"], r["start"], r["auc_mono"], r["auc_dense"], r["final_mono"], r["final_dense"],
                         r["n_points_sample_mean"], r["n_points_dense_mean"]])
    n = {c: sum(len(v) for v in E.get(c, {}).values()) for c in CFGS}
    complete = [c for c in CFGS if n[c] == 30]
    scenes = sorted({s for c in complete for s in E[c]})

    def sm(c, k):
        return np.asarray([np.mean([r[k] for r in E[c][s]]) for s in scenes])
    res = {"n_episodes": n, "complete": complete, "scenes": scenes, "configs": {}, "pairs": {}}
    for c in complete:
        res["configs"][c] = {k: float(sm(c, k).mean()) for k in ("auc_mono", "auc_dense", "final_mono", "final_dense")}
        res["configs"][c]["auc_gain_dense"] = boot(sm(c, "auc_dense") - sm(c, "auc_mono"))
        res["configs"][c]["final_gain_dense"] = boot(sm(c, "final_dense") - sm(c, "final_mono"))
    pairs = []
    for m in CK:
        for i in DN:
            pairs.append((f"{m}_{i}_D100", f"{m}_{i}_S10", "scoring D100 - S10"))
        for s in DN:
            pairs.append((f"{m}_D100_{s}", f"{m}_S10_{s}", "inference D100 - S10"))
    for i in DN:
        for s in DN:
            pairs.append((f"E2D_{i}_{s}", f"E2S_{i}_{s}", "training E2D - E2S"))
    for hi, lo, fam in pairs:
        if hi not in complete or lo not in complete:
            continue
        p = {"family": fam}
        for k in ("auc", "final"):
            dm = sm(hi, f"{k}_mono") - sm(lo, f"{k}_mono")
            dd = sm(hi, f"{k}_dense") - sm(lo, f"{k}_dense")
            p[k] = {"mono": boot(dm), "dense": boot(dd), "dense_minus_mono": boot(dd - dm)}
        res["pairs"][f"{hi} - {lo}"] = p
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(EXP, "results", "dense_cov_summary.json"), "w"), indent=1)
    with open(os.path.join(EXP, "results", "dense_cov_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "scene", "start", "auc_mono", "auc_dense", "final_mono", "final_dense", "pts_sample_mean", "pts_dense_mean"])
        w.writerows(sorted(rows))
    print("episodes per config", n)
    for c in complete:
        x = res["configs"][c]
        print(f"{c:14s} AUC mono {x['auc_mono']:.4f} dense {x['auc_dense']:.4f} (+{x['auc_gain_dense']['mean']:.4f}) | "
              f"Final mono {x['final_mono']:.4f} dense {x['final_dense']:.4f} (+{x['final_gain_dense']['mean']:.4f})")
    f_ = lambda b: f"{b['mean']:+.4f} [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}] {b['wins']}/{b['losses']}"  # noqa: E731
    for k, p in res["pairs"].items():
        print(f"{k:30s} AUC mono {f_(p['auc']['mono'])} | dense {f_(p['auc']['dense'])} | DiD {f_(p['auc']['dense_minus_mono'])}")


if __name__ == "__main__":
    main()
