"""Experiment XB / Task D -- uncertainty decomposition from EXISTING closed-loop records (CPU, no new runs).
Sources of variation in the All-14 numbers: (1) scene-to-scene, (2) start-to-start within a scene (same model), (3) model-to-model among the seven near-equivalent
models (T1 + six XA variants: same data, recipe, seed; validation within 1.5 % except XA-MR) -- the spread of their All-14 means is an UPPER bound on the retraining
(seed) variance plus any architecture effect, since no seed replicate exists. Also: how large a per-model difference the scene-bootstrap CI can resolve, and
how many T1 seeds would be needed to detect 0.005 / 0.010 / 0.020 differences. Output results/D_uncertainty.json + printed summary."""
import csv
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
XA = f"{R}/experiments/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene"
TB = f"{R}/work_dirs/cvpr_results/table1/episodes"
OUT = f"{R}/experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/results"
IDS = ["xa_s2d", "xa_sr", "xa_si", "xa_m2d", "xa_mr", "xa_mi"]


def om(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "cov20": float(c[20]), "auc100": float(np.trapz(c[:101]) / 100), "final": float(c[100])}


E = {}   # model -> {(scene, start): metrics}
for f in glob.glob(f"{TB}/V3-D1-H3-NORISK/*.json"):
    r = json.load(open(f)); E.setdefault("T1_D1", {})[(r["scene"], int(r["start"]))] = om(r["coverage_official"])
for vid in IDS:
    for row in csv.DictReader(open(f"{XA}/results/{vid}/episodes.csv")):
        if row["domain"] == "outdoor":
            E.setdefault(vid, {})[(row["scene"], int(row["start"]))] = {k: float(row[k]) for k in ("auc20", "cov20", "auc100", "final")}
models = ["T1_D1"] + IDS
scenes = sorted({k[0] for k in E["T1_D1"]})
out = {"n_models": len(models), "metrics": {}}
for m in ("auc20", "cov20", "auc100", "final"):
    per_scene = {md: {s: np.mean([E[md][(s, k)][m] for k in range(5)]) for s in scenes} for md in models}
    all14 = {md: float(np.mean(list(per_scene[md].values()))) for md in models}
    # (1) scene-to-scene sd of the T1 scene means; (2) within-scene start sd (pooled over scenes, T1); (3) model-to-model sd of All-14 means (7 models)
    sd_scene = float(np.std(list(per_scene["T1_D1"].values()), ddof=1))
    sd_start = float(np.sqrt(np.mean([np.var([E["T1_D1"][(s, k)][m] for k in range(5)], ddof=1) for s in scenes])))
    sd_model = float(np.std(list(all14.values()), ddof=1))
    rng = np.random.default_rng(0)
    # SE of a paired All-14 difference under scene bootstrap, for a typical model (xa_sr) vs T1
    d = np.array([per_scene["xa_sr"][s] - per_scene["T1_D1"][s] for s in scenes])
    b = d[rng.integers(0, 14, (10000, 14))].mean(1)
    se_pair = float(b.std())
    # start-level paired differences (70 pairs): how much of a model difference is start noise
    d70 = np.array([E["xa_sr"][(s, k)][m] - E["T1_D1"][(s, k)][m] for s in scenes for k in range(5)])
    # seeds needed: with per-seed All-14 sd = sd_model (upper bound), the SE of a mean of n seeds = sd_model / sqrt(n); to resolve delta at 2 SE (two-model comparison -> sqrt(2))
    need = {str(delta): int(np.ceil((2 * np.sqrt(2) * sd_model / delta) ** 2)) for delta in (0.005, 0.01, 0.02)}
    out["metrics"][m] = {"all14_by_model": all14, "sd_scene_T1": sd_scene, "sd_start_within_scene_T1": sd_start, "sd_all14_across_7_models": sd_model,
                         "range_all14_across_7_models": [float(min(all14.values())), float(max(all14.values()))], "se_paired_scene_bootstrap_xa_sr_vs_T1": se_pair,
                         "sd_start_level_paired_diff_xa_sr_vs_T1": float(d70.std(ddof=1)), "mean_paired_diff_xa_sr_vs_T1": float(d70.mean()),
                         "seeds_needed_to_resolve_delta_at_2SE_upper_bound": need}
    print(f"{m}: All-14 range over 7 models [{min(all14.values()):.3f}, {max(all14.values()):.3f}] sd {sd_model:.4f} | scene sd (T1) {sd_scene:.3f} | within-scene start sd (T1) {sd_start:.3f} | "
          f"paired scene-bootstrap SE {se_pair:.4f} | start-level paired diff sd {d70.std(ddof=1):.3f} | seeds to resolve 0.005/0.01/0.02: {need}")
# colosseum-type bimodality: per start spread of Final, all models
col = {md: [E[md][("colosseum", k)]["final"] for k in range(5)] for md in models}
out["colosseum_final_per_start"] = col
print("colosseum Final per start:", {md: [round(x, 3) for x in v] for md, v in col.items()})
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(f"{OUT}/D_uncertainty.json", "w"), indent=1)
