"""Experiment G -- fixed-state summary of D0 / G0 / G1 / G4 (protocol G-v1 section 7; CPU only) from g_fixed.py rows.
Per state and arm (first-layer candidates, labelled UNKNOWN rays, every sample counted):
  hit_acc = (TP + TN) / (K N_lm); precision = TP / (TP + FP); recall = TP / (TP + FN); brier = sum (p - 1[HIT])^2 / N_lm (p = share of the K
  samples that hit); false_wall = predicted hits on labelled NOHIT rays / (K N_NOHIT); depth_mae_tp = mean |z_hat - z*| on true positives
  (runtime z-depth); depth_lt1_tp = share of true positives with |z_hat - z*| < 1.0;
  ranking with the arm's own D1 score: gain_pp of the pick (m_score.choose), top1, Spearman rho (score vs official one-step gain); regret
  only in the json. State means; paired differences with a trajectory-cluster bootstrap (10,000, seed 0, 95 % percentile), explore /
  confirm / pooled. Regression: G4 D1 scores == EXP9 Table 4 V2 T0 rows (bitwise) on every state.
Writes results/fixed/fixed_summary.{json,md,csv}. Usage: python g_analyze_fixed.py [--states DIR] [--out DIR]"""
import argparse
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
ARMS = ["D0", "G0", "G1", "G4"]
MET = ["hit_acc", "precision", "recall", "brier", "false_wall", "depth_mae_tp", "depth_lt1_tp", "gain_pp", "top1", "rho", "regret_pp"]
PAIRS = [("D0", "G1"), ("D0", "G4"), ("G0", "G1"), ("G1", "G4"), ("D0", "G0")]


def state_metrics(r):
    K = r["K"]
    s = {k: 0.0 for k in ("tp", "fp", "fn", "tn", "n_lm", "brier", "fp_nohit", "n_nohit", "sae_tp", "lt1")}
    for c in r["cams"]:
        for k in ("tp", "fp", "fn", "tn", "n_lm", "brier", "sae_tp"):
            s[k] += c[k]
        s["fp_nohit"] += c["fp_nohit"]
        s["n_nohit"] += c["cat_unk"].get("NOHIT", 0)
        s["lt1"] += sum(c["hist_tp"][:3])                   # EDGES 0, .25, .5, 1.0 -> |e| < 1.0
    d = lambda a, b: (a / b) if b > 0 else None  # noqa: E731
    rk = r["rank_d1"]
    return {"hit_acc": d(s["tp"] + s["tn"], K * s["n_lm"]), "precision": d(s["tp"], s["tp"] + s["fp"]), "recall": d(s["tp"], s["tp"] + s["fn"]),
            "brier": d(s["brier"], s["n_lm"]), "false_wall": d(s["fp_nohit"], K * s["n_nohit"]), "depth_mae_tp": d(s["sae_tp"], s["tp"]),
            "depth_lt1_tp": d(s["lt1"], s["tp"]), "gain_pp": rk.get("gain"), "top1": None if rk.get("top1") is None else float(rk["top1"]),
            "rho": rk.get("rho"), "regret_pp": rk.get("regret")}


def cluster_boot(diffs, clusters, n=10000, seed=0):
    """diffs [N] (state-level paired differences), clusters [N] (trajectory tag) -> mean, lo, hi."""
    diffs, clusters = np.asarray(diffs, float), np.asarray(clusters)
    if len(diffs) == 0:
        return None, None, None
    u = sorted(set(clusters))
    idx = {c: np.flatnonzero(clusters == c) for c in u}
    sums = np.array([diffs[idx[c]].sum() for c in u])
    cnts = np.array([len(idx[c]) for c in u])
    rng = np.random.default_rng(seed)
    b = rng.integers(0, len(u), (n, len(u)))
    bm = sums[b].sum(1) / cnts[b].sum(1)
    return float(diffs.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(R, "work_dirs/exp6_v3/experiment_G/fixed/states"))
    ap.add_argument("--out", default=os.path.join(GD, "results/fixed"))
    a = ap.parse_args()
    rows = [json.loads(x) for f in sorted(glob.glob(os.path.join(a.states, "*.jsonl"))) for x in open(f)]
    by = {}
    reg = {"G4_d1_bitwise": 0, "G4_d1_checked": 0, "keys_eq_F": 0, "paths_eq_F": 0, "rows": len(rows)}
    for r in rows:
        by.setdefault((r["set"], r["tag"], r["t"]), {})[r["arm"]] = state_metrics(r)
        reg["keys_eq_F"] += int(r["regression"]["keys_eq_F"])
        reg["paths_eq_F"] += int(r["regression"]["paths_eq_F"])
        if r["arm"] == "G4" and r["regression"].get("d1_bitwise_T4V2_T0") is not None:
            reg["G4_d1_checked"] += 1
            reg["G4_d1_bitwise"] += int(r["regression"]["d1_bitwise_T4V2_T0"])
    states = {k: v for k, v in by.items() if all(x in v for x in ARMS)}
    out = {"n_states": len(states), "n_states_any": len(by), "regression": reg, "sets": {}}
    for sname, keep in (("explore", lambda k: k[0] == "explore"), ("confirm", lambda k: k[0] == "confirm"), ("pooled", lambda k: True)):
        ks = sorted(k for k in states if keep(k))
        res = {"n_states": len(ks), "n_trajectories": len({k[1] for k in ks}), "means": {}, "paired": {}}
        for arm in ARMS:
            res["means"][arm] = {}
            for m in MET:
                v = [states[k][arm][m] for k in ks if states[k][arm][m] is not None]
                res["means"][arm][m] = (float(np.mean(v)) if v else None, len(v))
        for x, y in PAIRS:
            res["paired"][f"{x}-{y}"] = {}
            for m in MET:
                kk = [k for k in ks if states[k][x][m] is not None and states[k][y][m] is not None]
                res["paired"][f"{x}-{y}"][m] = cluster_boot([states[k][x][m] - states[k][y][m] for k in kk], [k[1] for k in kk])
        out["sets"][sname] = res
    os.makedirs(a.out, exist_ok=True)
    json.dump(out, open(os.path.join(a.out, "fixed_summary.json"), "w"), indent=1)
    fm = lambda x: "—" if x is None else f"{x:.4f}"  # noqa: E731
    md = ["# Experiment G — fixed-state summary (protocol G-v1 §7)", "",
          f"States with all four arms: {out['n_states']} (any arm: {out['n_states_any']}); regression G4 D1 == Table 4 T0 bitwise: "
          f"{reg['G4_d1_bitwise']}/{reg['G4_d1_checked']}; candidate keys == F rows {reg['keys_eq_F']}/{reg['rows']}, paths {reg['paths_eq_F']}/{reg['rows']}.", ""]
    csv = []
    for sname, res in out["sets"].items():
        md += [f"## {sname} ({res['n_states']} states, {res['n_trajectories']} trajectories)", "", "| Arm | " + " | ".join(MET[:-1]) + " |", "|---|" + "---|" * (len(MET) - 1)]
        for arm in ARMS:
            md.append(f"| {arm} | " + " | ".join(fm(res["means"][arm][m][0]) for m in MET[:-1]) + " |")
            csv.append([sname, arm, "mean"] + [res["means"][arm][m][0] for m in MET])
        md += ["", "| Paired (cluster bootstrap) | " + " | ".join(MET[:-1]) + " |", "|---|" + "---|" * (len(MET) - 1)]
        for p, v in res["paired"].items():
            md.append(f"| {p} | " + " | ".join("—" if v[m][0] is None else f"{v[m][0]:+.4f} [{v[m][1]:+.4f}, {v[m][2]:+.4f}]" for m in MET[:-1]) + " |")
            csv.append([sname, p, "diff"] + [v[m][0] for m in MET])
        md.append("")
    open(os.path.join(a.out, "fixed_summary.md"), "w").write("\n".join(md) + "\n")
    import csv as _csv
    with open(os.path.join(a.out, "fixed_summary.csv"), "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["set", "arm_or_pair", "kind"] + MET)
        w.writerows(csv)
    print("\n".join(md))


if __name__ == "__main__":
    main()
