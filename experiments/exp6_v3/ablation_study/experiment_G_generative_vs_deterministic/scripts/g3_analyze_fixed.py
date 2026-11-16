"""Experiment G3 -- fixed-state table (protocol G3-v1 section 5; CPU) from g3_fixed.py rows (work_dirs/exp6_v3/experiment_G/g3/fixed/*.jsonl).
Per arm (G4, G1, D0, M-raw, M-depth): ranking with the arm's own D1 score (selected-candidate official one-step gain pp, Top-1, within-state
Spearman; state means), point-forecast quality on labelled UNKNOWN rays (fair CRPS, fair ES per view, hit_ratio, Brier, precision, recall, TP
depth error and share <= 1.0 runtime), floating-surface rates over predicted UNKNOWN hits (S, T, S & T; S only for D0 / M-raw / M-depth).
Statistics: trajectory-cluster bootstrap (10,000, seed 0, 95 % percentile); ratio metrics = ratio of cluster sums; paired differences vs G4.
Writes results/g3/fixed_table.{md,csv,json}. Usage: python g3_analyze_fixed.py [--states DIR] [--out DIR]"""
import argparse
import glob
import json
import os

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
ARMS = ["D0", "G1", "MRAW", "MDEPTH", "G4"]
LABEL = {"D0": "D0", "G1": "G1", "MRAW": "M-raw", "MDEPTH": "M-depth", "G4": "G4 (Ours)"}
# name -> (numerator key, denominator key) in the per-state accumulators, or ("rank", field)
METRICS = {"gain_pp": ("rank", "gain"), "top1": ("rank", "top1"), "rho": ("rank", "rho"),
           "crps": ("crps", "n_lm"), "fair_es": ("es", "n_views"), "hit_ratio": ("p", "n_lab"), "brier": ("brier", "n_lm"),
           "precision": ("tp", "hit"), "recall": ("tp", "n_lab"), "tp_err_rt": ("tp_err", "tp_n"), "tp_le_1": ("tp_le_tau", "tp_n"),
           "float_S": ("S", "pred"), "float_T": ("T", "pred"), "float_ST": ("ST", "pred")}
NB = 10000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default=os.path.join(R, "work_dirs/exp6_v3/experiment_G/g3/fixed"))
    ap.add_argument("--out", default=os.path.join(GD, "results/g3"))
    a = ap.parse_args()
    rows = [json.loads(x) for f in sorted(glob.glob(os.path.join(a.states, "*.jsonl"))) for x in open(f)]
    by = {}
    for r in rows:
        by.setdefault((r["tag"], r["t"]), {})[r["arm"]] = r
    keys = sorted(k for k, v in by.items() if all(x in v for x in ARMS))
    tags = sorted({k[0] for k in keys})
    ci_ = np.asarray([tags.index(k[0]) for k in keys])
    C = len(tags)
    rng = np.random.default_rng(0)
    W = np.zeros((NB, C))
    idx = rng.integers(0, C, (NB, C))
    for b in range(NB):
        W[b] = np.bincount(idx[b], minlength=C)
    reg = {"states": len(keys), "trajectories": C, "G4_d1_bitwise": sum(bool(by[k]["G4"]["regression"].get("d1_bitwise_T4V2_T0")) for k in keys),
           "G4_d1_checked": sum(by[k]["G4"]["regression"].get("d1_bitwise_T4V2_T0") is not None for k in keys),
           "M_samples_equal_G4": sum(bool(by[k][m]["regression"].get("samples_equal_G4")) for k in keys for m in ("MRAW", "MDEPTH"))}
    val, boots = {}, {}
    for arm in ARMS:
        val[arm] = {}
        for m, (nk, dk) in METRICS.items():
            if nk == "rank":
                x = np.asarray([np.nan if by[k][arm]["rank_d1"].get(dk) is None else float(by[k][arm]["rank_d1"][dk]) for k in keys])
                ok = np.isfinite(x)
                num = np.bincount(ci_[ok], weights=x[ok], minlength=C)
                den = np.bincount(ci_[ok], minlength=C).astype(float)
            else:
                if m in ("float_S", "float_ST") and arm in ("G4", "G1"):
                    continue
                num = np.bincount(ci_, weights=np.asarray([by[k][arm]["acc"][nk] for k in keys]), minlength=C)
                den = np.bincount(ci_, weights=np.asarray([by[k][arm]["acc"][dk] for k in keys]), minlength=C)
            with np.errstate(invalid="ignore", divide="ignore"):
                val[arm][m] = float(num.sum() / den.sum())
                boots[(arm, m)] = (W @ num) / (W @ den)
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))  # noqa: E731
    paired = {arm: {m: {"diff": val[arm][m] - val["G4"][m], "ci": ci(boots[(arm, m)] - boots[("G4", m)])} for m in METRICS
                    if (arm, m) in boots and ("G4", m) in boots} for arm in ARMS if arm != "G4"}
    vci = {arm: {m: ci(boots[(arm, m)]) for m in METRICS if (arm, m) in boots} for arm in ARMS}
    os.makedirs(a.out, exist_ok=True)
    json.dump({"regression": reg, "values": val, "ci": vci, "paired_vs_G4": paired}, open(os.path.join(a.out, "fixed_table.json"), "w"), indent=1)
    import csv
    with open(os.path.join(a.out, "fixed_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm"] + list(METRICS) + [f"d_{m}_vs_G4" for m in METRICS] + [f"ci_{m}_vs_G4" for m in METRICS])
        for arm in ARMS:
            w.writerow([LABEL[arm]] + [val[arm].get(m) for m in METRICS] + [paired.get(arm, {}).get(m, {}).get("diff") for m in METRICS]
                       + [paired.get(arm, {}).get(m, {}).get("ci") for m in METRICS])
    fm = lambda x, nd=3: "—" if x is None else f"{x:.{nd}f}"  # noqa: E731

    def cell(arm, m, nd=3):
        if m not in val[arm]:
            return "—"
        if arm == "G4" or m not in paired.get(arm, {}):
            return fm(val[arm][m], nd)
        p = paired[arm][m]
        return f"{fm(val[arm][m], nd)} ({p['diff']:+.{nd}f} [{p['ci'][0]:+.{nd}f}, {p['ci'][1]:+.{nd}f}])"
    md = ["# G3 fixed states (755-state F-pool; protocol G3-v1 §5)", "",
          f"States with all five arms: {reg['states']} ({reg['trajectories']} trajectories). Regression: G4 D1 == Table 4 T0 bitwise {reg['G4_d1_bitwise']}/{reg['G4_d1_checked']}; "
          f"M-arm samples == G4 samples {reg['M_samples_equal_G4']}/{2 * reg['states']}. In brackets: paired difference vs G4, trajectory-cluster bootstrap (10,000, seed 0).", "",
          "## Ranking (each arm's own D1 score)", "", "| Arm | gain of pick (pp) | Top-1 | within-state ρ |", "|---|---|---|---|"]
    for arm in ARMS:
        md.append(f"| {LABEL[arm]} | {cell(arm, 'gain_pp')} | {cell(arm, 'top1')} | {cell(arm, 'rho')} |")
    md += ["", "## Point-forecast quality (labelled UNKNOWN rays; l in UNKNOWN-arc voxels)", "",
           "| Arm | CRPS | fair ES | hit_ratio | Brier | precision | recall | TP depth error (runtime) | TP within 1.0 |", "|---|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        md.append(f"| {LABEL[arm]} | " + " | ".join(cell(arm, m) for m in ("crps", "fair_es", "hit_ratio", "brier", "precision", "recall", "tp_err_rt", "tp_le_1")) + " |")
    md += ["", "## Floating surfaces (all predicted UNKNOWN hits; τ = 1.0 runtime)", "",
           "| Arm | S: far from every G4 sample | T: far from GT first hit | S ∧ T (floating) |", "|---|---|---|---|"]
    for arm in ARMS:
        md.append(f"| {LABEL[arm]} | {cell(arm, 'float_S', 4)} | {cell(arm, 'float_T', 4)} | {cell(arm, 'float_ST', 4)} |")
    md += ["", "S is defined relative to the G4 samples, so it is 0 by construction for G4 / G1 (not reported); T for G4 = per-sample rate."]
    open(os.path.join(a.out, "fixed_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
