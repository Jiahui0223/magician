"""Paper table "Why multiple generative hypotheses?" (2026-11-16, user request). Every number is read from Experiment G / G2 / G3 outputs:
  closed loop  experiment_G.../results/g3/final_table_{outdoor,indoor}.csv  (All-14 / Unseen-6 outdoor, ALL-22 indoor; paired scene /
               level bootstrap vs G4, 10,000, seed 0)
  geometry     experiment_G.../results/g3/fixed_table.csv  (Experiment F 755-state pool, labelled UNKNOWN rays; CRPS / fair ES on the
               UNKNOWN-arc first hit, lower = better; paired trajectory-cluster bootstrap vs G4)
  uncertainty  G2 spread-skill on the same pool (G4 cluster-mean Spearman rho, results/g2/G2_ITEMS12.md)
Rows: D0 (deterministic U-Net, same network / data / planner), G1 (one generated sample), M-raw / M-depth (the 4 generated samples
averaged into one map; M-depth = pre-registered primary averaging variant of G3), Ours (4 samples scored separately, aggregated by D1).
AUC@100 is kept on purpose (pre-registered primary metric of G / G3: parity, not a gain). Mark: bold = best; '†' = difference to Ours
with 95 % CI excluding 0. Outputs fig/table_hypotheses/table_hypotheses.{csv,md,tex,pdf,png}."""
import ast
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
G = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_G_generative_vs_deterministic", "results")
OUT = os.path.join(L.FIG, "table_hypotheses")
ROWS = [("D0", "Deterministic U-Net", "1 (regressed mean)"), ("G1", "Single generated sample", "1"),
        ("M-raw", "Average of 4 samples (b, y)", "4 → 1"), ("M-depth", "Average of 4 samples (depth)", "4 → 1"),
        ("G4", "Ours: 4 hypotheses, aggregated by CDGR", "4")]
SPREAD_SKILL_G4 = 0.938          # results/g2/G2_ITEMS12.md, F-pool, G4 cluster-mean rho [0.926, 0.948]


def rd(p):
    return list(csv.DictReader(open(p)))


def sig(lo_hi):
    lo, hi = ast.literal_eval(lo_hi)
    return lo > 0 or hi < 0


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {(r["group"], r["method"]): r for r in rd(os.path.join(G, "g3", "final_table_outdoor.csv"))}
    ind = {(r["group"], r["method"]): r for r in rd(os.path.join(G, "g3", "final_table_indoor.csv"))}
    fx = {r["arm"]: r for r in rd(os.path.join(G, "g3", "fixed_table.csv"))}
    key = {"D0": "D0", "G1": "G1", "M-raw": "M-raw", "M-depth": "M-depth", "G4": "G4 (Ours)"}
    alias = lambda d, grp, m: d.get((grp, m)) or d.get((grp, {"M-raw": "MRAW", "M-depth": "MDEPTH", "G4": "Ours (G4)"}.get(m, m)))
    cols = [("crps", "CRPS ↓", "fx"), ("fair_es", "Energy score ↓", "fx"), ("unc", "Uncertainty", None),
            ("AUC@100", "AUC@100 (out)", ("All-14", out)), ("Final", "Final (out)", ("All-14", out)),
            ("Final", "Final (out, unseen-6)", ("Unseen-6", out)), ("Final", "Final (in)", ("ALL-22", ind))]
    vals, marks = {}, {}
    for m, _, _ in ROWS:
        for j, (k, _, src) in enumerate(cols):
            if src == "fx":
                r = fx[key[m]]
                v = float(r[k])
                vals[(m, j)] = v
                marks[(m, j)] = m != "G4" and sig(r[f"ci_{k}_vs_G4"])
            elif src is None:
                vals[(m, j)] = SPREAD_SKILL_G4 if m == "G4" else None
                marks[(m, j)] = False
            else:
                grp, d = src
                r = alias(d, grp, m)
                assert r is not None, (grp, m)
                vals[(m, j)] = float(r[k])
                marks[(m, j)] = m != "G4" and sig(r[f"ci_{k}_vs_G4"])
    lower_better = {0, 1}
    body, bold = [], set()
    for i, (m, name, hyp) in enumerate(ROWS):
        row = [name, hyp]
        for j in range(len(cols)):
            v = vals[(m, j)]
            if j == 2:
                row.append(f"ρ = {v:.2f}" if v is not None else "—")
                continue
            row.append(f"{v:.2f}" if j in lower_better else f"{v:.3f}")
            if marks[(m, j)]:
                row[-1] += "†"
        body.append(row)
    for j in range(len(cols)):
        if j == 2:
            bold.add((len(ROWS) - 1, 2 + j))
            continue
        vs = [round(vals[(m, j)], 2 if j in lower_better else 3) for m, _, _ in ROWS]
        best = min(vs) if j in lower_better else max(vs)
        bold |= {(i, 2 + j) for i, v in enumerate(vs) if v == best}
    hdr = ["Prediction / combination", "Hypotheses to planner"] + [c[1] for c in cols]
    L.write_csv(os.path.join(OUT, "table_hypotheses.csv"), hdr, body)
    notes = ["Same V3 network, training data, D1 (CDGR) planner and protocol in every row; only the prediction head (D0) or the way the "
             "K = 4 generated samples are used changes. CRPS / energy score: distributional accuracy of the predicted first-hit surface on "
             "755 planning states (lower is better). Uncertainty: rank correlation between sample spread and actual error. AUC@100 / Final: "
             "closed-loop coverage, Macarons++ 14 scenes x 5 starts (out), 6 scenes no method was trained on (unseen-6), MP3D 22 levels "
             "x 2 starts (in). † = difference to Ours with 95% bootstrap CI excluding 0. Bold = best."]
    RD.table(os.path.join(OUT, "table_hypotheses"), hdr, body, bold=bold, notes=notes,
             title="Why multiple generative hypotheses?", rules_after={len(ROWS) - 2})
    cap = ("Why multiple generative hypotheses. All rows share the network, training data, CDGR planner and evaluation protocol; only the "
           "prediction head or the use of the K = 4 generated samples changes. Representing the future surface by several hypotheses "
           "gives the most accurate predictive distribution (CRPS, energy score), a usable uncertainty estimate, and the highest final "
           "coverage, while keeping the same exploration efficiency (AUC@100). Collapsing the hypotheses into one map (single sample, "
           "deterministic mean, or averaging) loses this. † difference to ours with 95% CI excluding 0.")
    RD.tex(os.path.join(OUT, "table_hypotheses.tex"), hdr, body, bold=bold, caption=cap, label="tab:hypotheses",
           rules_after={len(ROWS) - 2}, colspec="ll" + "c" * len(cols),
           notes=["generated by experiments/cvpr_results/scripts/make_table_hypotheses.py -- do not edit by hand (needs booktabs)"])
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join((f"**{c}**" if (i, j) in bold else c) for j, c in enumerate(r)) + " |" for i, r in enumerate(body)]
    open(os.path.join(OUT, "table_hypotheses.md"), "w").write("\n".join(md) + "\n\n" + notes[0] + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
