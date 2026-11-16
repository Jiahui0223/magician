"""Table 5 v2 -- "Why multiple generative hypotheses?" on Unseen-6 (2026-11-16, user request; v1 = fig/table5/table5_v1*).
Rows (same V3-T0 network family, training data, CDGR planner, protocol; only the prediction head or the use of the K = 4 samples changes):
  Deterministic   D0, deterministic U-Net head (one regression pass), experiment_G closed loop
  Single sample   K = 1 + CDGR. ALIGNMENT FIX vs v1: mean over the four single-map runs on noise maps 0, 1, 2, 3 (V3-T0-K1D2, -N1, -N2,
                  -N3), i.e. the maps the 4-hypothesis rows use -- the same rule as Table 3 v4 (v1 used map 0 only)
  Mean (b, y)     M-raw: the 4 samples (maps 0-3, identical to Ours' samples) averaged in (b, y), decoded once
  Mean depth      M-depth: the 4 samples decoded, >= 2/4 vote, mean hit depth
  Seen2Map        Ours: the 4 samples scored separately and aggregated by CDGR (= Table 1 v4 / Table 3 v4 K = 4 + CDGR cell)
Columns
  CRPS / Energy score  predictive-distribution accuracy of the first-hit surface on labelled UNKNOWN rays (Experiment G3 fixed-state pool,
                       restricted to the states of unseen scenes: barts, eiffel, sestino_museum = 3 of Unseen-6, 211 states, 16
                       trajectories); lower is better; trajectory-cluster bootstrap (10,000, seed 0). Full-pool values are written too.
  AUC@100              closed-loop efficiency, trapz(C[0..100]) / 100
  AUC@50-100           late-phase coverage, trapz(C[50..100]) / 50 (mean coverage over the second half of the budget)
  Final                C[100]
Closed loop: Macarons++ Unseen-6 x 5 official starts, official coverage; scene = mean of the 5 starts, scene-equal mean; paired scene
bootstrap vs Ours (10,000, seed 0). Non-linear metrics of the single-sample row are computed per noise map, then averaged.
'†' = difference to Ours with 95 % CI excluding 0; bold = best (after rounding).
Also writes table5_v2_metric_panel.csv: every metric computed in the audit (AUC@10..100, Cov@10..100, late AUCs, steps-to-threshold,
success rates, worst start), all rows, paired CIs -- the full record, so the table columns can be checked against the rest.
Cross-checks (assert): D0 / M-raw / M-depth / Ours AUC@100 and Final == G3 final_table_outdoor.csv (Unseen-6); single sample and Ours
== Table 3 v4 CDGR K = 1 / K = 4 cells; full-pool CRPS / ES == G3 fixed_table.csv.
Outputs fig/table5/table5_v2.{csv,md,tex,pdf,png}, table5_v2_metric_panel.csv. Usage: python make_table5_v2.py"""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_G_generative_vs_deterministic")
sys.path.insert(0, os.path.join(GD, "scripts"))
import g3_collect as G3  # noqa: E402

OUT = os.path.join(L.FIG, "table5")
U6 = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
assert len(U6) == 6 and set(U6) <= set(L.SCENE_ORDER)
ROWS = [("D0", "Deterministic"), ("G1", "Single sample"), ("MRAW", "Mean (b, y)"), ("MDEPTH", "Mean depth"), ("G4", "Seen2Map (Ours)")]
G1_REPS = ["V3-T0-K1D2", "V3-T0-K1D2-N1", "V3-T0-K1D2-N2", "V3-T0-K1D2-N3"]
FIXED_STATES = os.path.join(R, "work_dirs", "exp6_v3", "experiment_G", "g3", "fixed")
NB = 10000


def episodes():
    """arm -> array [scene (U6 order), start, 101]; G1 -> list of 4 such arrays (noise maps 0..3)."""
    E = {"D0": G3.load_dir(G3.WG, "D0")[0], "MRAW": G3.load_dir(os.path.join(G3.WG, "g3"), "MRAW")[0],
         "MDEPTH": G3.load_dir(os.path.join(G3.WG, "g3"), "MDEPTH")[0], "G4": L.load_manifest("V3-T0-D1")}
    arr = lambda e: np.stack([np.stack([np.asarray(e[(s, k)]["coverage_official"], float) for k in range(L.N_START)]) for s in U6])  # noqa: E731
    ep = {a: arr(E[a]) for a in E}
    ep["G1"] = [arr(L.load_manifest(m)) for m in G1_REPS]
    return ep


def auc(c, t0, t1):
    return np.trapz(c[..., t0:t1 + 1], axis=-1) / (t1 - t0)


def first(c, thr):
    hit = c >= thr
    return np.where(hit.any(-1), hit.argmax(-1), 101).astype(float)


# name -> (per-episode function [.., 101] -> [..], lower_is_better); scene value = mean over starts unless the function reduces starts
PANEL = {}
for _t in range(10, 101, 10):
    PANEL[f"AUC@{_t}"] = ((lambda t: lambda c: auc(c, 0, t).mean(-1))(_t), False)
for _t in range(10, 101, 10):
    PANEL[f"Cov@{_t}"] = ((lambda t: lambda c: c[..., t].mean(-1))(_t), False)
PANEL["AUC@50-100"] = (lambda c: auc(c, 50, 100).mean(-1), False)
PANEL["AUC@60-100"] = (lambda c: auc(c, 60, 100).mean(-1), False)
PANEL["Gain C[100]-C[50]"] = (lambda c: (c[..., 100] - c[..., 50]).mean(-1), False)
for _thr in (0.80, 0.85, 0.90):
    PANEL[f"Steps to {_thr:.2f}"] = ((lambda thr: lambda c: first(c, thr).mean(-1))(_thr), True)
PANEL["P(Final>=0.90)"] = (lambda c: (c[..., 100] >= 0.90).mean(-1), False)
PANEL["P(Final>=0.95)"] = (lambda c: (c[..., 100] >= 0.95).mean(-1), False)
PANEL["Worst-start Final"] = (lambda c: c[..., 100].min(-1), False)
PANEL["Std over starts of Final"] = (lambda c: c[..., 100].std(-1), True)
PANEL["Worst-start AUC@100"] = (lambda c: auc(c, 0, 100).min(-1), False)


def closed_loop(ep):
    """metric -> arm -> per-scene values (G1: metric per noise map, then mean)."""
    out = {}
    for m, (f, _) in PANEL.items():
        out[m] = {a: (np.mean([f(x) for x in ep[a]], 0) if a == "G1" else f(ep[a])) for a, _ in ROWS}
    return out


def paired_scene(x, y):
    d = np.asarray(x) - np.asarray(y)
    b = d[np.random.default_rng(0).integers(0, len(d), (NB, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def fixed(select):
    """CRPS / fair ES from the G3 fixed-state rows (same accumulators and cluster bootstrap as g3_analyze_fixed.py)."""
    arms = [a for a, _ in ROWS]
    rows = [json.loads(x) for f in sorted(glob.glob(os.path.join(FIXED_STATES, "*.jsonl"))) for x in open(f)]
    by = {}
    for r in rows:
        by.setdefault((r["tag"], r["t"]), {})[r["arm"]] = r
    keys = sorted(k for k, v in by.items() if all(a in v for a in arms) and select(k[0]))
    tags = sorted({k[0] for k in keys})
    ci_ = np.asarray([tags.index(k[0]) for k in keys])
    C = len(tags)
    idx = np.random.default_rng(0).integers(0, C, (NB, C))
    W = np.stack([np.bincount(i, minlength=C) for i in idx]).astype(float)
    res = {"states": len(keys), "trajectories": C, "tags": tags}
    for m, (nk, dk) in {"CRPS": ("crps", "n_lm"), "Energy score": ("es", "n_views")}.items():
        val, bo = {}, {}
        for a in arms:
            num = np.bincount(ci_, weights=np.asarray([by[k][a]["acc"][nk] for k in keys]), minlength=C)
            den = np.bincount(ci_, weights=np.asarray([by[k][a]["acc"][dk] for k in keys]), minlength=C)
            val[a], bo[a] = float(num.sum() / den.sum()), (W @ num) / (W @ den)
        res[m] = {a: (val[a],) + ((val[a] - val["G4"], float(np.percentile(bo[a] - bo["G4"], 2.5)),
                                   float(np.percentile(bo[a] - bo["G4"], 97.5))) if a != "G4" else (0.0, 0.0, 0.0)) for a in arms}
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    ep = episodes()
    cl = closed_loop(ep)
    unseen_tag = lambda t: t.rsplit("_s", 1)[0].endswith(tuple("_" + s for s in U6))  # tags like conf_eiffel_s0, g2_B0_barts_s3  # noqa: E731
    fx_u, fx_all = fixed(unseen_tag), fixed(lambda t: True)
    # ---------------------------------------------------------------- cross-checks
    ft = {r["method"]: r for r in csv.DictReader(open(os.path.join(G3.OUT, "final_table_outdoor.csv"))) if r["group"] == "Unseen-6"}
    for a, nm in (("D0", "D0"), ("MRAW", "M-raw"), ("MDEPTH", "M-depth"), ("G4", "Ours (G4)")):
        for m in ("AUC@100", "Final"):
            v = cl[m if m != "Final" else "Cov@100"][a].mean()
            assert abs(v - float(ft[nm][m])) < 1e-9, (a, m, v, ft[nm][m])
    t3 = {(r["planner"], r["K"]): r for r in csv.DictReader(open(os.path.join(L.FIG, "table3", "table3_hypothesis_planning_v4.csv")))}
    for a, k in (("G1", "1"), ("G4", "4")):
        for m, col in (("AUC@100", "AUC@100"), ("Cov@100", "Final"), ("AUC@20", "AUC@20")):
            assert abs(cl[m][a].mean() - float(t3[("CDGR", k)][col])) < 6e-5, (a, m, cl[m][a].mean(), t3[("CDGR", k)][col])
    fxt = {r["arm"]: r for r in csv.DictReader(open(os.path.join(G3.OUT, "fixed_table.csv")))}
    lab = {"D0": "D0", "G1": "G1", "MRAW": "M-raw", "MDEPTH": "M-depth", "G4": "G4 (Ours)"}
    for a, _ in ROWS:
        assert abs(fx_all["CRPS"][a][0] - float(fxt[lab[a]]["crps"])) < 1e-9 and abs(fx_all["Energy score"][a][0] - float(fxt[lab[a]]["fair_es"])) < 1e-9
    assert fx_all["states"] == 755 and set(s for s in U6 if any(s in t for t in fx_u["tags"])) == {"barts", "eiffel", "sestino_museum"}
    for a in ("D0", "MRAW", "MDEPTH", "G4"):
        assert ep[a].shape == (6, 5, 101)
    assert all(x.shape == (6, 5, 101) for x in ep["G1"])
    # ---------------------------------------------------------------- table
    cols = [("CRPS", "CRPS ↓", "fx", 2), ("Energy score", "Energy score ↓", "fx", 2), ("AUC@100", "AUC@100 ↑", "cl", 3),
            ("AUC@50-100", "AUC@50–100 ↑", "cl", 3), ("Cov@100", "Final ↑", "cl", 3)]
    val, mark = {}, {}
    for a, _ in ROWS:
        for c, _, src, nd in cols:
            if src == "fx":
                v, d, lo, hi = fx_u[c][a]
            else:
                v = float(cl[c][a].mean())
                d, lo, hi = paired_scene(cl[c][a], cl[c]["G4"]) if a != "G4" else (0.0, 0.0, 0.0)
            val[(a, c)], mark[(a, c)] = v, a != "G4" and (lo > 0 or hi < 0)
    body, bold = [], set()
    for i, (a, name) in enumerate(ROWS):
        body.append([name] + [f"{val[(a, c)]:.{nd}f}" + ("†" if mark[(a, c)] else "") for c, _, _, nd in cols])
    for j, (c, _, src, nd) in enumerate(cols):
        vs = [round(val[(a, c)], nd) for a, _ in ROWS]
        best = min(vs) if src == "fx" else max(vs)
        bold |= {(i, 1 + j) for i, v in enumerate(vs) if v == best}
    hdr = ["Prediction strategy"] + [h for _, h, _, _ in cols]
    L.write_csv(os.path.join(OUT, "table5_v2.csv"), ["Prediction strategy"] + [c for c, _, _, _ in cols], body)
    note = ("Macarons++ Unseen-6 (6 scenes no method was trained on) x 5 official starts, 100 steps, official coverage; same network family, "
            "training data, CDGR planner and protocol in every row. Single sample: mean over noise maps 0-3 (the maps of the 4-hypothesis rows). "
            "CRPS / energy score: accuracy of the predicted distribution of the future first-hit surface on held-out planning states of "
            "unseen scenes (lower is better). AUC@50-100: mean coverage over steps 50-100. † = difference to Seen2Map with 95% paired "
            "bootstrap CI excluding 0. Bold = best.")
    RD.table(os.path.join(OUT, "table5_v2"), hdr, body, bold=bold, notes=[note], rules_after={len(ROWS) - 2})
    cap = ("Why multiple generative hypotheses (unseen scenes). All rows share the network family, training data, CDGR planner and "
           "evaluation protocol; only the prediction head or the use of the K = 4 generated samples changes. Keeping the hypotheses separate "
           "gives the most accurate predictive distribution of the unseen surface (CRPS, energy score) and the highest late-phase and final "
           "coverage, at the same overall efficiency (AUC@100). " + note)
    RD.tex(os.path.join(OUT, "table5_v2.tex"), hdr, body, bold=bold, caption=cap, label="tab:table5", rules_after={len(ROWS) - 2},
           colspec="l" + "c" * len(cols), notes=["generated by experiments/cvpr_results/scripts/make_table5_v2.py -- do not edit by hand (needs booktabs)"])
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join((f"**{x}**" if (i, j) in bold else x) for j, x in enumerate(r)) + " |" for i, r in enumerate(body)]
    open(os.path.join(OUT, "table5_v2.md"), "w").write("\n".join(md) + "\n\n" + note + "\n")
    # ---------------------------------------------------------------- full metric panel (record)
    prow = []
    for m, (_, lower) in PANEL.items():
        for a, name in ROWS:
            d, lo, hi = paired_scene(cl[m][a], cl[m]["G4"]) if a != "G4" else (0.0, 0.0, 0.0)
            prow.append(["closed loop, Unseen-6", m, "lower" if lower else "higher", name, f"{cl[m][a].mean():.5f}",
                         "" if a == "G4" else f"{d:+.5f}", "" if a == "G4" else f"{lo:+.5f}", "" if a == "G4" else f"{hi:+.5f}"])
    for grp, fx in (("fixed states, unseen scenes (3 of Unseen-6)", fx_u), ("fixed states, full 755-state pool", fx_all)):
        for m in ("CRPS", "Energy score"):
            for a, name in ROWS:
                v, d, lo, hi = fx[m][a]
                prow.append([grp, m, "lower", name, f"{v:.5f}", "" if a == "G4" else f"{d:+.5f}", "" if a == "G4" else f"{lo:+.5f}",
                             "" if a == "G4" else f"{hi:+.5f}"])
    L.write_csv(os.path.join(OUT, "table5_v2_metric_panel.csv"),
                ["group", "metric", "better", "row", "value", "diff_vs_Seen2Map", "ci95_lo", "ci95_hi"], prow)
    print("\n".join(md))
    print(f"\nfixed states (unseen): {fx_u['states']} states / {fx_u['trajectories']} trajectories; full pool {fx_all['states']} / "
          f"{fx_all['trajectories']}")


if __name__ == "__main__":
    main()
