"""EXP6 V5 / F1 -- figures from the generated results (CPU). G3 (official coverage, extended Dev-3 group when complete, else
pilot): coverage_curve_0_100, coverage_curve_0_20 (scene-equal mean; shaded = B0 scene-level 95% CI), auc_horizon,
final_coverage_by_scene; G1/G2 (results/g12_summary.json if present): prediction_ranking_diagnostics. PNG (200 dpi) + PDF.
Fixed colours: B0 dark grey dashed, F1-A blue (2k light / 5k dark), F1-B orange (2k light / 5k dark)."""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
FIG = os.path.join(F1, "figures")
MODELS = ("B0", "A2k", "A5k", "B2k", "B5k")
COL = {"B0": "#333333", "A2k": "#8fb8e8", "A5k": "#1f5fbf", "B2k": "#f5b27a", "B5k": "#d9480f"}
LAB = {"B0": "B0 parent", "A2k": "F1-A Continue @2k", "A5k": "F1-A Continue @5k", "B2k": "F1-B Query @2k", "B5k": "F1-B Query @5k"}
SCENES = ("barts", "pantheon", "colosseum")
KS = (10, 20, 40, 50, 60, 80, 100)


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def curves(group_starts):
    C = {}
    with open(os.path.join(F1, "results", "g3_curves.csv")) as f:
        for r in csv.DictReader(f):
            if r["protocol"] == "official" and int(r["start"]) in group_starts:
                C.setdefault(r["model"], {}).setdefault(r["scene"], {})[int(r["start"])] = np.asarray([float(r[f"c{t}"]) for t in range(101)])
    out = {}
    for m, d in C.items():
        if all(s in d and all(k in d[s] for k in group_starts) for s in SCENES):
            out[m] = np.asarray([np.mean([d[s][k] for k in group_starts], 0) for s in SCENES])     # [scenes, 101]
    return out


def boot_band(mat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = mat[rng.integers(0, len(mat), (n, len(mat)))].mean(1)
    return np.quantile(b, 0.025, 0), np.quantile(b, 0.975, 0)


def fig_curves(S, tmax, name, gname, n_ep):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for m in MODELS:
        if m not in S:
            continue
        x = np.arange(tmax + 1)
        mu = S[m][:, :tmax + 1].mean(0)
        ax.plot(x, mu, color=COL[m], lw=2.0 if m == "B0" else 1.6, ls="--" if m == "B0" else "-", label=LAB[m])
        if m == "B0":
            lo, hi = boot_band(S[m][:, :tmax + 1])
            ax.fill_between(x, lo, hi, color=COL[m], alpha=0.12, lw=0, label="B0 scene-level 95% CI")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("exploration step")
    ax.set_ylabel("coverage (MAGICIAN official)")
    ax.set_title(f"Dev-3 {gname} ({n_ep} episodes / model; scene-equal mean)", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, name)


def main():
    style()
    summ = json.load(open(os.path.join(F1, "results", "summary.json")))["official"]["groups"]
    g = "extended" if all(summ["extended"]["models"][m]["complete"] for m in MODELS) else "pilot"
    starts = (0, 1, 2, 3, 4) if g == "extended" else (0, 1)
    S = curves(starts)
    n_ep = 3 * len(starts)
    fig_curves(S, 100, "coverage_curve_0_100", g, n_ep)
    fig_curves(S, 20, "coverage_curve_0_20", g, n_ep)
    # AUC@k vs horizon
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for m in MODELS:
        r = summ[g]["models"][m]
        if not r["complete"]:
            continue
        ax.plot(KS, [r[f"auc{k}"] for k in KS], marker="o", ms=3.5, color=COL[m], ls="--" if m == "B0" else "-", label=LAB[m])
    ax.set_xticks(KS)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("horizon k (steps)")
    ax.set_ylabel("AUC@k")
    ax.set_title(f"Dev-3 {g}: AUC@k (official coverage)", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "auc_horizon")
    # per-scene AUC@20 / AUC@100 / Final
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, k, t in zip(axs, ("auc20", "auc100", "final"), ("AUC@20", "AUC@100", "Final coverage")):
        w = 0.16
        for i, m in enumerate(MODELS):
            ps = summ[g]["models"][m][f"{k}_per_scene"]
            xs = [j + (i - 2) * w for j, s in enumerate(SCENES) if s in ps]
            ax.bar(xs, [ps[s] for s in SCENES if s in ps], width=w * 0.92, color=COL[m], label=LAB[m])
        ax.set_xticks(range(len(SCENES)))
        ax.set_xticklabels(SCENES)
        ax.set_ylim(0, 1.0)
        ax.set_title(t, loc="left", fontsize=9)
    axs[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "final_coverage_by_scene")
    # G1 / G2 diagnostics
    gp = os.path.join(F1, "results", "g12_summary.json")
    if os.path.exists(gp):
        G = json.load(open(gp))["pooled"]
        panels = [("mae_unknown", "unknown-ray depth MAE (lower better)"), ("hit_f1_unk", "hit F1 on UNKNOWN rays"), ("brier_unk", "No-Hit Brier (UNKNOWN; lower better)"),
                  ("gain", "true one-step gain of the pick (pp)"), ("top1", "Top-1 (oracle)")]
        fig, axs = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.0))
        for ax, (k, t) in zip(axs, panels):
            for j, ph in enumerate(("early", "late", "all")):
                vals = [G[ph][m].get(k) if m in G[ph] else None for m in MODELS]
                ax.bar([j + (i - 2) * 0.16 for i in range(len(MODELS))], [v if v is not None else 0 for v in vals], width=0.15, color=[COL[m] for m in MODELS])
            ax.set_xticks(range(3))
            ax.set_xticklabels(["t≤20", "t>20", "all"])
            ax.set_title(t, loc="left", fontsize=8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COL[m]) for m in MODELS]
        axs[-1].legend(handles, [LAB[m] for m in MODELS], fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        save(fig, "prediction_ranking_diagnostics")
    print("figures written to", FIG, "group", g)


if __name__ == "__main__":
    main()
