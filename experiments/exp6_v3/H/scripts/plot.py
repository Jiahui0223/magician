"""EXP6 V3 / H -- figures from results/ (CPU; generated). Official coverage; scene-equal means; shaded = H0 scene-level 95% CI.
coverage_{dev,test,all14}_{0_20,0_100}, auc_horizon, final_coverage_by_scene, pairwise_early_auc, runtime_vs_auc (PNG + PDF)."""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(HERE)
FIG = os.path.join(H, "figures")
POL = ("H0", "H1", "H2", "H3", "B0", "B2", "MAG-B")
COL = {"H0": "#333333", "H1": "#2a78d6", "H2": "#1baf7a", "H3": "#d6336c", "B0": "#9e9e9e", "B2": "#c9a227", "MAG-B": "#7b4fd6"}
LAB = {"H0": "V3-T1 / H0 DistP1", "H1": "V3-H1 coverage-aligned", "H2": "V3-H2 risk-aware", "H3": "V3-H3 two-step", "B0": "B0", "B2": "B2", "MAG-B": "MAGICIAN Beam"}
KS = (10, 20, 40, 50, 60, 80, 100)


def save(fig, n):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, n + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, n + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})
    S = json.load(open(os.path.join(H, "results", "summary.json")))["protocols"]["official"]["groups"]
    split = {}
    C = {}
    with open(os.path.join(H, "results", "per_scene.csv")) as f:
        for r in csv.DictReader(f):
            split[r["scene"]] = r["split"]
    with open(os.path.join(H, "results", "coverage_curves.csv")) as f:
        for r in csv.DictReader(f):
            if r["protocol"] == "official":
                C.setdefault(r["policy"], {}).setdefault(r["scene"], {})[int(r["start"])] = np.asarray([float(r[f"c{t}"]) for t in range(101)])
    for g in ("dev", "test", "all14", "train"):
        sc = sorted(s for s in split if g == "all14" or split[s] == g)
        for tmax in (20, 100):
            fig, ax = plt.subplots(figsize=(7.2, 3.6))
            for p in POL:
                d = C.get(p, {})
                if not all(s in d and len(d[s]) == 5 for s in sc):
                    continue
                mat = np.asarray([np.mean([d[s][k] for k in range(5)], 0) for s in sc])[:, :tmax + 1]
                x = np.arange(tmax + 1)
                ax.plot(x, mat.mean(0), color=COL[p], lw=2.0 if p in ("H0", "H3") else 1.4, ls="--" if p in ("B0", "B2", "MAG-B") else "-", label=LAB[p])
                if p == "H0":
                    rng = np.random.default_rng(0)
                    b = mat[rng.integers(0, len(mat), (2000, len(mat)))].mean(1)
                    ax.fill_between(x, np.quantile(b, 0.025, 0), np.quantile(b, 0.975, 0), color=COL[p], alpha=0.12, lw=0, label="H0 scene-level 95% CI")
            ax.set_xlim(0, tmax)
            ax.set_ylim(0, 1)
            ax.set_xlabel("exploration step")
            ax.set_ylabel("coverage (MAGICIAN official)")
            ax.set_title(f"{g} ({len(sc)} scenes x 5 starts; scene-equal mean)", loc="left", fontsize=9)
            ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
            save(fig, f"coverage_{g}_0_{tmax}")
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), sharey=True)
    for ax, g in zip(axs, ("train", "dev", "test", "all14")):
        for p in POL:
            r = S[g][p]
            if r["complete"]:
                ax.plot(KS, [r[f"auc{k}"] for k in KS], marker="o", ms=3, color=COL[p], ls="--" if p in ("B0", "B2", "MAG-B") else "-", label=LAB[p])
        ax.set_xticks(KS)
        ax.set_ylim(0, 1)
        ax.set_title(g, loc="left", fontsize=9)
        ax.set_xlabel("horizon k")
    axs[0].set_ylabel("AUC@k")
    axs[-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, "auc_horizon")
    sc = sorted(split, key=lambda s: (["train", "dev", "test"].index(split[s]), s))
    fig, axs = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True)
    for ax, k, t in zip(axs, ("auc20", "auc100", "final"), ("AUC@20", "AUC@100", "Final")):
        w = 0.8 / 5
        for i, p in enumerate(("H0", "H1", "H2", "H3", "MAG-B")):
            ps = S["all14"][p][f"{k}_per_scene"]
            ax.bar([j + (i - 2) * w for j in range(len(sc))], [ps.get(s, np.nan) for s in sc], width=w * 0.92, color=COL[p], label=LAB[p])
        ax.set_ylim(0, 1)
        ax.set_ylabel(t)
    axs[-1].set_xticks(range(len(sc)))
    axs[-1].set_xticklabels([f"{s}\n({split[s]})" for s in sc], rotation=30, ha="right", fontsize=7)
    axs[0].legend(fontsize=7, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.25))
    save(fig, "final_coverage_by_scene")
    P = json.load(open(os.path.join(H, "results", "summary.json")))["protocols"]["official"]["pairs"]["all14"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, (a, b) in zip(axs, (("H1", "H0"), ("H2", "H1"), ("H3", "H2"))):
        q = next(x for x in P if x["a"] == a and x["b"] == b)
        if q["auc20"]:
            ps = q["auc20"]["per_scene"]
            ax.barh(range(len(sc)), [ps.get(s, np.nan) for s in sc], color=[COL[a] if ps.get(s, 0) > 0 else "#bbbbbb" for s in sc])
            ax.axvline(0, color="#333", lw=0.8)
            ax.set_title(f"ΔAUC@20 {a} − {b}: {q['auc20']['d']:+.3f} [{q['auc20']['ci'][0]:+.3f}, {q['auc20']['ci'][1]:+.3f}]", loc="left", fontsize=8.5)
    axs[0].set_yticks(range(len(sc)))
    axs[0].set_yticklabels([f"{s} ({split[s]})" for s in sc], fontsize=7)
    save(fig, "pairwise_early_auc")
    D = json.load(open(os.path.join(H, "results", "summary.json"))).get("decisions", {})
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, k in zip(axs, ("auc20", "auc100")):
        for p in ("H0", "H1", "H2", "H3"):
            if p in D and S["all14"][p]["complete"]:
                ax.scatter(D[p]["decision_s_median"], S["all14"][p][k], color=COL[p], s=40, label=LAB[p])
                ax.annotate(p, (D[p]["decision_s_median"], S["all14"][p][k]), fontsize=8, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("planner decision time per step, median (s; shared GPUs)")
        ax.set_ylabel(f"All-14 {k.upper().replace('AUC', 'AUC@')}")
    axs[-1].legend(fontsize=7, loc="lower right")
    save(fig, "runtime_vs_auc")
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
