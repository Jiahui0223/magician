"""CVPR Table 1 -- paper figures from the generated results (CPU; primary protocol = MAGICIAN official coverage).
A coverage vs step (All-15 and per split; scene-equal mean, shaded scene-level bootstrap 95% CI)
B early coverage, steps 0-20
C AUC@k vs horizon k in {10, 20, 40, 50, 60, 80, 100}
D per-scene AUC@20 / AUC@100 / Final (grouped by split)
E paired differences vs MAG-B (dAUC@20, dAUC@100, dFinal; scene-level CI)
Every figure as PNG (200 dpi) and PDF. Fixed model colours / order from MODEL_REGISTRY.yaml; y axes start at 0 for
coverage / AUC levels (no truncation). Usage: python plot.py
"""
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import aggregate as AG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = AG.T1
FIG = os.path.join(T1, "figures")
PALETTE = ["#333333", "#2a78d6", "#1baf7a", "#eb6834", "#a855c7", "#c9a227", "#0f9fb5", "#d6336c",
           "#8a8a8a", "#e8a2c0", "#7fd3a8", "#9b7fe6", "#7ab8f5"]       # main 8 first, then optional (lighter)
GN = {"train": "Train", "dev": "Dev", "test": "Test", "other": "Other / Unassigned", "all14": "All-14", "all15": "All-15"}


def style():
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                         "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def load():
    split, order, label, _ = AG.load_registries()
    curves = {}
    with open(os.path.join(T1, "results", "coverage_curves.csv")) as f:
        for r in csv.DictReader(f):
            curves.setdefault((r["model"], r["scene"]), {})[int(r["start"])] = np.asarray([float(r[f"c{t}"]) for t in range(AG.T_MAX + 1)])
    S = json.load(open(os.path.join(T1, "results", "summary.json")))
    present = S["models"]
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(order)}
    role = {m["id"]: m.get("role", "main") for m in AG.load_registries()[3]}
    return split, order, label, curves, S, present, color, role


def group_scenes(split, g):
    if g == "all14":
        return sorted(AG.RUNNABLE)
    return sorted(split) if g == "all15" else sorted(s for s in split if split[s] == g)


def scene_curve(curves, m, s):
    c = curves.get((m, s), {})
    return np.mean([c[i] for i in range(AG.N_START)], 0) if all(i in c for i in range(AG.N_START)) else None


def ci_band(mat, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mat), (n, len(mat)))
    bs = mat[idx].mean(1)
    return np.quantile(bs, 0.025, 0), np.quantile(bs, 0.975, 0)


def fig_curves(split, label, curves, present, color, g, tmax, name):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    sc = group_scenes(split, g)
    for m in present:
        cs = [scene_curve(curves, m, s) for s in sc]
        if any(c is None for c in cs) or not cs:
            continue
        mat = np.asarray(cs)[:, :tmax + 1]
        mu = mat.mean(0)
        lo, hi = ci_band(mat)
        x = np.arange(tmax + 1)
        ax.plot(x, mu, color=color[m], lw=2.0 if m == AG.REF else 1.5, ls="--" if m == AG.REF else "-", label=label[m])
        if m == AG.REF:
            ax.fill_between(x, lo, hi, color=color[m], alpha=0.12, lw=0, label=f"{label[m]} 95% CI")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("exploration step")
    ax.set_ylabel("coverage (MAGICIAN official)")
    ax.set_title(f"{GN[g]} ({len(sc)} scenes x 5 starts; scene-equal mean; shaded: {AG.REF} scene-level 95% CI)", loc="left", fontsize=8.5)
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, name)


def fig_auc_horizon(S, label, present, color, groups, name):
    fig, axs = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.2), squeeze=False)
    for ax, g in zip(axs[0], groups):
        for m in present:
            r = next((x for x in S["per_split"] if x["model"] == m and x["group"] == g), None)
            if r is None:
                continue
            y = [r[f"auc{k}"] for k in AG.KS]
            lo = [r[f"auc{k}_ci"][0] for k in AG.KS]
            hi = [r[f"auc{k}_ci"][1] for k in AG.KS]
            ax.plot(AG.KS, y, marker="o", ms=3.5, color=color[m], lw=1.4, ls="--" if m == AG.REF else "-", label=label[m])
            ax.fill_between(AG.KS, lo, hi, color=color[m], alpha=0.10, lw=0)
        ax.set_xticks(AG.KS)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("horizon k (steps)")
        ax.set_ylabel("AUC@k")
        ax.set_title(GN[g], loc="left", fontsize=9)
    axs[0][-1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))
    save(fig, name)


def fig_per_scene(S, split, label, present, color, metric, name):
    rows = {(r["model"], r["scene"]): r for r in S["per_scene"] if r["complete"]}
    sc = sorted(split, key=lambda s: (["train", "dev", "test", "other"].index(split[s]) if split[s] in ("train", "dev", "test", "other") else 9, s))
    sc = [s for s in sc if any((m, s) in rows for m in present)]
    fig, ax = plt.subplots(figsize=(max(7, 0.75 * len(sc) + 2), 3.4))
    w = 0.8 / max(1, len(present))
    for i, m in enumerate(present):
        xs = [j + (i - (len(present) - 1) / 2) * w for j, s in enumerate(sc) if (m, s) in rows]
        ys = [rows[(m, s)][metric] for s in sc if (m, s) in rows]
        ax.bar(xs, ys, width=w * 0.92, color=color[m], label=label[m])
    ax.set_xticks(range(len(sc)))
    ax.set_xticklabels(sc, rotation=30, ha="right", fontsize=7.5)
    prev = None
    for j, s_ in enumerate(sc):
        if split[s_] != prev:
            if j:
                ax.axvline(j - 0.5, color="#999", lw=0.8)
            ax.text(j - 0.4, 0.97, GN.get(split[s_], split[s_]), fontsize=8, va="top")
            prev = split[s_]
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric.upper().replace("AUC", "AUC@").replace("FINAL", "Final coverage"))
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, name)


def fig_improve(S, label, present, color, name):
    groups = [g for g in ("train", "dev", "test", "all14") if any(p["group"] == g for p in S["pairwise"])] or sorted({p["group"] for p in S["pairwise"]})
    mets = [("auc20", "ΔAUC@20"), ("auc100", "ΔAUC@100"), ("final", "ΔFinal")]
    ms = [m for m in present if m != AG.REF and any(p["model"] == m for p in S["pairwise"])]
    fig, axs = plt.subplots(len(groups), 3, figsize=(10, 0.28 * len(ms) * len(groups) + 1.2 * len(groups)), squeeze=False,
                            sharey=True, sharex="col", constrained_layout=True)
    for gi, g in enumerate(groups):
        pr = {p["model"]: p for p in S["pairwise"] if p["group"] == g}
        for mi, (k, t) in enumerate(mets):
            ax = axs[gi][mi]
            rows = [(i, pr[m]) for i, m in enumerate(ms) if m in pr]
            y = [p[f"d_{k}"] for _, p in rows]
            err = np.asarray([[p[f"d_{k}"] - p[f"d_{k}_ci"][0], p[f"d_{k}_ci"][1] - p[f"d_{k}"]] for _, p in rows]).T
            ax.barh([i for i, _ in rows], y, height=0.7, color=[color[ms[i]] for i, _ in rows], xerr=err, error_kw={"lw": 0.8, "capsize": 2, "ecolor": "#444"})
            ax.axvline(0, color="#333", lw=0.8)
            ax.xaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_yticks(range(len(ms)))
            ax.set_yticklabels(ms, fontsize=7.5)
            ax.invert_yaxis() if gi == 0 and mi == 0 else None
            ax.set_title(f"{t} vs {AG.REF} — {GN.get(g, g)}", loc="left", fontsize=8.5)
    fig.text(0.0, -0.01, "model ids: " + "; ".join(f"{m} = {label[m]}" for m in ms), fontsize=6.5, ha="left", va="top", wrap=True)
    save(fig, name)


def main():
    style()
    split, order, label, curves, S, present_all, color, role = load()
    for suffix, present in (("", [m for m in present_all if role.get(m) == "main"]), ("_all", present_all)):
        draw(split, label, curves, S, present, color, suffix)
    print("figures written to", FIG)


def draw(split, label, curves, S, present, color, suffix):
    for g, nm in (("all14", "coverage_all14"), ("train", "coverage_train"), ("dev", "coverage_dev"), ("test", "coverage_test")):
        fig_curves(split, label, curves, present, color, g, AG.T_MAX, nm + suffix)
    fig_curves(split, label, curves, present, color, "all14", 20, "coverage_early20" + suffix)
    for g in ("train", "dev", "test"):
        fig_curves(split, label, curves, present, color, g, 20, f"coverage_early20_{g}" + suffix)
    fig_auc_horizon(S, label, present, color, ["all14"], "auc_horizon_all14" + suffix)
    fig_auc_horizon(S, label, present, color, [g for g in ("train", "dev", "test") if any(r["group"] == g for r in S["per_split"])], "auc_horizon_by_split" + suffix)
    for k, nm in (("auc20", "per_scene_auc20"), ("auc100", "per_scene_auc100"), ("final", "per_scene_final")):
        fig_per_scene(S, split, label, present, color, k, nm + suffix)
    if S["pairwise"]:
        fig_improve(S, label, present, color, "improvement_vs_magician" + suffix)


if __name__ == "__main__":
    main()
