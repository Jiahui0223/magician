"""Weekly-report figures and tables for EXP6 (V2 / V3 / V3.1 / V4): everything is read from the measured result files
(reports/g3_v4.json, reports/g12_v4_s_final_v4.json, the training eval logs, Experiment A results). English labels in
the figures (no CJK font dependency); the Markdown report is Chinese. Usage: python make_weekly_report.py"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E4 = os.path.join(R, "experiments", "exp6_v4")
OUT = os.path.join(E4, "weekly_report")
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

# ---------------------------------------------------------------- palette (reference instance of the dataviz skill, light mode)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"
FAMILY = {"MAG-G": "Official MAGICIAN", "MAG-B": "Official MAGICIAN", "V2O": "Outdoor-only scratch", "B0": "Mixed fine-tune (V2 lineage)", "B2": "Mixed fine-tune (V2 lineage)",
          "E2D": "Mixed fine-tune (V2 lineage)", "T0": "Mixed scratch (V3)", "T1": "Mixed scratch (V3)", "V31": "Outdoor-only scratch", "V4": "Outdoor-only scratch"}
FCOL = {"Official MAGICIAN": MUTED, "Outdoor-only scratch": BLUE, "Mixed fine-tune (V2 lineage)": AQUA, "Mixed scratch (V3)": ORANGE}
GCOL = {g: FCOL[f] for g, f in FAMILY.items()}
LINECOL = {"V4": BLUE, "V31": ORANGE, "B2": AQUA, "E2D": YELLOW, "T1": MAGENTA, "V2O": VIOLET, "MAG-B": MUTED}
NAME = {"MAG-G": "MAG-G", "MAG-B": "MAG-B", "V2O": "V2 Original", "B0": "B0", "B2": "B2", "E2D": "E2-D100", "T0": "V3-T0", "T1": "V3-T1", "V31": "V3.1", "V4": "V4"}
NEW = ("V31", "V4")
GRP = ["MAG-G", "MAG-B", "V2O", "B0", "B2", "E2D", "T0", "T1", "V31", "V4"]
GPUH = {"MAG-G": None, "MAG-B": None, "V2O": 18.7, "B0": 24.2, "B2": 24.2, "E2D": 27.8, "T0": 63.7, "T1": 65.0, "V31": 57.6, "V4": 17.3}
PARAMS = {"V2O": 24.82, "B0": 24.82, "B2": 24.82, "E2D": 24.82, "T0": 23.54, "T1": 23.73, "V31": 23.54, "V4": 24.85}
TRAIN = {"V2O": "Outdoor-only scratch 60k (sparse P1)", "B0": "Mixed FT 20k (sparse / sparse)", "B2": "Mixed FT 20k (D100 / sparse)", "E2D": "Mixed FT 20k + 5k (D100)",
         "T0": "Mixed scratch 40k (D100)", "T1": "Mixed scratch 40k (D100) + bottleneck X-Attn", "V31": "Outdoor-only scratch 40k (D100)", "V4": "Outdoor-only scratch 40k (D100) + Ray X-Attn",
         "MAG-G": "official weights, greedy", "MAG-B": "official weights, beam 10x10"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.titlecolor": INK, "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5, "font.family": "DejaVu Sans"})


def clean(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


G3 = json.load(open(os.path.join(E4, "reports", "g3_v4.json")))
G12 = json.load(open(os.path.join(E4, "reports", "g12_v4_s_final_v4.json")))["summary"]
AX = os.path.join(E4, "ablation_study", "experiment_A_distp1_dense_vs_sparse", "results")
AG3 = json.load(open(os.path.join(AX, "a_g3.json")))
AFX = json.load(open(os.path.join(AX, "a_fixed_summary.json")))
DOM = {"indoor": "MP3D zero-shot (10 ep x 150 steps)", "outdoor": "Macarons++ (30 ep x 100 steps)"}


def mean_ci(d, met, g):
    m = G3["means"][d][met][g]
    return m["mean"], m["ci95"][0], m["ci95"][1]


# ---------------------------------------------------------------- Figure 1: architectures
def fig_architectures():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    rows = [("V2 Original / B0 / B2 / E2-D100  (EXP5, V2 family)", AQUA,
             [("Partial TSDF P1\n(coarse 8x grid)", False), ("3D U-Net\nF0 (32ch @8x) + F1 (64ch @16x)", False), ("32-point ray sampling\n+ Conv1d ray encoder (64ch)", False), ("2D Rectified-Flow U-Net\n(b, y) on 64 x 114 rays", False), ("DistP1, K = 4\nGreedy-NV planner", False)],
             "24.8 M params | scratch 60k outdoor (V2O) -> Mixed FT (B0/B2/E2D)"),
            ("V3-T0 / V3-T1  (EXP6 V3, Mixed scratch 40k)", ORANGE,
             [("Partial TSDF P1\n(4x grid + valid)", False), ("Simple 3D VAE\nmu / logvar (16ch @8x)\n+ train-time decoder", True), ("32-point ray sampling on mu\n+ Conv1d ray encoder", True), ("2D Rectified-Flow U-Net\n(T1: + bottleneck X-Attn\nto scene memory, 191k)", True), ("DistP1, K = 4\nGreedy-NV planner", False)],
             "23.5 M (T0) / 23.7 M (T1) | L = L_FM + L_recon + beta KL"),
            ("V3.1  (this week: V3-T0 architecture, Outdoor-only scratch 40k)", BLUE,
             [("Partial TSDF P1\n(4x grid + valid)", False), ("Simple 3D VAE\nmu / logvar (16ch @8x)", False), ("32-point ray sampling on mu\n+ Conv1d ray encoder", False), ("2D Rectified-Flow U-Net\n(unchanged)", False), ("DistP1, K = 4\nGreedy-NV planner", False)],
             "23.5 M | same recipe as T0, zero MP3D samples | 57.6 GPU-h"),
            ("V4  (this week: V2 U-Net + Ray-Aligned X-Attn, Outdoor-only scratch 40k)", BLUE,
             [("Partial TSDF P1\n(coarse 8x grid)", False), ("3D U-Net\nF0 + F1 (unchanged)", False), ("32-point ray sampling + Conv1d\n+ Ray-Aligned X-Attn (25k):\nper-ray attention over its\n32 sample tokens, residual", True), ("2D Rectified-Flow U-Net\n(unchanged)", False), ("DistP1, K = 4\nGreedy-NV planner", False)],
             "24.85 M | V2 recipe, zero MP3D samples | 17.3 GPU-h")]
    y0, dy, bw, bh = 86, 23.5, 16.5, 12
    xs = [3, 22, 41, 60, 79]
    for ri, (title, col, boxes, note) in enumerate(rows):
        y = y0 - ri * dy
        ax.text(2, y + bh / 2 + 2.2, title, fontsize=10, fontweight="semibold", color=col, va="bottom")
        for bi, (txt, changed) in enumerate(boxes):
            x = xs[bi]
            ax.add_patch(FancyBboxPatch((x, y - bh / 2), bw, bh, boxstyle="round,pad=0.3,rounding_size=1.2", linewidth=1.6 if changed else 0.9,
                                        edgecolor=col if changed else INK2, facecolor=(col if changed else GRID), alpha=0.22 if changed else 0.45))
            ax.text(x + bw / 2, y, txt, ha="center", va="center", fontsize=7.6, color=INK, linespacing=1.25)
            if bi < len(boxes) - 1:
                ax.add_patch(FancyArrowPatch((x + bw + 0.3, y), (xs[bi + 1] - 0.3, y), arrowstyle="-|>", mutation_scale=9, color=INK2, linewidth=0.9))
        ax.text(97.5, y - bh / 2 - 1.2, note, ha="right", va="top", fontsize=7.8, color=INK2)
    ax.text(2, 2, "Boxes with a coloured border = the module that differs from the V2 baseline. Planner, scoring (DistP1), candidates and the 2D flow are identical across all rows.", fontsize=8, color=INK2)
    fig.savefig(os.path.join(FIG, "fig1_architectures.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 2: G3 bars
def fig_g3_bars():
    fig, axs = plt.subplots(2, 3, figsize=(14, 7.2))
    for ri, d in enumerate(("indoor", "outdoor")):
        for ci, (met, lab) in enumerate((("auc", "AUC"), ("final", "Final coverage"), ("cov20", "Cov@20"))):
            ax = axs[ri, ci]
            xs = np.arange(len(GRP))
            for i, g in enumerate(GRP):
                m, lo, hi = mean_ci(d, met, g)
                ax.bar(i, m, width=0.72, color=GCOL[g], alpha=0.95 if g in NEW else 0.6, edgecolor=SURF, linewidth=1.0, hatch="//" if g in NEW else None)
                ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK2, elinewidth=1.0, capsize=2.5)
                ax.text(i, m + (hi - m) + 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.2, color=INK, fontweight="semibold" if g in NEW else "normal")
            ax.set_xticks(xs); ax.set_xticklabels([NAME[g] for g in GRP], rotation=35, ha="right", fontsize=8)
            for t, g in zip(ax.get_xticklabels(), GRP):
                t.set_fontweight("semibold" if g in NEW else "normal")
            ax.set_title(f"{DOM[d].split(' (')[0]} — {lab}", loc="left")
            ax.set_ylim(0, {"indoor": 0.86, "outdoor": 1.10}[d] if met != "auc" else {"indoor": 0.64, "outdoor": 0.94}[d])
            clean(ax); ax.grid(axis="x", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=FCOL[f], alpha=0.75) for f in FCOL]
    fig.legend(handles, list(FCOL), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("G3 closed loop: scene/floor-mean with scene-level bootstrap 95% CI (hatched = this week's Outdoor-only models; MP3D is zero-shot for V2 Original, V3.1, V4)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(os.path.join(FIG, "fig2_g3_bars.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 4: paired differences (forest)
PAIRS_SHOW = [("V4-V31", "V4 − V3.1 (X-Attn U-Net vs VAE, same data/budget)"), ("V31-T0", "V3.1 − V3-T0 (Outdoor-only vs Mixed, same arch)"), ("V4-T1", "V4 − V3-T1"),
              ("V4-V2O", "V4 − V2 Original"), ("V31-V2O", "V3.1 − V2 Original"), ("V4-B2", "V4 − B2"), ("V31-B2", "V3.1 − B2"), ("V4-E2D", "V4 − E2-D100"), ("V31-E2D", "V3.1 − E2-D100"),
              ("V4-MAG-B", "V4 − MAG-B"), ("V31-MAG-B", "V3.1 − MAG-B")]


def fig_paired():
    fig, axs = plt.subplots(1, 4, figsize=(15, 5.2), sharey=True)
    ys = np.arange(len(PAIRS_SHOW))[::-1]
    for ci, (d, met, lab) in enumerate((("indoor", "auc", "MP3D ΔAUC"), ("indoor", "final", "MP3D ΔFinal"), ("outdoor", "auc", "Macarons++ ΔAUC"), ("outdoor", "final", "Macarons++ ΔFinal"))):
        ax = axs[ci]
        for y, (k, _) in zip(ys, PAIRS_SHOW):
            p = G3["pairs"][k][d]["paired"][met]
            c = G3["pairs"][k][d]["criteria"]
            col = GREEN if c["meaningful"] else (RED if c["degraded"] else INK2)
            ax.plot([p["ci95"][0], p["ci95"][1]], [y, y], color=col, lw=2.0, solid_capstyle="round")
            ax.plot(p["mean"], y, "o", color=col, ms=6.5, markeredgecolor=SURF, markeredgewidth=1.0)
            ax.text(max(p["ci95"][1], 0) + 0.006, y, f"{p['mean']:+.3f} ({p['wins']}/{p['losses']})", va="center", fontsize=7.4, color=INK2)
        ax.axvline(0, color=INK2, lw=0.9)
        ax.axvspan(-0.010, 0.010, color=GRID, alpha=0.5, lw=0)
        ax.set_title(lab, loc="left"); ax.set_xlim(-0.16, 0.30); clean(ax); ax.grid(axis="y", visible=False)
        if ci == 0:
            ax.set_yticks(ys); ax.set_yticklabels([l for _, l in PAIRS_SHOW], fontsize=8.2)
    fig.suptitle("Paired scene-level differences with bootstrap 95% CI (wins/losses in brackets); grey band = ±0.010 (Task 3 §4 threshold); "
                 "green = 'meaningful' on this domain, red = 'degraded' (either metric), grey = neither", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(FIG, "fig4_paired_differences.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 5: per scene dots
def fig_per_scene():
    keys = ["MAG-B", "V2O", "B2", "E2D", "T1", "V31", "V4"]
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4), gridspec_kw={"width_ratios": [5, 6]})
    for ax, d, met in ((axs[0], "indoor", "auc"), (axs[1], "outdoor", "auc")):
        scenes = list(G3["per_scene"][d])
        ys = np.arange(len(scenes))[::-1]
        for y, s in zip(ys, scenes):
            vals = G3["per_scene"][d][s]
            ax.plot([min(vals[g]["auc"] for g in keys), max(vals[g]["auc"] for g in keys)], [y, y], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            for g in keys:
                ax.plot(vals[g]["auc"], y, "o" if g in NEW else "o", ms=9 if g in NEW else 6.5, color=LINECOL[g], markeredgecolor=INK if g in NEW else SURF, markeredgewidth=1.2 if g in NEW else 0.8, zorder=3)
        ax.set_yticks(ys); ax.set_yticklabels([s.replace("_", " ") for s in scenes], fontsize=8.5)
        ax.set_title(f"{DOM[d].split(' (')[0]} — AUC per {'floor' if d == 'indoor' else 'scene'}", loc="left"); clean(ax); ax.grid(axis="y", visible=False)
        ax.set_xlabel("AUC (mean over the 2 or 5 starts)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=LINECOL[g], ms=8 if g in NEW else 6, markeredgecolor=INK if g in NEW else SURF, label=NAME[g]) for g in keys]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(FIG, "fig5_per_scene_auc.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 6: training curves + cost
def evals(path):
    return [json.loads(x) for x in open(path)]


def fig_training():
    curves = {"V4": (evals(f"{R}/work_dirs/exp6_v4/runs/main/eval.jsonl"), "fm_sc_sub", "samp_sc_sub"),
              "V31": (evals(f"{R}/work_dirs/exp6_v3_1/runs/main/eval.jsonl"), "fm_out_sc", "samp_out_sc"),
              "T0": (evals(f"{R}/work_dirs/exp6_v3/task0_vae/runs/main/eval.jsonl"), "fm_out_sc", "samp_out_sc"),
              "T1": (evals(f"{R}/work_dirs/exp6_v3/task1_xattn/runs/main/eval.jsonl"), "fm_out_sc", "samp_out_sc"),
              "V2O": (evals(f"{R}/work_dirs/exp5_future_view/task1/training/main/eval.jsonl"), "fm_sc_sub", "samp_sc_sub")}
    cols = {"V4": BLUE, "V31": ORANGE, "T0": AQUA, "T1": MAGENTA, "V2O": VIOLET}
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for g, (ev, fk, sk) in curves.items():
        ev = [e for e in ev if e["step"] > 0]
        axs[0].plot([e["step"] for e in ev], [e[fk] for e in ev], color=cols[g], lw=2 if g in NEW else 1.3, ls="-" if g in NEW else "--", label=NAME[g])
        es = [(e["step"], e[sk]["fair_es_arclength_vox"]) for e in ev if e.get(sk)]
        axs[1].plot([s for s, _ in es], [v for _, v in es], "o-", color=cols[g], lw=2 if g in NEW else 1.3, ls="-" if g in NEW else "--", ms=4, label=NAME[g])
    axs[0].set_yscale("log"); axs[0].set_title("Dev FM loss on S-c (held-out Train-6 trajectories), EMA", loc="left"); axs[0].set_xlabel("step"); axs[0].legend(ncol=2)
    axs[1].set_title("Dev fair ES on S-c (voxels, lower is better), EMA", loc="left"); axs[1].set_xlabel("step"); axs[1].legend(ncol=2)
    for ax in axs[:2]:
        clean(ax)
    order = ["V2O", "B2", "E2D", "T0", "T1", "V31", "V4"]
    xs = np.arange(len(order))
    axs[2].bar(xs, [GPUH[g] for g in order], color=[GCOL[g] for g in order], alpha=0.85, hatch=["//" if g in NEW else None for g in order], edgecolor=SURF)
    for i, g in enumerate(order):
        axs[2].text(i, GPUH[g] + 1, f"{GPUH[g]:.1f}", ha="center", fontsize=8)
    axs[2].set_xticks(xs); axs[2].set_xticklabels([NAME[g] for g in order], rotation=30, ha="right"); axs[2].set_title("Training cost (GPU-hours, L40S; FT lineages cumulative)", loc="left"); clean(axs[2]); axs[2].grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_training_and_cost.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 7: fixed-state G1/G2
def fig_fixed():
    models = ["B2", "E2D", "T0", "T1", "V31", "V4"]
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))
    for ri, (d, lab) in enumerate((("valsel", "MP3D VALSEL (90 states, zero-shot)"), ("dev3", "Macarons++ Dev-3 (275 states)"))):
        M = G12[d]["all"]["means"]
        for ci, (met, ml, kind) in enumerate((("gain", "True one-step coverage gain of the pick (pp, higher is better)", "gain"), ("top1", "Top-1 (share of states where the pick is an oracle-best camera)", "top1"), ("mae_unknown", "Depth MAE on unknown rays (lower is better)", "mae"))):
            ax = axs[ri, ci]
            for i, m in enumerate(models):
                v = M[f"{m}_{met}"]
                ax.bar(i, v["mean"], width=0.7, color=GCOL[m], alpha=0.95 if m in NEW else 0.6, hatch="//" if m in NEW else None, edgecolor=SURF)
                ax.errorbar(i, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2.5)
                ax.text(i, v["ci95"][1] + 0.01 * max(1, v["ci95"][1]), f"{v['mean']:.3f}", ha="center", va="bottom", fontsize=7.5)
            if kind == "gain":
                ax.axhline(M["B2_oracle"]["mean"], color=INK2, lw=1, ls=":"); ax.text(len(models) - 0.5, M["B2_oracle"]["mean"], f" oracle {M['B2_oracle']['mean']:.2f}", va="bottom", ha="right", fontsize=7.5, color=INK2)
            ax.set_xticks(range(len(models))); ax.set_xticklabels([NAME[m] for m in models], rotation=25, ha="right")
            ax.set_title(f"{lab}\n{ml}", loc="left", fontsize=9); clean(ax); ax.grid(axis="x", visible=False)
    fig.suptitle("Fixed-state G1/G2 on the development states (same candidates, same K = 4 noise, sparse DistP1); hatched = Outdoor-only models", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG, "fig7_fixed_state_g1g2.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 8: Experiment A
def fig_expA():
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw={"width_ratios": [4.2, 4, 3.2]})
    ax = axs[0]
    x = 0
    ticks, labels = [], []
    for d in ("indoor", "outdoor"):
        for met, ml in (("auc", "AUC"), ("final", "Final"), ("cov20", "Cov@20")):
            for j, (g, col, h) in enumerate((("V4", AQUA, None), ("A1", BLUE, "//"))):
                m = AG3["means"][d][met][g]
                ax.bar(x + j * 0.38, m["mean"], width=0.36, color=col, alpha=0.9, hatch=h, edgecolor=SURF)
                ax.errorbar(x + j * 0.38, m["mean"], yerr=[[m["mean"] - m["ci95"][0]], [m["ci95"][1] - m["mean"]]], fmt="none", ecolor=INK2, elinewidth=1, capsize=2)
                ax.text(x + j * 0.38, m["ci95"][1] + 0.012, f"{m['mean']:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
            ticks.append(x + 0.19); labels.append(f"{'MP3D' if d == 'indoor' else 'Mac++'}\n{ml}")
            x += 1.1
        x += 0.5
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=8); ax.set_title("Exp. A — G3: A0 sparse (= V4) vs A1 dense scoring", loc="left"); clean(ax); ax.grid(axis="x", visible=False)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=AQUA), plt.Rectangle((0, 0), 1, 1, color=BLUE, hatch="//")], ["A0 S10 band_dil", "A1 D100 band_dil"], loc="upper left")
    ax = axs[1]
    scenes = list(AG3["per_scene"]["indoor"]) + list(AG3["per_scene"]["outdoor"])
    vals = {**AG3["per_scene"]["indoor"], **AG3["per_scene"]["outdoor"]}
    ys = np.arange(len(scenes))[::-1]
    for y, s in zip(ys, scenes):
        a0, a1 = vals[s]["V4"]["auc"], vals[s]["A1"]["auc"]
        ax.annotate("", xy=(a1, y), xytext=(a0, y), arrowprops={"arrowstyle": "-|>", "color": GREEN if a1 > a0 else RED, "lw": 1.6, "mutation_scale": 10})
        ax.plot(a0, y, "o", color=AQUA, ms=6, zorder=3); ax.plot(a1, y, "o", color=BLUE, ms=6, zorder=3)
    ax.set_yticks(ys); ax.set_yticklabels([s.replace("_", " ") for s in scenes], fontsize=8); ax.set_title("AUC per floor / scene: A0 → A1", loc="left"); clean(ax); ax.grid(axis="y", visible=False)
    ax.axhline(5.5, color=GRID, lw=1); ax.text(0.98, 5.55, "MP3D above / Macarons++ below", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=7.5, color=INK2)
    ax = axs[2]
    x = 0; ticks, labels = [], []
    for d, dl in (("valsel", "MP3D VALSEL"), ("dev3", "Mac++ Dev-3")):
        M = AFX[d]["all"]["means"]
        for met, ml in (("gain", "pick gain pp"), ("known_frac", "known-hit frac")):
            for j, (a, col, h) in enumerate((("A0", AQUA, None), ("A1", BLUE, "//"))):
                v = M[f"{a}_{met}"]
                ax.bar(x + j * 0.38, v["mean"], width=0.36, color=col, alpha=0.9, hatch=h, edgecolor=SURF)
                ax.text(x + j * 0.38, v["mean"] + 0.015, f"{v['mean']:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
            ticks.append(x + 0.19); labels.append(f"{dl}\n{ml}"); x += 1.1
        x += 0.4
    ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5); ax.set_ylim(0, 1.32); ax.set_title("Fixed states: pick gain (pp), known-hit share", loc="left", fontsize=9); clean(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig8_experiment_A.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- Figure 9: cost vs performance
def fig_cost():
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, d in zip(axs, ("indoor", "outdoor")):
        for g in ["V2O", "B2", "E2D", "T0", "T1", "V31", "V4"]:
            m, lo, hi = mean_ci(d, "auc", g)
            ax.errorbar(GPUH[g], m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=GRID, elinewidth=1.2)
            ax.plot(GPUH[g], m, "o", ms=11 if g in NEW else 8, color=GCOL[g], markeredgecolor=INK if g in NEW else SURF, markeredgewidth=1.2 if g in NEW else 0.8, zorder=3)
            ax.annotate(NAME[g], (GPUH[g], m), xytext=(-8, -14) if g == "V31" else (6, 5), textcoords="offset points", ha="right" if g == "V31" else "left", fontsize=8.5, fontweight="semibold" if g in NEW else "normal")
        for g, ls in (("MAG-G", ":"), ("MAG-B", "--")):
            m, _, _ = mean_ci(d, "auc", g)
            ax.axhline(m, color=MUTED, lw=1, ls=ls); ax.text(88, m, f"{g} {m:.3f} ", va="bottom", ha="right", fontsize=7.5, color=INK2)
        ax.set_xscale("log"); ax.set_xlim(14, 90); ax.set_xlabel("training GPU-hours (L40S, log scale; fine-tune lineages cumulative)"); ax.set_ylabel("G3 AUC")
        ax.set_title(f"{DOM[d].split(' (')[0]}: closed-loop AUC vs training cost", loc="left"); clean(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig9_cost_vs_performance.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- tables
def tables():
    def fmt(d, met, g):
        m, lo, hi = mean_ci(d, met, g)
        return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
    L = ["| 模型 | 训练来源 / P1 口径 | MP3D zero-shot? | MP3D AUC@150 | Final | Cov@20 | 停滞 ep | Mac++ AUC@100 | Final | Cov@20 | 参数 M | 训练 GPU-h |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for g in GRP:
        zs = {"V2O": "yes", "V31": "yes", "V4": "yes", "MAG-G": "official", "MAG-B": "official"}.get(g, "no (Mixed)")
        L.append(f"| {NAME[g]} | {TRAIN[g]} | {zs} | {fmt('indoor', 'auc', g)} | {fmt('indoor', 'final', g)} | {fmt('indoor', 'cov20', g)} | {G3['stall_mp3d'][g]['episodes_with_stall']} | "
                 f"{fmt('outdoor', 'auc', g)} | {fmt('outdoor', 'final', g)} | {fmt('outdoor', 'cov20', g)} | {PARAMS.get(g, '—')} | {GPUH[g] if GPUH[g] else '—'} |")
        rows.append([NAME[g], TRAIN[g], zs] + [f"{mean_ci(d, m, g)[0]:.4f}" for d in ("indoor", "outdoor") for m in ("auc", "final", "cov20")] + [G3["stall_mp3d"][g]["episodes_with_stall"], PARAMS.get(g, ""), GPUH[g] or ""])
    open(os.path.join(TAB, "T3_g3_unified.md"), "w").write("\n".join(L) + "\n")
    with open(os.path.join(TAB, "T3_g3_unified.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "training", "mp3d_zero_shot", "mp3d_auc", "mp3d_final", "mp3d_cov20", "mac_auc", "mac_final", "mac_cov20", "mp3d_stall_episodes", "params_M", "train_gpu_h"]); w.writerows(rows)
    L = ["| 配对 | 数据集 | ΔAUC [CI]（胜/负） | ΔFinal [CI]（胜/负） | ΔCov@20 | 判定 |", "|---|---|---|---|---|---|"]
    for k, lab in PAIRS_SHOW:
        for d in ("indoor", "outdoor"):
            p, c = G3["pairs"][k][d]["paired"], G3["pairs"][k][d]["criteria"]
            v = "有意义" if c["meaningful"] else ("退化" if c["degraded"] else "无差异")
            L.append(f"| {lab} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {p['auc']['mean']:+.3f} [{p['auc']['ci95'][0]:+.3f}, {p['auc']['ci95'][1]:+.3f}]（{p['auc']['wins']}/{p['auc']['losses']}） | "
                     f"{p['final']['mean']:+.3f} [{p['final']['ci95'][0]:+.3f}, {p['final']['ci95'][1]:+.3f}]（{p['final']['wins']}/{p['final']['losses']}） | {p['cov20']['mean']:+.3f} | {v} |")
    open(os.path.join(TAB, "T4_paired_differences.md"), "w").write("\n".join(L) + "\n")
    models = ["B2", "E2D", "T0", "T1", "V31", "V4"]
    L = ["| 集 | 指标 | " + " | ".join(NAME[m] for m in models) + " |", "|---|---|" + "---|" * len(models)]
    for d, dl in (("valsel", "MP3D VALSEL（90 态）"), ("dev3", "Macarons++ Dev-3（275 态）")):
        M = G12[d]["all"]["means"]
        for met, ml in (("gain", "被选动作真实增益 pp"), ("oracle", "oracle 上限 pp"), ("top1", "Top-1"), ("mae_unknown", "未知射线 MAE"), ("mae_new", "新表面 MAE"), ("hit_agree", "HIT 一致率"), ("rel_std", "K=4 相对离散度")):
            L.append(f"| {dl} | {ml} | " + " | ".join(f"{M[f'{m}_{met}']['mean']:.3f}" for m in models) + " |")
    open(os.path.join(TAB, "T5_fixed_state_g1g2.md"), "w").write("\n".join(L) + "\n")
    L = ["| 数据集 | 方法 | AUC [CI] | Final [CI] | Cov@20 [CI] |", "|---|---|---|---|---|"]
    for d in ("indoor", "outdoor"):
        for g, nm in (("V4", "A0 V4-SparseScore"), ("A1", "A1 V4-DenseScore")):
            M = AG3["means"][d]
            L.append(f"| {'MP3D' if d == 'indoor' else 'Macarons++'} | {nm} | " + " | ".join(f"{M[m][g]['mean']:.3f} [{M[m][g]['ci95'][0]:.3f}, {M[m][g]['ci95'][1]:.3f}]" for m in ("auc", "final", "cov20")) + " |")
    for d in ("indoor", "outdoor"):
        p = AG3["pairs"]["A1-V4"][d]["paired"]
        L.append(f"| {'MP3D' if d == 'indoor' else 'Macarons++'} | A1 − A0 | {p['auc']['mean']:+.3f} [{p['auc']['ci95'][0]:+.3f}, {p['auc']['ci95'][1]:+.3f}]（{p['auc']['wins']}/{p['auc']['losses']}） | "
                 f"{p['final']['mean']:+.3f} [{p['final']['ci95'][0]:+.3f}, {p['final']['ci95'][1]:+.3f}]（{p['final']['wins']}/{p['final']['losses']}） | {p['cov20']['mean']:+.3f} |")
    open(os.path.join(TAB, "T6_experiment_A.md"), "w").write("\n".join(L) + "\n")
    print("tables written")


if __name__ == "__main__":
    fig_architectures(); fig_g3_bars(); fig_paired(); fig_per_scene(); fig_training(); fig_fixed(); fig_expA(); fig_cost(); tables()
    import shutil
    shutil.copy(os.path.join(E4, "reports", "figures", "g3_v4_coverage_curves.png"), os.path.join(FIG, "fig3_coverage_curves.png"))
    print("figures:", sorted(os.listdir(FIG)))
