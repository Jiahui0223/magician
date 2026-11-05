"""E3 figures (CPU, read-only) from results/{fixed_summary.json, closed_loop_episodes.csv} and the episode files.
figures/fixed_state.{png,pdf}        fixed-state modes: Top-1, selected true gain (% of oracle best), wrong-cut share
figures/closed_loop_curves.{png,pdf} mean official coverage D1 vs D-v2 per evaluation set
figures/closed_loop_paired.{png,pdf} per-episode D-v2 - D1 differences by scene
figures/trajectories.{png,pdf}       top view of camera positions, D1 vs D-v2, colosseum episodes where D1 ends below 0.70"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import closed_loop_summary as CS  # noqa: E402

E3 = CS.E3
FIG = os.path.join(E3, "figures")
INK, MUTED = "#222222", "#666666"
C_D1, C_DV2 = "#D55E00", "#0072B2"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fixed():
    s = json.load(open(os.path.join(E3, "results", "fixed_summary.json")))["groups"]
    groups = [("gate set: all", "all\n120"), ("gate set: t<20", "t<20\n24"), ("gate set: t>=20", "t≥20\n96"), ("colosseum plateau", "col.\nplateau\n17"), ("pisa s1", "pisa s1\n20")]
    modes = [("off", "D1 (off)", C_D1), ("dv2_first", "first layer only", "#56B4E9"), ("dv2", "D-v2 (both layers)", C_DV2), ("ksv1", "E2 KSV-v1 (single ray)", "#999999")]
    panels = [("Top-1 rate (oracle best one-step view)", lambda g, m: g[m]["top1"]),
              ("selected true gain, % of oracle best", lambda g, m: 100 * g[m]["sel_gain_pp"] / g["best_gain_pp"]),
              ("wrongly cut share of the cut value", lambda g, m: g[m].get("wrong_cut_share"))]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.3))
    x = np.arange(len(groups))
    for ax, (title, f) in zip(axs, panels):
        use = [m for m in modes if f(s[groups[0][0]], m[0]) is not None]
        wd = 0.8 / len(use)
        for i, (m, lab, col) in enumerate(use):
            ax.bar(x + (i - (len(use) - 1) / 2) * wd, [f(s[g], m) for g, _ in groups], width=wd * 0.92, color=col, label=lab, edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([g[1] for g in groups], fontsize=7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
        style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save(fig, "fixed_state")


def curves(D, V):
    sets = [("colosseum (5 starts × 3 reps)", [k for k in V if k[0] == "colosseum" and k in D]), ("pisa (5 starts × 3 reps)", [k for k in V if k[0] == "pisa" and k in D]),
            ("Dev-5 (rep 0)", [k for k in V if k[2] == 0 and k[0] in CS.SPLIT["dev"] and k in D]), ("All-14 (rep 0)", [k for k in V if k[2] == 0 and k in D])]
    n_dev = len({k[0] for k in sets[2][1]})
    n_all = len({k[0] for k in sets[3][1]})
    sets = [s for s in sets[:2] if s[1]] + ([sets[2]] if n_dev == 5 else []) + ([sets[3]] if n_all == 14 else [])
    fig, axs = plt.subplots(1, len(sets), figsize=(4.1 * len(sets), 3.3), sharey=True, squeeze=False)
    for ax, (title, sel) in zip(axs[0], sets):
        for src, col, lab in ((D, C_D1, "D1"), (V, C_DV2, "D-v2")):
            ax.plot(np.mean([np.asarray(src[k]["cov"]) for k in sel], 0), color=col, lw=2.0, label=lab)
        ax.set_title(f"{title}, {len(sel)} episodes", fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("step", fontsize=8, color=MUTED)
        ax.grid(alpha=0.25)
        style(ax)
    axs[0][0].set_ylabel("official coverage", fontsize=8, color=MUTED)
    axs[0][0].legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "closed_loop_curves")


def paired():
    rows = list(csv.DictReader(open(os.path.join(E3, "results", "closed_loop_episodes.csv"))))
    scenes = [s for s in CS.ALL14 if any(r["scene"] == s for r in rows)]
    scenes = sorted(scenes, key=lambda s: (0 if s in CS.SPLIT["dev"] else 1 if s in CS.SPLIT["train"] else 2, s))
    wide = len(scenes) > 6
    fig, axs = plt.subplots(3, 1, figsize=(9.5, 8.2), sharex=True) if wide else plt.subplots(1, 3, figsize=(11.5, 3.3))
    rng = np.random.default_rng(0)
    for ax, (q, title) in zip(axs, (("d_auc20", "Δ AUC@20 (D-v2 − D1)"), ("d_auc100", "Δ AUC@100"), ("d_final", "Δ Final coverage"))):
        for i, s in enumerate(scenes):
            for reps, col, lab in (((0,), C_DV2, "rep 0"), ((1, 2), "#56B4E9", "rep 1 / 2")):
                v = [float(r[q]) for r in rows if r["scene"] == s and int(r["rep"]) in reps]
                if v:
                    ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=15, color=col, edgecolor="white", linewidth=0.6, zorder=3)
            m = np.mean([float(r[q]) for r in rows if r["scene"] == s])
            ax.plot([i - 0.32, i + 0.32], [m, m], color=INK, lw=1.6, zorder=4)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels([s.replace("_museum", "").replace("neuschwanstein", "neuschw.") for s in scenes], fontsize=7, rotation=35 if len(scenes) > 6 else 0, ha="right" if len(scenes) > 6 else "center")
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
        style(ax)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_DV2, label="rep 0"), plt.Line2D([], [], marker="o", ls="", color="#56B4E9", label="rep 1 / 2"),
         plt.Line2D([], [], color=INK, lw=1.6, label="scene mean")]
    fig.legend(handles=h, fontsize=7.5, frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03 if wide else -0.1))
    fig.tight_layout()
    save(fig, "closed_loop_paired")


def trajectories(D, V):
    keys = sorted(k for k in V if k[0] == "colosseum" and k in D and D[k]["cov"][100] < 0.70)[:4]
    if not keys:
        return
    lut, C = CS.centres("colosseum")
    hor = [0, 2]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 3.6), squeeze=False)
    for ax, k in zip(axs[0], keys):
        ax.scatter(C[:, hor[0]], C[:, hor[1]], s=2, color="#DDDDDD", zorder=1)
        for src, col, lab in ((D, C_D1, "D1"), (V, C_DV2, "D-v2")):
            P = np.asarray([C[lut[tuple(json.loads(x))]] for x in src[k]["keys"]])
            ax.plot(P[:, hor[0]], P[:, hor[1]], color=col, lw=1.4, alpha=0.9, label=f"{lab} (Final {src[k]['cov'][100]:.2f})", zorder=3)
            ax.scatter(P[-1, hor[0]], P[-1, hor[1]], s=28, color=col, edgecolor="white", zorder=4)
        P0 = C[lut[tuple(json.loads(D[k]["keys"][0]))]]
        ax.scatter(P0[hor[0]], P0[hor[1]], s=50, marker="*", color=INK, zorder=5)
        ax.set_title(f"colosseum start {k[1]}, rep {k[2]}", fontsize=8.5, color=INK, loc="left")
        ax.set_aspect("equal")
        ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#CCCCCC")
    fig.suptitle("Camera positions, top view (grey = all camera positions of the action space; star = start; dot = end)", fontsize=8.5, color=MUTED, y=1.0)
    fig.tight_layout()
    save(fig, "trajectories")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    D, V, _ = CS.load()
    fixed()
    curves(D, V)
    paired()
    trajectories(D, V)
