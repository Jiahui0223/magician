"""FisherRF audit A6 (CPU): figure of the fixed-state scorer test (Top-1 true coverage gain per decision rule, two candidate sets).
Input: <probe>/a5_summary.json (a5_analyze.py). Output: <probe>/fig_scorer_top1.png"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

probe = sys.argv[1]
S = json.load(open(os.path.join(probe, "a5_summary.json")))["scorer_summary"]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
rows = [("oracle", "Oracle (GT, upper bound)"), ("unk_true_fov", "Frontier count (never-observed voxels in FOV)"),
        ("novelty", "Rendered novelty (silhouette < 0.5)"), ("random", "Random pick"),
        ("eig", "FisherRF EIG (official pose_eval)"), ("eig_clean", "FisherRF EIG, phantoms removed")]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), facecolor=SURF)
for ax, (sub, title) in zip(axes, [("legal", "One-step legal neighbours (D3a decision set)"),
                                   ("uniform", "48 uniform grid poses (f1c4 candidate distribution)")]):
    L = S[sub]
    vals = [L[k] if k in ("oracle", "random") else L[k]["top1"] for k, _ in rows]
    y = np.arange(len(rows))[::-1]
    cols = [ORANGE if k.startswith("eig") else BLUE for k, _ in rows]
    ax.barh(y, vals, height=0.55, color=cols, edgecolor=SURF, linewidth=2)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.015, yi, f"{v:.2f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in rows], fontsize=9, color=INK)
    ax.set_xlabel("true coverage gain of the chosen view (pp)", fontsize=9, color=INK2)
    ax.set_title(f"{title}, {L['n_states']} states", fontsize=10, color=INK, loc="left")
    ax.set_facecolor(SURF)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.set_xlim(0, max(vals) * 1.15)
axes[1].set_yticklabels([])
fig.suptitle("Fixed-state scorer test on 8 recorded f1c4 episodes (states t = 5/15/30/50/75): FisherRF EIG does not beat a random pick",
             fontsize=10.5, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig(os.path.join(probe, "fig_scorer_top1.png"), dpi=150, facecolor=SURF)
print("saved", os.path.join(probe, "fig_scorer_top1.png"))
