"""E2 figures from results/audit_summary.json and results/ksv_fixed_summary.json (CPU, read-only).
figures/audit_class_shares.{png,pdf}: where D1's novel predicted value lands relative to the true (GT) first surface, executed views.
figures/ksv_fixed_state.{png,pdf}: fixed-state A/B, KSV off (= D1) vs on: Top-1, selected true gain as % of the oracle best, share of
novel predicted value lying behind the true surface."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E2 = os.path.dirname(HERE)
FIG = os.path.join(E2, "figures")
INK, MUTED = "#222222", "#666666"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=8)


def audit():
    a = json.load(open(os.path.join(E2, "results", "audit_summary.json")))["groups"]
    groups = [("colosseum low-plateau (t >= onset)", "colosseum, plateau states"), ("colosseum other states", "colosseum, other states"),
              ("pisa t<20", "pisa s1, t < 20"), ("pisa t>=20", "pisa s1, t ≥ 20"), ("alhambra t<20", "alhambra s0, t < 20"), ("alhambra t>=20", "alhambra s0, t ≥ 20")]
    parts = [("A: behind an observed surface outside the P1 domain", lambda g: g["share_A_out_obs"], "#0072B2"),
             ("A: behind an observed surface inside the domain", lambda g: g["share_A_in_obs"], "#56B4E9"),
             ("A: behind a not-yet-observed surface", lambda g: g["share_A_out_unobs"] + g["share_A_in_unobs"], "#999999"),
             ("B: passes a D100-known surface", lambda g: g["share_B"], "#E69F00"),
             ("C: on a D100 surface, novel only under S10", lambda g: max(0.0, g["share_C"] - g["share_AC"] - g["share_BC"]), "#F0E442"),
             ("in front of the true surface / no true surface", lambda g: g["share_F"] + g["share_N"], "#CC79A7"),
             ("consistent with the true first surface", lambda g: g["share_T"], "#009E73")]
    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    y = np.arange(len(groups))[::-1]
    for gi, (key, lab) in enumerate(groups):
        g = a[f"{key} | executed"]
        left = 0.0
        for _, f, col in parts:
            w = f(g)
            ax.barh(y[gi], w, left=left, color=col, edgecolor="white", linewidth=1.0, height=0.62)
            if w >= 0.08:
                ax.text(left + w / 2, y[gi], f"{w:.2f}", ha="center", va="center", fontsize=7, color="white" if col in ("#0072B2", "#999999", "#009E73") else INK)
            left += w
        ax.text(1.01, y[gi], f"{g['states']} states", va="center", fontsize=7, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([g[1] for g in groups], fontsize=8, color=INK)
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of D1-novel predicted value of the executed view (w(z)-weighted)", fontsize=8, color=MUTED)
    style(ax)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in parts], labels=[p[0] for p in parts], fontsize=7, frameon=False,
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    ax.set_title("Where D1's novel predicted first hits lie relative to the true first surface (GT, offline)", fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"audit_class_shares.{ext}"), dpi=200, bbox_inches="tight")


def fixed():
    s = json.load(open(os.path.join(E2, "results", "ksv_fixed_summary.json")))["groups"]
    groups = [("all", "all\n140"), ("t<20", "t<20\n28"), ("t>=20", "t≥20\n112"), ("colosseum plateau", "col.\nplateau\n17"),
              ("pisa", "pisa\n20"), ("alhambra", "alham.\n20")]
    panels = [("Top-1 rate (oracle best one-step view)", lambda g, m: g[f"top1_{m}"]),
              ("selected true gain, % of oracle best", lambda g, m: 100 * g[f"sel_gain_{m}_pp"] / g["best_gain_pp"]),
              ("novel predicted value behind the true surface", lambda g, m: g[f"novel_beyond_true_{m}"])]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3))
    x = np.arange(len(groups))
    for ax, (title, f) in zip(axs, panels):
        for i, (m, col, lab) in enumerate((("off", "#D55E00", "D1 (KSV off)"), ("on", "#0072B2", "D1 + KSV"))):
            v = [f(s[g], m) for g, _ in groups]
            ax.bar(x + (i - 0.5) * 0.36, v, width=0.34, color=col, label=lab, edgecolor="white", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([g[1] for g in groups], fontsize=7)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
        style(ax)
    axs[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"ksv_fixed_state.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    audit()
    fixed()


def paired():
    """figures/closed_loop_paired.{png,pdf}: per-episode KSV - D1 differences by scene (results/closed_loop_divergence.json)."""
    rows = json.load(open(os.path.join(E2, "results", "closed_loop_divergence.json")))
    scenes = ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.1))
    rng = np.random.default_rng(0)
    for ax, (q, title) in zip(axs, (("d_auc20", "Δ AUC@20 (KSV − D1)"), ("d_auc100", "Δ AUC@100"), ("d_final", "Δ Final"))):
        for i, s in enumerate(scenes):
            for rep, col in ((0, "#0072B2"), (1, "#56B4E9"), (2, "#56B4E9")):
                v = [r[q] for r in rows if r["scene"] == s and r["rep"] == rep]
                if v:
                    ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=16, color=col, edgecolor="white", linewidth=0.6, zorder=3,
                               label=("rep 0 (paired with Table 1 D1)" if rep == 0 else "rep 1 / 2 (colosseum only)") if (i == 1 and rep < 2) else None)
            m = np.mean([r[q] for r in rows if r["scene"] == s and r["rep"] == 0])
            ax.plot([i - 0.3, i + 0.3], [m, m], color=INK, lw=1.6, zorder=4)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(["barts", "colosseum", "eiffel", "pantheon", "sestino"], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
        style(ax)
    h_, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h_, l_, fontsize=7.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    axs[2].text(0.99, 0.02, "black bar = scene mean of rep 0", transform=axs[2].transAxes, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"closed_loop_paired.{ext}"), dpi=200, bbox_inches="tight")


if __name__ == "__main__" and os.path.exists(os.path.join(E2, "results", "closed_loop_divergence.json")):
    paired()
