"""Coverage-vs-step curves of the five hypothesis variants (2026-11-16, user request): D0 deterministic U-Net, single generated sample,
average of 4 samples (b, y), average of 4 samples (depth), Ours (4 hypotheses + CDGR). Same network / data / planner / protocol.
Curves from experiment_G.../scripts/g3_collect.outdoor() (validated loader; scene = mean of 5 official starts; group = scene-equal mean).
Left panel: steps 0-100; right panel: zoom on steps 50-100. Band = 95 % scene-bootstrap CI of the mean, drawn for Ours only (five
overlapping bands are unreadable). Groups: Unseen-6 (main, matches the compact table) and All-14 (supplementary).
Outputs fig/table_hypotheses/curves_{unseen6,all14}.{pdf,png} and curves_{unseen6,all14}.csv (mean curves)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_G_generative_vs_deterministic", "scripts"))
import g3_collect as G3  # noqa: E402

plt = RD.plt
OUT = os.path.join(L.FIG, "table_hypotheses")
GROUPS = {"unseen6": ("Unseen-6 scenes (no method trained on them)", ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]),
          "all14": ("All 14 Macarons++ scenes", list(L.SCENE_ORDER))}
# fixed categorical order (Okabe-Ito, colour-blind safe) + line style / marker as a second encoding; Ours = the paper's orange
ARMS = [("D0", "Deterministic U-Net", dict(color="#0072B2", ls="--", marker="s")),
        ("G1", "Single generated sample", dict(color="#009E73", ls=":", marker="v")),
        ("MRAW", "Average of 4 samples (b, y)", dict(color="#CC79A7", ls="-.", marker="^")),
        ("MDEPTH", "Average of 4 samples (depth)", dict(color="#56B4E9", ls=(0, (5, 1.5)), marker="D")),
        ("G4", "Ours: 4 hypotheses + CDGR", dict(color=RD.STYLE["V3-T0-D1"]["color"], ls="-", marker="o"))]


def main():
    os.makedirs(OUT, exist_ok=True)
    _, curves = G3.outdoor()
    for gk, (gtitle, scenes) in GROUPS.items():
        Y = {a: np.stack([np.asarray(curves[a][s], float) for s in scenes]) for a, _, _ in ARMS}
        mean = {a: Y[a].mean(0) for a in Y}
        idx = np.random.default_rng(0).integers(0, len(scenes), (10000, len(scenes)))
        b = Y["G4"][idx].mean(1)
        lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
        x = np.arange(101)
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.15, 1]})
        for ax, (x0, x1) in zip(axs, ((0, 100), (50, 100))):
            if x1 - x0 == 100:                                       # band only on the full panel (scene spread, not method difference)
                ax.fill_between(x[x0:x1 + 1], lo[x0:x1 + 1], hi[x0:x1 + 1], color=ARMS[-1][2]["color"], alpha=0.13, lw=0)
            for a, name, st in ARMS:
                ours = a == "G4"
                ax.plot(x[x0:x1 + 1], mean[a][x0:x1 + 1], color=st["color"], ls=st["ls"], lw=2.2 if ours else 1.3,
                        marker=st["marker"], markevery=10 if x1 - x0 == 100 else 5, ms=4.5 if ours else 3.4,
                        markeredgecolor="white", markeredgewidth=0.6, label=name, zorder=5 if ours else 3)
            ax.set_xlim(x0, x1)
            ax.grid(alpha=0.25, lw=0.4)
            ax.set_xlabel("exploration step")
        axs[0].set_ylim(0, 1.0)
        axs[0].set_ylabel("coverage")
        lo_z = min(mean[a][50] for a, _, _ in ARMS) - 0.01
        hi_z = max(mean[a][x].max() for a, _, _ in ARMS for x in [slice(50, 101)]) + 0.006
        axs[1].set_ylim(lo_z, hi_z)
        axs[0].set_title(f"{gtitle}: steps 0–100", fontsize=8.5)
        axs[1].set_title("zoom: steps 50–100", fontsize=8.5)
        gap = (hi_z - lo_z) * 0.055                               # de-overlap the final-value labels (keep order, minimum spacing)
        order = sorted(ARMS, key=lambda r: mean[r[0]][100])
        ypos, prev = {}, -1e9
        for a, _, _ in order:
            ypos[a] = max(mean[a][100], prev + gap)
            prev = ypos[a]
        shift = max(0.0, ypos[order[-1][0]] - (hi_z - gap / 2))
        for a, name, st in ARMS:                                   # final values at the right edge of the zoom (text ink, not series colour)
            axs[1].annotate(f"{mean[a][100]:.3f}", xy=(100, mean[a][100]), xytext=(103.5, ypos[a] - shift), textcoords="data",
                            fontsize=6.8, va="center", color="#222222", fontweight="bold" if a == "G4" else "normal",
                            annotation_clip=False, arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.6, shrinkA=0, shrinkB=0))
        h, lb = axs[0].get_legend_handles_labels()
        fig.legend(h, lb, loc="upper center", ncol=3, fontsize=7.2, bbox_to_anchor=(0.5, 1.13), handlelength=2.6)
        fig.text(0.5, -0.03, f"Macarons++, {len(scenes)} scenes x 5 official starts, official coverage, scene-equal mean. Same network, data, "
                 "CDGR planner and protocol for every curve. Band (left): 95% scene-bootstrap CI of Ours, i.e. scene spread; method differences "
                 "are tested pairwise in the table.", ha="center", fontsize=6.6, color="#444444")
        fig.tight_layout()
        RD.save(fig, os.path.join(OUT, f"curves_{gk}"))
        L.write_csv(os.path.join(OUT, f"curves_{gk}.csv"), ["step"] + [n for _, n, _ in ARMS] + ["Ours ci95_lo", "Ours ci95_hi"],
                    [[t] + [f"{mean[a][t]:.5f}" for a, _, _ in ARMS] + [f"{lo[t]:.5f}", f"{hi[t]:.5f}"] for t in range(101)])
        print(gk, {n: round(float(mean[a][100]), 3) for a, n, _ in ARMS})


if __name__ == "__main__":
    main()


def three_panel(group="unseen6"):
    """Variant requested 2026-11-16: full 0-100 | zoom 0-40 | zoom 50-100 (Unseen-6). Same data and styling; end-value labels in both zooms."""
    _, curves = G3.outdoor()
    gtitle, scenes = GROUPS[group]
    Y = {a: np.stack([np.asarray(curves[a][s], float) for s in scenes]) for a, _, _ in ARMS}
    mean = {a: Y[a].mean(0) for a in Y}
    idx = np.random.default_rng(0).integers(0, len(scenes), (10000, len(scenes)))
    b = Y["G4"][idx].mean(1)
    lo, hi = np.percentile(b, 2.5, 0), np.percentile(b, 97.5, 0)
    x = np.arange(101)
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 2.8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    spans = ((0, 100, "steps 0–100"), (10, 40, "zoom: steps 10–40"), (50, 100, "zoom: steps 50–100"))
    for ax, (x0, x1, ttl) in zip(axs, spans):
        if (x0, x1) == (0, 100):
            ax.fill_between(x, lo, hi, color=ARMS[-1][2]["color"], alpha=0.13, lw=0)
        for a, name, st in ARMS:
            ours = a == "G4"
            ax.plot(x[x0:x1 + 1], mean[a][x0:x1 + 1], color=st["color"], ls=st["ls"], lw=2.2 if ours else 1.3, marker=st["marker"],
                    markevery=10 if x1 - x0 >= 100 else 5, ms=4.5 if ours else 3.4, markeredgecolor="white", markeredgewidth=0.6,
                    label=name, zorder=5 if ours else 3)
        if (x0, x1) == (0, 100):
            for z0, z1 in ((10, 40), (50, 100)):                 # mark the two zoom windows on the full panel
                ax.axvspan(z0, z1, color="#888888", alpha=0.06, lw=0, zorder=0)
        ax.set_xlim(x0, x1)
        ax.grid(alpha=0.25, lw=0.4)
        ax.set_xlabel("exploration step")
        ax.set_title(ttl, fontsize=8.5)
        if (x0, x1) != (0, 100):
            vmin = min(mean[a][x0:x1 + 1].min() for a, _, _ in ARMS) if x0 > 0 else 0.0
            vmax = max(mean[a][x0:x1 + 1].max() for a, _, _ in ARMS)
            pad = (vmax - vmin) * 0.04
            ax.set_ylim(vmin - (pad if x0 > 0 else 0), vmax + pad)
            gap = (vmax - vmin + 2 * pad) * 0.07
            order = sorted(ARMS, key=lambda r: mean[r[0]][x1])
            ypos, prev = {}, -1e9
            for a, _, _ in order:
                ypos[a] = max(mean[a][x1], prev + gap)
                prev = ypos[a]
            shift = max(0.0, ypos[order[-1][0]] - (vmax + pad - gap / 2))
            dx = (x1 - x0) * 0.07
            for a, name, st in ARMS:
                ax.annotate(f"{mean[a][x1]:.3f}", xy=(x1, mean[a][x1]), xytext=(x1 + dx, ypos[a] - shift), textcoords="data", fontsize=6.6,
                            va="center", color="#222222", fontweight="bold" if a == "G4" else "normal", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-", color=st["color"], lw=0.6, shrinkA=0, shrinkB=0))
    axs[0].set_ylim(0, 1.0)
    axs[0].set_ylabel("coverage")
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc="upper center", ncol=5, fontsize=7.0, bbox_to_anchor=(0.5, 1.07), handlelength=2.6)
    fig.text(0.5, -0.03, f"{gtitle}: Macarons++, {len(scenes)} scenes x 5 official starts, official coverage, scene-equal mean. Same network, "
             "data, CDGR planner and protocol for every curve. Band (left): 95% scene-bootstrap CI of Ours (scene spread); values at the right "
             "edge of each zoom = coverage at its last step.", ha="center", fontsize=6.4, color="#444444")
    fig.tight_layout(w_pad=2.2)
    RD.save(fig, os.path.join(OUT, f"curves_{group}_3panel"))
    print({n: (round(float(mean[a][20]), 3), round(float(mean[a][40]), 3)) for a, n, _ in ARMS})
