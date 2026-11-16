"""Compact paper table + figure for "Why multiple generative hypotheses?" on Unseen-6 (2026-11-16, user request).
Unseen-6 = the 6 Macarons++ scenes no method was trained on (alhambra, neuschwanstein, eiffel, bridge, barts, sestino_museum; same
definition as fig3_v3). Arms (same network / data / CDGR planner / protocol; only the prediction head or the use of the K = 4 samples
changes): D0 deterministic U-Net, G1 single generated sample, M-raw / M-depth = 4 samples averaged into one map, G4 = Ours.
Curves come from experiment_G.../scripts/g3_collect.outdoor() (the validated G3 loader: replay == stored for the new arms, Table 1 /
Table 3 manifests for G1 / G4); scene = mean of the 5 official starts. Metrics AUC@t = trapz(C[0..t]) / t, Cov@t = C[t], Final = C[100].
Table: scene-equal means; † = paired scene bootstrap difference to Ours with 95 % CI excluding 0 (10,000, seed 0); bold = best.
Figure: two panels (Final coverage | AUC@100), mean over the 6 scenes with 95 % bootstrap CI of the mean (10,000, seed 0), per-scene
values as faint dots; Ours highlighted. Outputs fig/table_hypotheses/unseen6_table.{csv,md,tex,pdf,png} and unseen6_figure.{pdf,png}."""
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
UNSEEN = ["alhambra", "neuschwanstein", "eiffel", "bridge", "barts", "sestino_museum"]
ARMS = [("D0", "Deterministic U-Net", "1 (regressed mean)"), ("G1", "Single generated sample", "1"),
        ("MRAW", "Average of 4 samples (b, y)", "4 → 1"), ("MDEPTH", "Average of 4 samples (depth)", "4 → 1"),
        ("G4", "Ours: 4 hypotheses + CDGR", "4")]
MET = [("AUC@20", lambda c: np.trapz(c[:21]) / 20), ("Cov@20", lambda c: c[20]), ("AUC@100", lambda c: np.trapz(c[:101]) / 100),
       ("Final", lambda c: c[100])]


def boot_mean(x, n=10000):
    x = np.asarray(x)
    b = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    os.makedirs(OUT, exist_ok=True)
    _, curves = G3.outdoor()
    per = {a: {s: {m: float(f(np.asarray(curves[a][s]))) for m, f in MET} for s in UNSEEN} for a, _, _ in ARMS}
    mean = {a: {m: float(np.mean([per[a][s][m] for s in UNSEEN])) for m, _ in MET} for a, _, _ in ARMS}
    sig = {}
    for a, _, _ in ARMS:
        for m, _ in MET:
            if a == "G4":
                sig[(a, m)] = False
                continue
            d = np.array([per[a][s][m] - per["G4"][s][m] for s in UNSEEN])
            b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
            sig[(a, m)] = lo > 0 or hi < 0
    # cross-check against the G3 final table (AUC@20 / AUC@100 / Final on Unseen-6)
    import csv
    ft = {r["method"]: r for r in csv.DictReader(open(os.path.join(G3.OUT, "final_table_outdoor.csv"))) if r["group"] == "Unseen-6"}
    nm = {"D0": "D0", "G1": "G1", "MRAW": "M-raw", "MDEPTH": "M-depth", "G4": "Ours (G4)"}
    for a, _, _ in ARMS:
        for m in ("AUC@20", "AUC@100", "Final"):
            assert abs(mean[a][m] - float(ft[nm[a]][m])) < 1e-9, (a, m, mean[a][m], ft[nm[a]][m])
    # ---- table
    hdr = ["Prediction / combination", "Hypotheses to planner"] + [m + " ↑" for m, _ in MET]
    body, bold = [], set()
    for i, (a, name, hyp) in enumerate(ARMS):
        body.append([name, hyp] + [f"{mean[a][m]:.3f}" + ("†" if sig[(a, m)] else "") for m, _ in MET])
    for j, (m, _) in enumerate(MET):
        best = max(round(mean[a][m], 3) for a, _, _ in ARMS)
        bold |= {(i, 2 + j) for i, (a, _, _) in enumerate(ARMS) if round(mean[a][m], 3) == best}
    L.write_csv(os.path.join(OUT, "unseen6_table.csv"), ["Prediction / combination", "Hypotheses to planner"] + [m for m, _ in MET], body)
    note = ("Macarons++, 6 scenes no method was trained on (alhambra, neuschwanstein, eiffel, bridge, barts, sestino museum) x 5 official "
            "starts, 100 steps, official coverage. Same network, training data, CDGR planner and protocol in every row. † = difference to "
            "Ours with 95% paired scene-bootstrap CI excluding 0. Bold = best.")
    RD.table(os.path.join(OUT, "unseen6_table"), hdr, body, bold=bold, notes=[note], rules_after={len(ARMS) - 2})
    RD.tex(os.path.join(OUT, "unseen6_table.tex"), ["Prediction / combination", "Hypotheses to planner"] + [m + r" $\uparrow$" for m, _ in MET],
           body, bold=bold, rules_after={len(ARMS) - 2}, colspec="ll" + "c" * len(MET), label="tab:hypotheses_unseen6",
           caption="Why multiple generative hypotheses (unseen scenes). " + note,
           notes=["generated by experiments/cvpr_results/scripts/make_hypotheses_unseen6.py -- do not edit by hand (needs booktabs)"])
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join((f"**{c}**" if (i, j) in bold else c) for j, c in enumerate(r)) + " |" for i, r in enumerate(body)]
    open(os.path.join(OUT, "unseen6_table.md"), "w").write("\n".join(md) + "\n\n" + note + "\n")
    # ---- figure: two panels, horizontal dot + 95 % CI, per-scene faint dots, Ours highlighted
    ORANGE, GREY, LIGHT = RD.STYLE["V3-T0-D1"]["color"], "#555555", "#BBBBBB"
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    ys = np.arange(len(ARMS))[::-1]
    for ax, (m, title) in zip(axs, (("Final", "Final coverage (step 100)"), ("AUC@100", "Coverage AUC@100"))):
        for y, (a, name, _) in zip(ys, ARMS):
            col = ORANGE if a == "G4" else GREY
            vals = [per[a][s][m] for s in UNSEEN]
            ax.scatter(vals, np.full(len(vals), y) + np.linspace(-0.12, 0.12, len(vals)), s=9, color=LIGHT if a != "G4" else "#F2B48C",
                       zorder=1, linewidths=0)
            mu, lo, hi = boot_mean(vals)
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=2)
            ax.scatter([mu], [y], s=46 if a == "G4" else 30, color=col, zorder=3, edgecolors="white", linewidths=1.0)
            ax.text(hi + 0.002, y + 0.18, f"{mu:.3f}", fontsize=7.5, color="#222222" if a == "G4" else "#555555",
                    fontweight="bold" if a == "G4" else "normal", va="bottom", ha="left")
        ax.axvline(mean["G4"][m], color=ORANGE, lw=0.8, ls=":", zorder=0)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([name for _, name, _ in ARMS], fontsize=8)
    for t, (a, _, _) in zip(axs[0].get_yticklabels(), ARMS):
        if a == "G4":
            t.set_fontweight("bold"); t.set_color(ORANGE)
    lo_all = min(min(per[a][s][m] for a, _, _ in ARMS for s in UNSEEN) for m in ("Final", "AUC@100"))
    axs[0].set_xlim(min(per[a][s]["Final"] for a, _, _ in ARMS for s in UNSEEN) - 0.01, 1.0)
    axs[1].set_xlim(min(per[a][s]["AUC@100"] for a, _, _ in ARMS for s in UNSEEN) - 0.01,
                    max(per[a][s]["AUC@100"] for a, _, _ in ARMS for s in UNSEEN) + 0.03)
    fig.text(0.5, -0.04, "Unseen-6 scenes (no method trained on them), 5 starts each; dot = mean, bar = 95% bootstrap CI, light dots = scenes",
             ha="center", fontsize=7, color="#444444")
    fig.tight_layout()
    RD.save(fig, os.path.join(OUT, "unseen6_figure_absolute"))
    # ---- figure (main): paired difference to Ours, same scene, 95 % paired scene-bootstrap CI; absolute value annotated
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.3), sharey=True)
    others = [x for x in ARMS if x[0] != "G4"]
    ys = np.arange(len(others))[::-1]
    for ax, (m, title) in zip(axs, (("Final", "Final coverage: variant − Ours"), ("AUC@100", "Coverage AUC@100: variant − Ours"))):
        for y, (a, name, _) in zip(ys, others):
            d = np.array([per[a][s][m] - per["G4"][s][m] for s in UNSEEN])
            b = d[np.random.default_rng(0).integers(0, len(d), (10000, len(d)))].mean(1)
            lo, hi = np.percentile(b, 2.5), np.percentile(b, 97.5)
            col = GREY if hi >= 0 else "#333333"
            ax.plot([lo, hi], [y, y], color=col, lw=1.6, zorder=2)
            ax.scatter([d.mean()], [y], s=34, color=col, zorder=3, edgecolors="white", linewidths=1.0)
            ax.text(max(hi, 0.0) + (0.003 if m == "Final" else 0.0015), y, f"{d.mean():+.3f}" + ("†" if (lo > 0 or hi < 0) else "") + f"  ({mean[a][m]:.3f})",
                    fontsize=7.3, color="#333333", va="center", ha="left")
        ax.axvline(0, color=ORANGE, lw=1.6, zorder=1)
        ax.text(0, len(others) - 0.35, f" Ours = {mean['G4'][m]:.3f}", color=ORANGE, fontsize=7.8, fontweight="bold", va="bottom", ha="left")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", alpha=0.25, lw=0.4)
        ax.tick_params(axis="y", length=0)
    axs[0].set_yticks(ys)
    axs[0].set_yticklabels([name for _, name, _ in others], fontsize=8)
    axs[0].set_xlim(-0.08, 0.04)
    axs[1].set_xlim(-0.03, 0.035)
    axs[0].set_ylim(-0.6, len(others) - 0.1)
    fig.text(0.5, -0.05, "Unseen-6 scenes (no method trained on them) x 5 starts. Left of the orange line = worse than Ours. "
             "Dot = mean paired difference, bar = 95% scene-bootstrap CI, † = CI excludes 0; value of the variant in brackets.",
             ha="center", fontsize=6.8, color="#444444")
    fig.tight_layout()
    RD.save(fig, os.path.join(OUT, "unseen6_figure"))
    print("\n".join(md))


if __name__ == "__main__":
    main()
