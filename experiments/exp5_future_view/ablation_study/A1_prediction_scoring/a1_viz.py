"""EXP5 Ablation A1 figures (seen2scene env, CPU). Every number drawn is read from results/a1_summary.json,
results/a1_states_<dataset>.csv or the per-state arrays; figure annotations are dumped to figures/annotations.json for
the Stage-5 check.
  fig1_overview        Spearman / Top-1 / regret / pairwise of V2O, V2M, GTD, random (scene bootstrap CI)
  fig2_decomposition   pred-correct x GTD-correct tables (prediction vs scoring failures)
  fig3_pred_vs_rank    per scene: fair ES and Brier vs regret, arrow V2O -> V2M; state-level paired changes
  fig4_scenes          per-scene regret and Top-1
  fig5_strata          regret by exploration phase / coverage / max gain
  fig6_case_<dataset>_<case>  same-state candidate ranking (four scores) + top view of the candidates
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ABL, "common"))
import ab_common as A  # noqa: E402
import ab_style as ST  # noqa: E402

import numpy as np  # noqa: E402

plt = ST.plt
RES, FIG = os.path.join(HERE, "results"), os.path.join(HERE, "figures")
DS = {"indoor": "MP3D (5 val levels)", "outdoor": "Macarons++ (Test-3 + Dev-3)"}
ANN = {}


def rows(dataset):
    return list(csv.DictReader(open(os.path.join(RES, f"a1_states_{dataset}.csv"))))


def fv(x):
    return float(x) if x not in ("", None, "None") else None


def fig1(S):
    mets = (("spearman", "Spearman vs oracle gain ↑"), ("top1", "Top-1 (tie-aware) ↑"), ("pairwise", "Pairwise accuracy ↑"),
            ("regret", "Oracle regret (pp) ↓"))
    sc = ("V2O", "V2M", "GTD", "Random")
    fig, axs = plt.subplots(2, 4, figsize=(13, 5.2))
    for i, d in enumerate(("indoor", "outdoor")):
        for j, (k, lab) in enumerate(mets):
            ax = axs[i, j]
            for y, m in enumerate(sc):
                b = S[d]["scorers"][m][k]
                if b["mean"] is None:
                    continue
                ax.plot([b["ci95"][0], b["ci95"][1]], [y, y], color=ST.COL[m], lw=2, solid_capstyle="round")
                ax.scatter([b["mean"]], [y], s=46, color=ST.COL[m], edgecolor=ST.INK if m == "V2M" else "white", zorder=3)
                ax.annotate(f"{b['mean']:.3f}", (b["mean"], y), xytext=(0, 7), textcoords="offset points", ha="center",
                            fontsize=8, color=ST.INK2)
                ANN[f"fig1|{d}|{m}|{k}"] = b["mean"]
            ax.set_yticks(range(len(sc)))
            ax.set_yticklabels([ST.LAB[m] for m in sc] if j == 0 else [])
            ax.set_ylim(-0.6, len(sc) - 0.4)
            ax.invert_yaxis()
            ST.clean(ax, "x")
            if i == 0:
                ax.set_title(lab, loc="left")
        axs[i, 0].set_ylabel(DS[d], fontsize=9.5, color=ST.INK)
    fig.suptitle("A1: one-step candidate ranking on the same fixed states (dot = mean of scene means, bar = 95% scene-bootstrap CI)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_overview.png"))
    plt.close(fig)


def fig2():
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for i, d in enumerate(("indoor", "outdoor")):
        R = rows(d)
        by = {}
        for r in R:
            by.setdefault(r["sid"], {})[r["scorer"]] = r
        for j, m in enumerate(("V2O", "V2M")):
            ax = axs[2 * i + j]
            M = np.zeros((2, 2))
            for v in by.values():
                a = int(fv(v[m]["top1"]) == 1)
                b = int(fv(v["GTD"]["top1"]) == 1)
                M[1 - a, 1 - b] += 1
            P = M / M.sum()
            ax.imshow(P, cmap=ST.SEQ, vmin=0, vmax=1)
            names = [["both correct", "pred. correct,\nGTD wrong"], ["prediction failure\n(GTD correct)", "both wrong\n(scoring failure)"]]
            for a in range(2):
                for b in range(2):
                    ax.text(b, a, f"{names[a][b]}\n{int(M[a, b])} ({100 * P[a, b]:.0f}%)", ha="center", va="center", fontsize=8.2,
                            color="white" if P[a, b] > 0.55 else ST.INK)
                    ANN[f"fig2|{d}|{m}|{a}{b}"] = int(M[a, b])
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["GTD Top-1", "GTD wrong"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels([f"{m} Top-1", f"{m} wrong"])
            ax.set_title(f"{DS[d].split(' (')[0]} — {ST.LAB[m]}", loc="left")
    fig.suptitle("A1: is a wrong choice caused by the prediction or by the score?  (states; Top-1 tie-aware, tol 0.01 pp)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_decomposition.png"))
    plt.close(fig)


def fig3(S):
    fig, axs = plt.subplots(2, 3, figsize=(13, 7.4))
    for i, d in enumerate(("indoor", "outdoor")):
        ps = S[d]["per_scene"]
        for j, k in enumerate(("fair_es", "brier")):
            ax = axs[i, j]
            for s in sorted(ps["V2O"][k]):
                x0, x1 = ps["V2O"][k][s], ps["V2M"][k][s]
                y0, y1 = ps["V2O"]["regret"][s], ps["V2M"]["regret"][s]
                ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color=ST.MUTED, lw=1.2))
                ax.scatter([x0], [y0], s=40, color=ST.COL["V2O"], zorder=3)
                ax.scatter([x1], [y1], s=40, color=ST.COL["V2M"], edgecolor=ST.INK, zorder=3)
                ax.scatter([x1], [ps["GTD"]["regret"][s]], s=34, marker="_", color=ST.COL["GTD"], lw=2.2, zorder=3)
                ax.annotate(s.replace("_L", " L"), (x1, y1), xytext=(4, -10), textcoords="offset points", fontsize=7.5, color=ST.INK2)
                ANN[f"fig3|{d}|{s}|{k}"] = [x0, y0, x1, y1]
            ax.set_xlabel({"fair_es": "fair ES (unknown-arclength voxels) ↓", "brier": "hit Brier ↓"}[k])
            ax.set_ylabel("oracle regret (pp) ↓")
            ST.clean(ax, "both")
            ax.set_title(f"{DS[d]}: per scene, V2O → V2M", loc="left")
        ax = axs[i, 2]
        R = rows(d)
        by = {}
        for r in R:
            by.setdefault(r["sid"], {})[r["scorer"]] = r
        dx = np.asarray([fv(v["V2M"]["fair_es"]) - fv(v["V2O"]["fair_es"]) for v in by.values()
                         if fv(v["V2M"]["fair_es"]) is not None and fv(v["V2O"]["fair_es"]) is not None])
        dy = np.asarray([fv(v["V2M"]["regret"]) - fv(v["V2O"]["regret"]) for v in by.values()
                         if fv(v["V2M"]["fair_es"]) is not None and fv(v["V2O"]["fair_es"]) is not None])
        ax.scatter(dx, dy, s=9, color=ST.COL["V2M"], alpha=0.45, lw=0)
        q = np.quantile(dx, np.linspace(0, 1, 7))
        mx = [(dx[(dx >= a) & (dx <= b)].mean(), dy[(dx >= a) & (dx <= b)].mean()) for a, b in zip(q[:-1], q[1:])]
        ax.plot(*zip(*mx), color=ST.INK, lw=2, marker="o", ms=4, label="mean per sextile of ΔES")
        ax.axhline(0, color=ST.MUTED, lw=0.8)
        ax.axvline(0, color=ST.MUTED, lw=0.8)
        ax.set_xlabel("Δ fair ES (V2M − V2O), per state")
        ax.set_ylabel("Δ regret (pp), per state")
        ax.legend(loc="upper left", fontsize=8)
        ST.clean(ax, None)
        ax.set_title(f"{DS[d]}: paired per state ({len(dx)})", loc="left")
        ANN[f"fig3|{d}|state_pairs"] = int(len(dx))
    fig.suptitle("A1: does a better prediction give a lower regret?  (orange = V2-Mix, light = V2 Original, green tick = GT depth + DistP1)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_pred_vs_rank.png"))
    plt.close(fig)


def fig4(S):
    fig, axs = plt.subplots(1, 4, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1, 1]})
    for i, d in enumerate(("indoor", "outdoor")):
        ps = S[d]["per_scene"]
        scenes = sorted(ps["GTD"]["regret"], key=lambda s: (A.SPLIT_OF.get(s, ""), s))
        for j, k in enumerate(("regret", "top1")):
            ax = axs[2 * i + j]
            for y, s in enumerate(scenes):
                vals = [ps[m][k][s] for m in ("V2O", "V2M", "GTD", "Random")]
                ax.plot([min(vals), max(vals)], [y, y], color=ST.GRID, lw=1.2, zorder=1)
                for m, v in zip(("V2O", "V2M", "GTD", "Random"), vals):
                    ax.scatter([v], [y], s=34, color=ST.COL[m], edgecolor=ST.INK if m == "V2M" else "white", zorder=3,
                               label=ST.LAB[m] if y == 0 else None)
                    ANN[f"fig4|{d}|{s}|{m}|{k}"] = v
            lab = [s.replace("_L", " L") + (f" ({A.SPLIT_OF[s]})" if s in A.SPLIT_OF else "") for s in scenes]
            ax.set_yticks(range(len(scenes)))
            ax.set_yticklabels(lab if j == 0 else [])
            ax.invert_yaxis()
            ST.clean(ax, "x")
            ax.set_title(f"{DS[d].split(' (')[0]}: {'regret (pp) ↓' if k == 'regret' else 'Top-1 ↑'}", loc="left")
    axs[0].legend(loc="lower center", bbox_to_anchor=(2.2, -0.32), ncol=4, fontsize=8.5)
    fig.suptitle("A1: per scene (mean over episodes of the mean over states)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_scenes.png"))
    plt.close(fig)


def fig5(S):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4))
    for i, d in enumerate(("indoor", "outdoor")):
        for j, (nm, order) in enumerate((("t_bin", ["early", "mid", "late"]), ("coverage", ["low", "mid", "high"]),
                                         ("max_gain", ["low", "mid", "high"]))):
            ax = axs[i, j]
            st = S[d]["strata"][nm]
            for m in ("V2O", "V2M", "GTD", "Random"):
                ys = [st[o][m]["nregret"] if o in st else None for o in order]
                ax.plot(range(3), ys, color=ST.COL[m], lw=2, marker="o", ms=5, label=ST.LAB[m])
                ANN[f"fig5|{d}|{nm}|{m}"] = ys
            n = [st[o]["GTD"]["states"] if o in st else 0 for o in order]
            ax.set_xticks(range(3))
            ax.set_xticklabels([f"{o}\n({c} states)" for o, c in zip(order, n)])
            ax.set_ylabel("normalised regret ↓")
            ST.clean(ax)
            ax.set_title(f"{DS[d].split(' (')[0]}: by {dict(t_bin='exploration phase', coverage='current coverage', max_gain='best one-step gain')[nm]}", loc="left")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("A1: normalised regret (regret / best gain) by stratum (state means; tercile bins per dataset)", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_strata.png"))
    plt.close(fig)


def topview(ax, dataset, st, C, picks, gain):
    import numpy as _np
    rec = _np.load(os.path.join(A.REPO, st["sources"][0]["svc"]))
    pts = []
    for t in range(st["t"] + 1):
        p = os.path.join(os.path.dirname(os.path.join(A.REPO, st["sources"][0]["svc"])),
                         os.path.basename(st["sources"][0]["svc"]).replace(f"t{st['t']}.npz", f"t{t}.npz"))
        pts.append(_np.load(p)["sample_points"][::6])
    P = _np.concatenate(pts)
    zup = lambda q: _np.stack([q[..., 0], -q[..., 2], q[..., 1]], -1) / (7.0 if dataset == "indoor" else 1.0)  # noqa: E731
    P = zup(P)
    c = zup(rec["legal_center"])
    cur = zup(rec["cam_center"][None])[0]
    lo, hi = c[:, :2].min(0) - 2.0 * (1 if dataset == "indoor" else 8), c[:, :2].max(0) + 2.0 * (1 if dataset == "indoor" else 8)
    m = _np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), 1)
    ax.scatter(P[m, 0], P[m, 1], s=0.4, color=ST.KNOWN_GRAY, lw=0, rasterized=True)
    fwd = _np.stack([zup(R[:, 2][None])[0] * (7.0 if dataset == "indoor" else 1.0) for R in rec["legal_R"]])
    L = 0.35 if dataset == "indoor" else 3.0
    best = set(int(j) for j in C if gain[j] >= gain[C].max() - A.TIE_TOL * 100)
    for j in C:
        f = fwd[j, :2] / max(1e-9, _np.linalg.norm(fwd[j, :2]))
        col, lw = (ST.INK, 2.2) if j in best else ((ST.COL["V2M"], 2.2) if j == picks["V2M"] else
                                                   ((ST.COL["GTD"], 2.2) if j == picks["GTD"] else (ST.MUTED, 1)))
        ax.annotate("", c[j, :2] + L * f, c[j, :2], arrowprops=dict(arrowstyle="->", color=col, lw=lw))
        ax.text(c[j, 0] + 1.35 * L * f[0], c[j, 1] + 1.35 * L * f[1], str(j), fontsize=7, color=ST.INK, ha="center", va="center")
    ax.scatter([cur[0]], [cur[1]], s=50, marker="o", facecolor="white", edgecolor=ST.INK, zorder=4, label="current camera")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("top view (black = oracle best,\norange = V2M pick, green = GTD pick)", loc="left", fontsize=8.5)


def fig6(cases):
    import ab_load as AL
    for d in ("indoor", "outdoor"):
        sts = {r["sid"]: r for r in AL.load_states(d)}
        man = {s["sid"]: s for s in A.load_manifest(d)["states"]}
        for nm, c in cases[d].items():
            if c["sid"] is None:
                continue
            r = sts[c["sid"]]
            C = r["C"]
            order = C[np.argsort(-r["gain"][C], kind="stable")]
            fig = plt.figure(figsize=(13, 3.9))
            gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 1.25])
            for k, m in enumerate(("Oracle", "GTD", "V2M", "V2O")):
                ax = fig.add_subplot(gs[0, k])
                v = r["gain"][order] if m == "Oracle" else r["scores"][m][order]
                vn = v / v.max() if v.max() > 0 else v
                pk = None if m == "Oracle" else r["picks"][m]
                cols = [ST.COL[m] if (j == pk or (m == "Oracle" and r["gain"][j] >= r["gain"][C].max() - A.TIE_TOL * 100)) else ST.GRID
                        for j in order]
                ax.barh(range(len(order)), vn, color=cols, height=0.7)
                for y, (j, x) in enumerate(zip(order, v)):
                    ax.text(vn[y] + 0.02, y, f"{x:.2f}" if m == "Oracle" else f"{x:.0f}", va="center", fontsize=7.5, color=ST.INK2)
                ax.set_yticks(range(len(order)))
                ax.set_yticklabels([f"j={j}" for j in order] if k == 0 else [])
                ax.invert_yaxis()
                ax.set_xlim(0, 1.3)
                ax.set_xticks([])
                ST.clean(ax, None)
                ax.set_title({"Oracle": "oracle gain (pp)", "GTD": "GT depth DistP1", "V2M": "V2-Mix DistP1 (K=4)",
                              "V2O": "V2 Original DistP1 (K=4)"}[m], loc="left", color=ST.INK)
                ANN[f"fig6|{d}|{nm}|{m}"] = [float(x) for x in v]
            ax = fig.add_subplot(gs[0, 4])
            topview(ax, d, man[c["sid"]], C, r["picks"], r["gain"])
            g = r["gain"][C]
            fig.suptitle(f"{DS[d].split(' (')[0]} — case '{nm}': {r['scene']} start {r['start']}, t = {r['t']} (source {r['src']}); "
                         f"colored bar = the scorer's pick (oracle: best); regret V2M {g.max() - r['gain'][r['picks']['V2M']]:.2f} pp, "
                         f"GTD {g.max() - r['gain'][r['picks']['GTD']]:.2f} pp, V2O {g.max() - r['gain'][r['picks']['V2O']]:.2f} pp",
                         x=0.01, ha="left", fontsize=9.5)
            fig.tight_layout()
            fig.savefig(os.path.join(FIG, f"fig6_case_{d}_{nm}.png"))
            plt.close(fig)


def main():
    ST.setup()
    os.makedirs(FIG, exist_ok=True)
    S = json.load(open(os.path.join(RES, "a1_summary.json")))
    fig1(S)
    fig2()
    fig3(S)
    fig4(S)
    fig5(S)
    cp = os.path.join(ABL, "A2_stochastic_future_views", "results", "cases.json")
    if os.path.exists(cp):
        fig6(json.load(open(cp)))
    json.dump(ANN, open(os.path.join(FIG, "annotations.json"), "w"), indent=1, default=float)
    print("A1 figures:", sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
