"""CVPR overnight -- Fig. 3 (coverage and cumulative AUC curves) and Fig. 5 (per-scene AUC@20 radar + paired difference bars).
CPU only. python make_figures.py      Reads the same validated canonical records as make_tables.py; no smoothing, no cummax.
Only methods with all 70 episodes are drawn; the others are listed in fig3/curve_methods.csv with their completion count."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402
import render as RD  # noqa: E402
from make_tables import valid  # noqa: E402

plt = RD.plt
ORDER = ["RANDOM", "MACARONS", "MAG-B", "V3-T0-D1"]            # drawing order (ours last = on top); SCONE / FisherRF are BLOCKED


def curves():
    data, mc, ci, info = {}, {}, {}, []
    for m in L.TABLE1:
        E = valid(m)[0]
        info.append([m, L.METHODS[m]["display"], "BLOCKED" if L.METHODS[m].get("blocked") else ("COMPLETE" if len(E) == 70 else "PARTIAL"), len(E), 70, int(len(E) == 70)])
        if len(E) != 70:
            continue
        data[m] = E
        mean, per = L.mean_curve(E, L.SCENE_ORDER)
        P = np.stack([per[s] for s in L.SCENE_ORDER])                                   # [14, 101] scene-mean curves
        rng = np.random.default_rng(0)
        B = P[rng.integers(0, 14, (10000, 14))].mean(1)
        A = np.stack([[L.auc(p, T) for T in range(1, 101)] for p in P])                 # [14, 100] per-scene AUC@T
        BA = A[rng.integers(0, 14, (10000, 14))].mean(1)
        mc[m] = {"cov": mean, "auc": A.mean(0)}
        ci[m] = {"cov": (np.percentile(B, 2.5, 0), np.percentile(B, 97.5, 0)), "auc": (np.percentile(BA, 2.5, 0), np.percentile(BA, 97.5, 0))}
    return data, mc, ci, info


def panel_cov(ax, mc, ci, band, inset=True):
    for m in [x for x in ORDER if x in mc]:
        st = RD.STYLE[m]
        if band:
            ax.fill_between(np.arange(101), ci[m]["cov"][0], ci[m]["cov"][1], color=st["color"], alpha=0.14, lw=0)
        ax.plot(np.arange(101), mc[m]["cov"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.0); ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Exploration step"); ax.set_ylabel("Surface coverage (official)")
    ax.grid(color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)
    if inset:
        ia = ax.inset_axes([0.47, 0.09, 0.50, 0.50])
        for m in [x for x in ORDER if x in mc]:
            st = RD.STYLE[m]
            ia.plot(np.arange(21), mc[m]["cov"][:21], color=st["color"], ls=st["ls"], lw=st["lw"] * 0.9)
        ia.set_xlim(0, 20); ia.set_xticks([0, 5, 10, 15, 20]); ia.set_ylim(0, max(float(mc[m]["cov"][20]) for m in mc) * 1.08)
        ia.tick_params(labelsize=6.5, length=2); ia.grid(color="#E3E3E3", lw=0.4); ia.set_title("steps 0-20", fontsize=7, pad=2)
        for sp in ("top", "right"):
            ia.spines[sp].set_visible(True)
        for sp in ia.spines.values():
            sp.set_linewidth(0.5)


def panel_auc(ax, mc, ci, band):
    T = np.arange(1, 101)
    for m in [x for x in ORDER if x in mc]:
        st = RD.STYLE[m]
        if band:
            ax.fill_between(T, ci[m]["auc"][0], ci[m]["auc"][1], color=st["color"], alpha=0.14, lw=0)
        ax.plot(T, mc[m]["auc"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
    for t in (20, 50, 100):
        ax.axvline(t, color="#BBBBBB", lw=0.5, ls=(0, (2, 2)), zorder=0)
    ax.set_xlim(1, 100); ax.set_ylim(0, 0.85); ax.set_xticks([1, 20, 40, 60, 80, 100])
    ax.set_xlabel("Horizon $T$ (steps)"); ax.set_ylabel("Cumulative AUC@$T$")
    ax.grid(color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)


def fig3():
    out = os.path.join(L.FIG, "fig3")
    data, mc, ci, info = curves()
    L.write_csv(os.path.join(out, "curve_methods.csv"), ["method_id", "method", "status", "n_completed", "n_expected", "drawn"], info)
    rows = [[m, t, repr(float(mc[m]["cov"][t])), repr(float(ci[m]["cov"][0][t])), repr(float(ci[m]["cov"][1][t])), "" if t == 0 else repr(float(mc[m]["auc"][t - 1])),
             "" if t == 0 else repr(float(ci[m]["auc"][0][t - 1])), "" if t == 0 else repr(float(ci[m]["auc"][1][t - 1]))] for m in mc for t in range(101)]
    L.write_csv(os.path.join(out, "fig3_data.csv"), ["method_id", "step", "mean_coverage", "cov_ci95_lo", "cov_ci95_hi", "AUC@step", "auc_ci95_lo", "auc_ci95_hi"], rows)
    for name, band in (("fig3_combined", False), ("fig3_combined_ci", True)):
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"wspace": 0.28})
        panel_cov(axs[0], mc, ci, band); panel_auc(axs[1], mc, ci, band)
        axs[0].set_title("(a) Coverage vs. step", fontsize=9); axs[1].set_title("(b) Cumulative AUC@$T$", fontsize=9)
        h, lab = axs[1].get_legend_handles_labels()
        axs[1].legend(h[::-1], lab[::-1], fontsize=7.5, loc="lower right", handlelength=2.6)
        RD.save(fig, os.path.join(out, name))
    for name, fn in (("coverage_curve", panel_cov), ("auc_curve", panel_auc)):
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
        fn(ax, mc, ci, False)
        h, lab = ax.get_legend_handles_labels()
        ax.legend(h[::-1], lab[::-1], fontsize=7.5, loc="upper left" if name == "coverage_curve" else "lower right", handlelength=2.6)
        RD.save(fig, os.path.join(out, name))
    return info, {m: (float(mc[m]["cov"][20]), float(mc[m]["auc"][19]), float(mc[m]["auc"][99]), int(np.argmax(mc[m]["cov"] >= 0.5)) if (mc[m]["cov"] >= 0.5).any() else None) for m in mc}


def fig5():
    out = os.path.join(L.FIG, "fig5")
    meth = [m for m in ("MACARONS", "MAG-B", "V3-T0-D1") if len(valid(m)[0]) == 70]
    E = {m: valid(m)[0] for m in meth}
    tab = {m: L.scene_table(E[m], ["AUC@20"]) for m in meth}
    rows = []
    for s in L.SCENE_ORDER:
        d = [L.auc(E["V3-T0-D1"][(s, k)]["coverage_official"], 20) - L.auc(E["MAG-B"][(s, k)]["coverage_official"], 20) for k in range(5)]
        rows.append([s, L.NAME[s], L.SPLIT[s]] + [repr(tab[m][s]["AUC@20"]) for m in meth] + [repr(float(np.mean(d))), repr(float(np.std(d, ddof=1) / np.sqrt(5))), int(sum(x > 0 for x in d))])
    L.write_csv(os.path.join(out, "per_scene_auc20.csv"), ["scene_id", "official_name", "split"] + [f"AUC@20_{m}" for m in meth] + ["delta_ours_minus_magician", "delta_se_over_5_paired_starts", "starts_won_of_5"], rows)
    short = {s: L.NAME[s].replace(" Castle", "").replace(" Cathedral", "").replace(" Palace", "").replace("Statue of ", "").replace("Christ the ", "").replace("Manhattan ", "").replace(" Museum", "").replace(" Tower", "") for s in L.SCENE_ORDER}
    # ---- radar: 14 axes, one common 0..1 radial scale, frozen scene order
    n = len(L.SCENE_ORDER)
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    fig = plt.figure(figsize=(4.3, 4.3))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
    for m in meth:
        st = RD.STYLE[m]
        v = [tab[m][s]["AUC@20"] for s in L.SCENE_ORDER]
        ax.plot(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], ls=st["ls"], lw=st["lw"], marker=st["marker"], ms=3.2, label=st["label"])
        if m == "V3-T0-D1":
            ax.fill(np.r_[ang, ang[0]], np.r_[v, v[0]], color=st["color"], alpha=0.10, lw=0)
    ax.set_ylim(0, 1.0); ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=6.5, color="#555555")
    ax.set_rlabel_position(360.0 / n / 2); ax.set_xticks(ang); ax.set_xticklabels([])      # radial tick labels between the first two axes
    ax.grid(color="#CFCFCF", lw=0.5); ax.spines["polar"].set_linewidth(0.6)
    for a, s in zip(ang, L.SCENE_ORDER):
        ha = "center" if abs(np.sin(a)) < 0.15 else ("left" if np.sin(a) > 0 else "right")
        ax.text(a, 1.09, f"{short[s]}\n({RD.SPLIT_TAG[L.SPLIT[s]]})", ha=ha, va="center", fontsize=7, color=RD.SPLIT_COLOR[L.SPLIT[s]], linespacing=0.95)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=len(meth), handlelength=2.4)
    ax.set_title("AUC@20 per scene (mean of 5 starts; common 0-1 scale)", fontsize=8.5, pad=30)
    RD.save(fig, os.path.join(out, "per_scene_radar"))
    # ---- sorted paired difference bars
    o = sorted(range(n), key=lambda i: float(rows[i][-3]))
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    for y, i in enumerate(o):
        s = L.SCENE_ORDER[i]
        d, se = float(rows[i][-3]), float(rows[i][-2])
        ax.barh(y, d, height=0.66, color=RD.SPLIT_COLOR[L.SPLIT[s]], alpha=0.85, lw=0)
        ax.errorbar(d, y, xerr=se, color="#222222", lw=0.8, capsize=1.8)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(n)); ax.set_yticklabels([f"{short[L.SCENE_ORDER[i]]} ({RD.SPLIT_TAG[L.SPLIT[L.SCENE_ORDER[i]]]})" for i in o], fontsize=7.5)
    for t, i in zip(ax.get_yticklabels(), o):
        t.set_color(RD.SPLIT_COLOR[L.SPLIT[L.SCENE_ORDER[i]]])
    ax.set_xlabel("$\\Delta$AUC@20  (Ours $-$ MAGICIAN)"); ax.grid(axis="x", color="#E3E3E3", lw=0.5); ax.set_axisbelow(True)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RD.SPLIT_COLOR[k], alpha=0.85, label=RD.SPLIT_TAG[k]) for k in ("train", "dev", "test")], fontsize=7, loc="lower right", title="split of our model", title_fontsize=6.5)
    RD.save(fig, os.path.join(out, "per_scene_delta"))
    return meth, rows


if __name__ == "__main__":
    info, pts = fig3()
    print("FIG 3 methods:", [(r[0], r[2], f"{r[3]}/70") for r in info])
    for m, v in pts.items():
        print(f"   {m}: Cov@20 {v[0]:.3f} AUC@20 {v[1]:.3f} AUC@100 {v[2]:.3f} first step with mean coverage >= 0.5: {v[3]}")
    meth, rows = fig5()
    print("FIG 5 methods:", meth)
    for r in rows:
        print("   ", r[0].ljust(15), r[2].ljust(6), " ".join(f"{float(x):.3f}" for x in r[3:3 + len(meth)]), "| delta %+.3f +- %.3f, starts won %s/5" % (float(r[-3]), float(r[-2]), r[-1]))
