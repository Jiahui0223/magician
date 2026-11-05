"""Multi-View Coverage Reasoning Study -- figures (CPU, read-only).
Aggregate figures from results/summary.json + the per-state rows; case figures from the saved case states (real model predictions,
work_dirs/.../cases/<tag>_t<t>.npz written by mv_run.py --save) and the real renders of mv_render.py.
  figures/agg_rankings.png      Top-1 / regret of D-v2, K variants, unpaired, CVAR variants and the perfect-depth planner
  figures/agg_pairs.png         predicted vs true overlap of candidate pairs; c2 marginal gain predicted vs true
  figures/agg_consistency.png   cross-view inconsistency (paired / unpaired / distribution), GT attribution
  figures/case_<name>.jpg       one state: P1 + candidate cameras, real views, GT depth, K = 4 predicted depths, predicted vs true new surface"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(MV)))
ST = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "states")
CASES = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "cases")
FIG = os.path.join(MV, "figures")
INK, MUTED = "#222222", "#666666"
CAM = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=8)


def rows_all():
    rows = []
    for f in sorted(glob.glob(os.path.join(ST, "*.jsonl"))):
        rows += [json.loads(l) for l in open(f)]
    return rows


def agg_rankings(S):
    names = [("dv2", "D-v2 (K = 4, planner)"), ("one_step", "one-step only"), ("two_step_first1", "K = 1"), ("two_step_first2", "K = 2"),
             ("two_step_first8", "K = 8"), ("unpaired", "unpaired union"), ("cvar", "CVAR"), ("cvar_occ", "CVAR occlusion only"), ("cvar_free", "CVAR free-space only"),
             ("PD", "same value function,\nTRUE depth (two-step)")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (o, met, title) in zip(axs, (("official", "top1", "Top-1 (official one-step gain)"), ("official", "regret", "regret, official one-step gain (pp)"),
                                         ("geo2", "regret", "regret, geometric two-step coverage (pp)"))):
        x = np.arange(len(names))
        for i, (g, col) in enumerate((("t<20", "#56B4E9"), ("t>=20", "#0072B2"))):
            v = [S["perfect_depth"][g][f"gtvalue_two_step|{o}"][met] if n == "PD" else S["rank"][g][f"{n}|{o}"][met] for n, _ in names]
            ax.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=col, label=g.replace("<", " < ").replace(">=", " ≥ "), edgecolor="white", linewidth=0.6)
        ax.axvspan(len(names) - 1.5, len(names) - 0.5, color="#EEEEEE", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([n for _, n in names], rotation=35, ha="right", fontsize=7.5)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", alpha=0.25)
        style(ax)
    axs[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "agg_rankings.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def agg_pairs(rows):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
    po, to = [], []
    for r in rows:
        for p in r["pairs"]:
            if p["kind"] != "first":
                continue
            mn, mc = min(p["pred"]["VA"], p["pred"]["VB"]), min(p["cov"]["nA"], p["cov"]["nB"])
            if mn > 1e-9 and mc >= 20:
                po.append(p["pred"]["I"] / mn)
                to.append(p["cov"]["nI"] / mc)
    ax = axs[0]
    ax.scatter(to, po, s=5, alpha=0.35, color=CAM[0], linewidths=0)
    ax.plot([0, 1], [0, 1], color="#999999", lw=0.8, ls="--")
    ax.axhspan(0, 0.2, xmin=0.5, xmax=1.0, color="#D55E00", alpha=0.07)
    ax.text(0.98, 0.1, "predicted complementary,\ntruly redundant", fontsize=7.5, color="#D55E00", ha="right", va="center")
    ax.set_xlabel("true overlap of new coverage  |N_A ∩ N_B| / min(|N_A|, |N_B|)", fontsize=8, color=MUTED)
    ax.set_ylabel("predicted overlap  I / min(V_A, V_B)", fontsize=8, color=MUTED)
    ax.set_title(f"first-layer candidate pairs ({len(po)} pairs, both views cover >= 20 GT points)", fontsize=9, color=INK, loc="left")
    style(ax)
    pm, tm = [], []
    for r in rows:
        for p in r["pairs"]:
            if p["kind"] == "path":
                pm.append(p["pred"]["U"] - p["pred"]["VA"])
                tm.append(p["cov"]["pp_marg_B"])
    ax = axs[1]
    ax.scatter(tm, pm, s=5, alpha=0.3, color=CAM[1], linewidths=0)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("true marginal coverage of c2 after c1 (pp)", fontsize=8, color=MUTED)
    ax.set_ylabel("predicted marginal of c2 after c1 (weighted cells)", fontsize=8, color=MUTED)
    ax.set_title(f"the planner's own two-step paths ({len(pm)} (c1, c2) pairs)", fontsize=9, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "agg_pairs.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def agg_consistency(S):
    C = S["task_c"]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    labs = [("paired", "same\nsample index"), ("unpaired", "different\nsample index"), ("dist", "distribution\nlevel"), ("dist_novel", "distribution,\nnovel points only")]
    x = np.arange(len(labs))
    for i, (kind, col, nm) in enumerate((("first", CAM[0], "first-layer pairs"), ("path", CAM[1], "two-step paths"))):
        v = [C["all"][kind][l]["inconsistency"] for l, _ in labs]
        ax.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=col, label=nm, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n in labs], fontsize=8)
    ax.set_title("co-visible generated surface: share the other view sees through", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    style(ax)
    ax = axs[1]
    for i, kind in enumerate(("first", "path")):
        a = C["all"][kind]["gt_attribution_of_conflicts"]
        left = 0
        for (k, col, nm) in (("A_point_is_phantom", "#D55E00", "the projected point is not a true surface"), ("B_misses_a_true_surface", "#0072B2", "the other view misses a true surface"),
                             ("both_right_geometrically", "#999999", "both consistent with the GT (grid / tolerance)")):
            ax.barh(i, a[k], left=left, color=col, label=nm if i == 0 else None, height=0.55)
            if a[k] > 0.06:
                ax.text(left + a[k] / 2, i, f"{100 * a[k]:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += a[k]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["first-layer pairs", "two-step paths"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_title("who is wrong when two views disagree (offline GT)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "agg_consistency.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def case(name, c, row):
    z = np.load(os.path.join(CASES, f"{c['tag']}_t{c['t']}.npz"))
    rgbf = os.path.join(CASES, f"rgb_{c['tag']}_t{c['t']}.npz")
    rgb = np.load(rgbf) if os.path.exists(rgbf) else None
    keys1 = [tuple(k) for k in row["keys1"]]
    sk = lambda k: "_".join(map(str, k))  # noqa: E731
    sb = json.load(open(os.path.join(DATA, row["scene"], "settings.json")))["scene"]
    lo, hi = np.asarray(sb["x_min"]) * 10, np.asarray(sb["x_max"]) * 10
    show = [tuple(k) for k in c["show_keys"]]
    n = len(show)
    fig = plt.figure(figsize=(19, 3.0 * (n + 2)))
    gs = fig.add_gridspec(n + 2, 7, height_ratios=[1.5] + [1.0] * n + [1.5])
    # --- row 0: P1 (observed band) top view + cameras; bar chart of scores and truths
    ax = fig.add_subplot(gs[0, :3])
    band = z["band"].astype(np.float64)
    o, kM = z["origin"], float(z["kM"])
    pm = (band + 0.5) * 0.011 * kM + o                                         # z-up runtime
    py = pm @ np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])  # back to y-up (x, y, z)
    sel = np.random.default_rng(0).permutation(len(py))[:120000]
    ax.scatter(py[sel, 0], py[sel, 2], c=py[sel, 1], s=0.3, cmap="Greys", rasterized=True, linewidths=0)
    ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec="#00C389", lw=1.5))
    pick, best = row["rank"]["dv2"]["official"]["pick"], int(np.nanargmax(np.asarray(row["oracle_one_step_pp"], float)))
    for j, k in enumerate(keys1):
        cc, dd = z[f"c_{sk(k)}"], z[f"d_{sk(k)}"]
        dm = dd[len(dd) // 2 + 57]
        col = CAM[show.index(k)] if k in show else "#777777"
        ax.annotate("", xy=(cc[0] + 8 * dm[0], cc[2] + 8 * dm[2]), xytext=(cc[0], cc[2]), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.scatter(cc[0], cc[2], s=40, color=col, zorder=5, edgecolor="white", linewidths=0.6)
        tag = ("D-v2 pick " if j == pick else "") + ("best " if j == best else "")
        ax.annotate(f"{j}{(' ' + tag.strip()) if tag else ''}", (cc[0], cc[2]), textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK)
    ax.set_aspect("equal")
    ax.set_title("observed P1 surface (top view), ROI box, legal cameras", fontsize=9, color=INK, loc="left")
    style(ax)
    ax = fig.add_subplot(gs[0, 3:])
    x = np.arange(len(keys1))
    def nz(v):
        v = np.asarray(v, float)
        return v / max(np.nanmax(np.abs(v)), 1e-9)
    for i, (v, nm, col) in enumerate(((row["oracle_one_step_pp"], "true one-step gain (official)", "#999999"), (row["geo_two_step_pp"], "true two-step coverage (geometric)", "#555555"),
                                     (row["D"]["scores"]["dv2"], "D-v2 two-step score", CAM[0]), (row["D"]["scores"]["cvar"], "CVAR score", CAM[2]))):
        ax.bar(x + (i - 1.5) * 0.2, nz(v), width=0.19, color=col, label=nm)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{j}" for j in x], fontsize=8)
    ax.set_xlabel("legal camera", fontsize=8, color=MUTED)
    ax.set_title("truth and scores per legal camera (each scaled to its maximum)", fontsize=9, color=INK, loc="left")
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    style(ax)
    # --- rows 1..n: per shown camera: real view, GT depth, K = 4 predicted depths, novel rays
    vmax = 70.0
    for i, k in enumerate(show):
        tt = z[f"tt_{sk(k)}"].reshape(64, 114)
        t4 = z[f"t_{sk(k)}"].reshape(4, 64, 114)
        hu = z[f"hu_{sk(k)}"].reshape(4, 64, 114)
        nv = z[f"nov_{sk(k)}"].reshape(4, 64, 114)
        j = keys1.index(k) if k in keys1 else None
        lab = f"camera {j}" if j is not None else "child " + str(list(k))
        ax = fig.add_subplot(gs[1 + i, 0])
        if rgb is not None and f"rgb_{sk(k)}" in rgb.files:
            ax.imshow(rgb[f"rgb_{sk(k)}"])
        ax.set_title(f"{lab}: real view", fontsize=8.5, color=CAM[i], loc="left")
        ax.axis("off")
        ax = fig.add_subplot(gs[1 + i, 1])
        ax.imshow(np.where(np.isfinite(tt), tt, np.nan), cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title("true depth (GT, offline)", fontsize=8.5, color=INK, loc="left")
        ax.axis("off")
        for kk in range(4):
            ax = fig.add_subplot(gs[1 + i, 2 + kk])
            img = np.where(np.isfinite(t4[kk]), t4[kk], np.nan)
            ax.imshow(img, cmap="viridis", vmin=0, vmax=vmax)
            ax.contour(nv[kk].astype(float), levels=[0.5], colors="#FF2FB3", linewidths=0.6)
            ax.set_title(f"predicted depth, sample {kk}" + (" (pink = counted new)" if kk == 0 else ""), fontsize=8, color=INK, loc="left")
            ax.axis("off")
        ax = fig.add_subplot(gs[1 + i, 6])
        gen = hu.mean(0)
        ax.imshow(gen, cmap="magma", vmin=0, vmax=1)
        ax.set_title("share of the 4 samples that generate a surface", fontsize=8, color=INK, loc="left")
        ax.axis("off")
    # --- last row: predicted new surface vs true new coverage in world space (top view)
    ax1 = fig.add_subplot(gs[n + 1, :3])
    ax2 = fig.add_subplot(gs[n + 1, 3:6])
    gt = z["gt"].astype(np.float64)
    cov0 = np.zeros(len(gt), bool)
    cov0[z["covered0"]] = True
    ax2.scatter(gt[cov0, 0], gt[cov0, 2], s=0.2, color="#DDDDDD", rasterized=True, linewidths=0)
    ax2.scatter(gt[~cov0, 0], gt[~cov0, 2], s=0.2, color="#AAAAAA", rasterized=True, linewidths=0)
    cnt = np.zeros(len(gt), int)
    for i, k in enumerate(show):
        cnt[z[f"N_{sk(k)}"]] += 1
    for i, k in enumerate(show):
        cc, dd, t4, nv = z[f"c_{sk(k)}"], z[f"d_{sk(k)}"], z[f"t_{sk(k)}"], z[f"nov_{sk(k)}"]
        pts = np.concatenate([cc[None] + dd[nv[kk]] * t4[kk][nv[kk]][:, None] for kk in range(4)])
        if len(pts):
            ax1.scatter(pts[:, 0], pts[:, 2], s=1.2, color=CAM[i], alpha=0.35, rasterized=True, linewidths=0, label=f"{'camera ' + str(keys1.index(k)) if k in keys1 else 'child'}")
        N = z[f"N_{sk(k)}"]
        only = N[cnt[N] == 1]
        ax2.scatter(gt[only, 0], gt[only, 2], s=2.5, color=CAM[i], rasterized=True, linewidths=0)
    shared = np.flatnonzero(cnt >= 2)
    ax2.scatter(gt[shared, 0], gt[shared, 2], s=3.5, color="#111111", rasterized=True, linewidths=0, label="seen new by >= 2 of these cameras")
    for ax in (ax1, ax2):
        ax.add_patch(plt.Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2], fill=False, ec="#00C389", lw=1.2))
        ax.set_xlim(lo[0] - 3, hi[0] + 3)
        ax.set_ylim(lo[2] - 3, hi[2] + 3)
        ax.set_aspect("equal")
        style(ax)
    ax1.set_title("predicted NEW surface counted by the planner (all 4 samples; top view; colour = camera)", fontsize=9, color=INK, loc="left")
    ax1.legend(fontsize=7.5, frameon=False, markerscale=6)
    ax2.set_title("TRUE new coverage of the same cameras (GT target points; black = shared; grey = target, light = already covered)", fontsize=9, color=INK, loc="left")
    ax2.legend(fontsize=7.5, frameon=False, markerscale=4, loc="lower right")
    fig.suptitle(c["title"], fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"case_{name}.jpg"), dpi=130, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


def main():
    os.makedirs(FIG, exist_ok=True)
    S = json.load(open(os.path.join(MV, "results", "summary.json")))
    rows = rows_all()
    agg_rankings(S)
    agg_pairs(rows)
    agg_consistency(S)
    cf = os.path.join(MV, "results", "cases.json")
    if os.path.exists(cf):
        idx = {(r["tag"], r["t"]): r for r in rows}
        for c in json.load(open(cf))["cases"]:
            if os.path.exists(os.path.join(CASES, f"{c['tag']}_t{c['t']}.npz")):
                case(c["name"], c, idx[(c["tag"], c["t"])])


if __name__ == "__main__":
    main()
