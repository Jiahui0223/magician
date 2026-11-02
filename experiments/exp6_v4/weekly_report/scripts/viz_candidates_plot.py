"""Plot one V4 planning step (from viz_extract.py bundles) and the all-state summary (CPU).

Per state, two figures:
  figA  overview: (a) current camera's observed depth, (b) top-down current map + candidate cameras (arrows coloured by
        the predicted DistP1 score; V4 pick ringed, true best starred), (c) top-down: predicted surface points of the
        UNKNOWN rays of the V4 pick (all K = 4 samples) over the GT surface points of the same rays, (d) predicted score
        (mean, min-max over K) vs true one-step coverage gain for every candidate.
  figB  candidate depth grid: rows = candidates sorted by predicted score; columns = known depth of the current map,
        GT mesh depth, the K = 4 predicted depth maps, K = 4 spread, new-surface agreement (predicted vs GT).
Summary figure: over all fixed states (MP3D VALSEL 90, Macarons++ Dev-3 275), mean true gain of a random candidate,
the V4 pick, the B2 pick and the oracle best, by phase.
Usage: python viz_candidates_plot.py"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
V = os.path.join(R, "work_dirs", "exp6_v4", "weekly_report_viz")
FIG = os.path.join(R, "experiments", "exp6_v4", "weekly_report", "figures")
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "shared"))
import analyze_g12 as A12  # noqa: E402

EA, EP = A12.EA, A12.EP
import ab_load as AL  # noqa: E402

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 9.5, "axes.titleweight": "semibold", "legend.frameon": False, "font.family": "DejaVu Sans"})
GH, GW = 64, 114
LET = "ABCDEFGHIJ"


def load(nm):
    z = np.load(os.path.join(V, f"{nm}.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def disp_scale(b):
    """model ray length -> displayed depth: metres for MP3D (runtime = 7 x metres), MAGICIAN scene units for Macarons++."""
    return (float(b["k_M"]) / 7.0, "m") if str(b["set"]) == "valsel" else (float(b["k_M"]), "scene units")


def manifest_state(b):
    sid = str(b["sid"])
    if str(b["set"]) == "valsel":
        p = os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")
    else:
        p = os.path.join(EA.ORIG_W if str(b["stratum"]) == "B0" else EA.P.ROOT_B2, "manifests", "states_outdoor.json")
    return next(s for s in json.load(open(p))["states"] if s["sid"] == sid)


def pick_of(b):
    dname = "indoor" if str(b["set"]) == "valsel" else "outdoor"
    st = manifest_state(b)
    assert list(st["cand"]) == b["cand"].tolist() and int(st["t"]) == int(b["t"])
    rec = {k: np.asarray(x) for k, x in np.load(os.path.join(R, str(b["svc_src"]))).items() if k in ("legal_idx", "executed_pose_idx_history")}
    return int(AL.pick_of(dname, st, b["scores"].mean(1), rec, "V2M")[0])


def depth_img(t, dz, sc):
    z = np.where(np.isfinite(t), t * dz, np.nan) * sc
    return z.reshape(GH, GW)


def topdown(ax, b, sc, unit, light=False):
    """Top-down view of the current map: dark = observed surface above the ground (walls, objects; the ground and, indoors,
    the ceiling are removed), light = observed free space / ground, grey = still UNKNOWN, background = outside the map."""
    c = b["coarse"].astype(float) / 255.0                   # EMPTY, BAND, in-domain fractions [X', Y', Z']
    band, empty, dom = c[1] > 0, c[0] > 0, c[2] > 0
    zh = band.sum((0, 1)).astype(float)
    Z = band.shape[2]
    ground = int(np.argmax(zh[: Z // 2]))                  # most populated BAND layer in the lower half = ground / floor
    top = Z - 3 if str(b["set"]) == "valsel" else Z        # indoors also drop the ceiling layers
    struct = band[:, :, ground + 3:top].any(2)
    seen = (empty | band).any(2)
    img = np.zeros(band.shape[:2], int)
    img[dom.any(2)] = 1; img[seen] = 2; img[struct] = 3
    cols = [SURF, "#d6d5d0", "#f6f5f1", "#3a3a38"] if not light else [SURF, "#e4e3df", "#fbfbf9", "#9a9994"]
    cell = float(b["block_m"]) * sc
    ax.imshow(img.T, origin="lower", cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, extent=[0, img.shape[0] * cell, 0, img.shape[1] * cell], interpolation="nearest")
    ax.set_xlabel(f"x ({unit})"); ax.set_ylabel(f"y ({unit})")
    return cell


def cams(ax, b, sc, pick, best, L):
    s = b["scores"].mean(1)
    order = list(np.argsort(-s))
    cmap = plt.get_cmap("Blues")
    C = set(b["cand"].tolist())
    placed, pts = [], []
    for rank, j in enumerate(order):
        if j not in C:
            continue
        o = b["o"][j][:2] * sc
        dv = b["d"][j].mean(0)[:2]                          # horizontal part of the mean ray: shorter when pitched up / down
        dv = dv / max(np.linalg.norm(b["d"][j].mean(0)), 1e-9)
        tip = o + dv * L
        col = cmap(0.35 + 0.65 * (1 - rank / max(len(order) - 1, 1)))
        ax.annotate("", xy=tip, xytext=o, arrowprops={"arrowstyle": "-|>", "color": col, "lw": 2.6 if j == pick else 1.5, "mutation_scale": 12}, zorder=4)
        lab = tip + dv * L * 0.28
        for _ in range(12):                                 # nudge labels apart
            if all(np.linalg.norm(lab - q) > 0.22 * L for q in placed):
                break
            lab = lab + np.array([-dv[1], dv[0]]) * 0.25 * L
        placed.append(lab)
        ax.text(*lab, LET[rank], ha="center", va="center", fontsize=9, fontweight="bold", color=INK, zorder=6,
                bbox={"boxstyle": "round,pad=0.12", "fc": SURF, "ec": "none", "alpha": 0.8})
        if j == best:
            ax.plot(*tip, marker="*", ms=15, color=ORANGE, markeredgecolor=INK, zorder=5)
        if j == pick:
            ax.plot(*tip, marker="o", ms=17, mfc="none", mec=BLUE, mew=2.2, zorder=5)
        pts += [o, tip, lab]
    cc, cd = b["cur_center"][:2] * sc, b["cur_dir"][:2]
    cd = cd / max(np.linalg.norm(cd), 1e-9)
    ax.plot(*cc, marker="s", ms=8, color=INK, zorder=6)
    ax.annotate("", xy=cc + cd * L * 0.8, xytext=cc, arrowprops={"arrowstyle": "-|>", "color": INK, "lw": 1.6, "ls": "--"}, zorder=6)
    pts += [cc]
    return order, np.array(pts)


def fig_overview(nm, b, title):
    sc, unit = disp_scale(b)
    pick = pick_of(b)
    C = b["cand"]
    g = b["gain"] * 100
    best = int(C[np.argmax(g[C])])
    indoor = str(b["set"]) == "valsel"
    fig = plt.figure(figsize=(19, 5.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.3, 1.0, 1.0, 1.0], wspace=0.38)
    ax = fig.add_subplot(gs[0])
    zb = np.where(b["cur_mask"], np.clip(b["cur_zbuf"], 0.5, 750), np.nan) / (7.0 if indoor else 1.0)
    im = ax.imshow(zb, cmap="viridis", vmin=0, vmax=np.nanpercentile(zb, 99))
    ax.set_title(f"(a) current camera (t = {int(b['t'])}): observed depth\n({unit})", loc="left"); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    ax = fig.add_subplot(gs[1])
    cell = topdown(ax, b, sc, unit)
    ext = np.array(b["coarse"].shape[1:3]) * cell
    L = (1.1 if indoor else 0.08 * ext.max())
    order, pts = cams(ax, b, sc, pick, best, L)
    m = (3.0 if indoor else 2.2 * L)
    lo, hi = pts.min(0) - m, pts.max(0) + m
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title("(b) current map (top-down, zoomed) + candidates\n■ current camera; letters = rank by predicted score\n○ V4 pick, ★ true best", loc="left")
    hs = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in ("#3a3a38", "#f6f5f1", "#d6d5d0")]
    ax.legend(hs, ["observed structure", "observed free / ground", "unknown"], loc="lower left", fontsize=7, framealpha=0.9, frameon=True)
    ax = fig.add_subplot(gs[2])
    topdown(ax, b, sc, unit, light=True)
    j = pick
    unk = b["cls"][j] == b["cls_codes"][0]
    o, d = b["o"][j], b["d"][j]
    P = []
    for k in range(b["t_pred"].shape[1]):
        tk = b["t_pred"][j, k]
        mk = unk & np.isfinite(tk)
        p = (o[None] + d[mk] * tk[mk, None]) * sc
        P.append(p)
        ax.scatter(p[:, 0], p[:, 1], s=1.5, color=BLUE, alpha=0.22, lw=0, zorder=3)
    ts = b["tstar"][j]
    mk = unk & np.isfinite(ts)
    pg = (o[None] + d[mk] * ts[mk, None]) * sc
    ax.scatter(pg[:, 0], pg[:, 1], s=1.5, color=ORANGE, alpha=0.45, lw=0, zorder=4)
    ax.plot(*(o[:2] * sc), marker="o", ms=10, mfc="none", mec=BLUE, mew=2, zorder=5)
    allp = np.concatenate(P + [pg, (o[None, :] * sc)])
    lo, hi = np.percentile(allp[:, :2], 1, axis=0), np.percentile(allp[:, :2], 99, axis=0)
    pad = 0.08 * (hi - lo).max() + (0.5 if indoor else 2)
    ax.set_xlim(lo[0] - pad, hi[0] + pad); ax.set_ylim(lo[1] - pad, hi[1] + pad); ax.set_aspect("equal")
    ax.legend([plt.Line2D([], [], marker="o", ls="", color=BLUE, alpha=0.6), plt.Line2D([], [], marker="o", ls="", color=ORANGE, alpha=0.8)],
              ["V4 predicted surface (K = 4 samples)", "GT surface (mesh)"], loc="lower left", fontsize=7, framealpha=0.9, frameon=True)
    ax.set_title(f"(c) V4 pick ({LET[order.index(j)]}): where the model expects\nsurfaces on rays still UNKNOWN, vs the GT", loc="left")
    ax = fig.add_subplot(gs[3])
    s = b["scores"]
    for jj in C:
        r = order.index(jj)
        col = BLUE if jj == pick else (ORANGE if jj == best else INK2)
        ax.errorbar(s[jj].mean(), g[jj], xerr=[[s[jj].mean() - s[jj].min()], [s[jj].max() - s[jj].mean()]], fmt="o", color=col, ms=8 if jj in (pick, best) else 6, elinewidth=1, capsize=2)
        ax.annotate(LET[r], (s[jj].mean(), g[jj]), xytext=(5, 4), textcoords="offset points", fontsize=9, fontweight="bold")
    from scipy.stats import spearmanr
    rho = spearmanr(s[C].mean(1), g[C]).correlation
    ax.set_xlabel("predicted DistP1 score (mean of K = 4; bar = min-max)"); ax.set_ylabel("true one-step coverage gain (pp)")
    ax.set_title(f"(d) predicted score vs true gain (Spearman {rho:.2f})\nV4 pick {g[pick]:.2f} pp | best {g[best]:.2f} pp | random {g[C].mean():.2f} pp", loc="left")
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold", y=1.03)
    fig.savefig(os.path.join(FIG, f"viz_{nm}_overview.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    return pick, best, order


def fig_grid(nm, b, title, pick, best, order):
    sc, unit = disp_scale(b)
    C = list(b["cand"]); g = b["gain"] * 100; s = b["scores"].mean(1)
    rows = [pick] + ([best] if best != pick else [])
    for j in [x for x in order if x in C]:
        if len(rows) >= 4:
            break
        if j not in rows and (len(rows) < 3 or j == [x for x in order if x in C][-1]):
            rows.append(j)
    last = [x for x in order if x in C][-1]
    if last not in rows:
        rows[-1] = last
    rows = sorted(rows, key=lambda j: -s[j])
    K = b["t_pred"].shape[1]
    allgt = np.concatenate([depth_img(b["tstar"][j], b["dz"][j], sc).ravel() for j in rows])
    vmax = np.nanpercentile(allgt, 98)
    ncol = 5 + K
    fig, axs = plt.subplots(len(rows), ncol, figsize=(2.05 * ncol + 3.2, 1.45 * len(rows) + 1.4), gridspec_kw={"wspace": 0.04, "hspace": 0.22})
    cm = plt.get_cmap("viridis").copy(); cm.set_bad(SURF)
    cm_t = plt.get_cmap("viridis").copy(); cm_t.set_bad((0, 0, 0, 0))
    new_cmap = ListedColormap([GRID, AQUA, ORANGE, BLUE, SURF])
    heads = ["current map: known depth\n(grey = UNKNOWN rays)", "GT depth (mesh)"] + [f"V4 predicted depth\nsample {k + 1} of 4" for k in range(K)] + ["K = 4 spread\n(std / mean, UNKNOWN rays)", "planner's view: predicted\nNEW surface (DistP1 rule)", "truth: NEW surface\n(coverage rule, GT mesh)"]
    for ri, j in enumerate(rows):
        unk = (b["cls"][j] == b["cls_codes"][0]).reshape(GH, GW)
        kn = depth_img(np.where(b["cls"][j] == b["cls_codes"][1], b["t_known"][j], np.inf), b["dz"][j], sc)
        ax = axs[ri, 0]
        ax.imshow(np.where(unk, 1.0, 0.0), cmap=ListedColormap([SURF, "#cfcec9"]), vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.ma.masked_invalid(kn), cmap=cm_t, vmin=0, vmax=vmax)
        ax = axs[ri, 1]
        ax.imshow(depth_img(b["tstar"][j], b["dz"][j], sc), cmap=cm, vmin=0, vmax=vmax)
        zk = []
        for k in range(K):
            ax = axs[ri, 2 + k]
            zi = depth_img(b["t_pred"][j, k], b["dz"][j], sc)
            zk.append(zi)
            im = ax.imshow(zi, cmap=cm, vmin=0, vmax=vmax)
            ax.contour(unk.astype(float), levels=[0.5], colors=[INK], linewidths=0.6)
        zk = np.stack(zk)
        allhit = np.isfinite(zk).all(0) & unk
        spread = np.where(allhit, zk.std(0) / np.maximum(zk.mean(0), 1e-9), np.nan)
        ax = axs[ri, 2 + K]
        im2 = ax.imshow(spread, cmap=plt.get_cmap("Purples").with_extremes(bad=SURF), vmin=0, vmax=0.3)
        pn = (b["new_pred"][j].mean(0) >= 0.5).reshape(GH, GW)            # DistP1 novelty of the planner (>= 2 of 4 samples)
        gn = b["gt_new"][j].reshape(GH, GW)
        zg = depth_img(b["tstar"][j], b["dz"][j], sc)
        both_hit = np.isfinite(zk).all(0) & np.isfinite(zg) & unk
        err = float(np.nanmedian(np.abs(zk.mean(0) - zg)[both_hit])) if both_hit.any() else float("nan")
        hit_p = np.isfinite(zk).mean(0) >= 0.5                                # surface predicted by >= 2 of 4 samples
        agree = float((hit_p == np.isfinite(zg))[unk].mean()) if unk.any() else float("nan")
        nos = ~np.isfinite(zg)
        ax = axs[ri, 3 + K]
        ax.imshow(np.where(pn, 1, np.where(np.isfinite(zk).any(0), 0, 2)), cmap=ListedColormap([GRID, BLUE, SURF]), vmin=-0.5, vmax=2.5, interpolation="nearest")
        ax = axs[ri, 4 + K]
        ax.imshow(np.where(gn, 1, np.where(nos, 2, 0)), cmap=ListedColormap([GRID, ORANGE, SURF]), vmin=-0.5, vmax=2.5, interpolation="nearest")
        tag = ("  ◀ V4 pick" if j == pick else "") + ("  ★ true best" if j == best else "")
        axs[ri, 4 + K].text(1.06, 0.5, f"{LET[list(order).index(j)]}{tag}\npred score {s[j]:.0f}  (rank {list(order).index(j) + 1}/{len(order)})\ntrue gain {g[j]:.2f} pp  (rank {int((g[C] > g[j]).sum()) + 1}/{len(C)})\nUNKNOWN rays {unk.mean() * 100:.0f} %\nsurface / no-surface agreement\n  on UNKNOWN rays {agree * 100:.0f} %\ndepth error where both have a surface\n  (median |mean pred - GT|) {err:.2f} {unit}",
                            transform=axs[ri, 4 + K].transAxes, va="center", fontsize=8.2, color=BLUE if j == pick else INK)
        for c in range(ncol):
            axs[ri, c].set_xticks([]); axs[ri, c].set_yticks([])
            for sp in axs[ri, c].spines.values():
                sp.set_color(GRID)
            if ri == 0:
                axs[ri, c].set_title(heads[c], fontsize=8.2)
    cax = fig.add_axes([0.13, 0.02, 0.35, 0.018]); fig.colorbar(im, cax=cax, orientation="horizontal").set_label(f"depth ({unit}); white = no surface within sensor range", fontsize=8)
    cax2 = fig.add_axes([0.55, 0.02, 0.1, 0.018]); fig.colorbar(im2, cax=cax2, orientation="horizontal").set_label("K = 4 relative spread", fontsize=8)
    hs = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, GRID)]
    fig.legend(hs, ["predicted NEW (what the planner counts)", "truly NEW (what coverage counts)", "not new / already observed"], loc="lower right", ncol=1, fontsize=7.6, bbox_to_anchor=(0.99, -0.02))
    fig.suptitle(title + " — rows sorted by V4's predicted DistP1 score; black outline = rays UNKNOWN in the current map (only there does the model generate depth;\nKNOWN rays keep the map depth). The two right columns compare where the planner expects new surface with where new surface really is.", x=0.01, ha="left", fontsize=10, fontweight="semibold")
    fig.savefig(os.path.join(FIG, f"viz_{nm}_candidates.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def summary():
    """All fixed states: mean true gain of random / V4 pick / B2 pick / oracle best, by phase (scene-mean, then mean)."""
    W4 = os.path.join(R, "work_dirs", "exp6_v4", "eval_fixed")
    res = {}
    for sset in ("valsel", "dev3"):
        dname = "indoor" if sset == "valsel" else "outdoor"
        rows = [json.loads(x) for f in sorted(glob.glob(f"{W4}/{sset}_s_final_v4/*/*.jsonl")) for x in open(f)]
        if sset == "valsel":
            man = {s["sid"]: s for s in json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]}
            roots = {s: os.path.join(EP.W, "valsel") for s in man}
        else:
            m0 = {s["sid"]: s for s in json.load(open(os.path.join(EA.ORIG_W, "manifests", "states_outdoor.json")))["states"]}
            m2 = {s["sid"]: s for s in json.load(open(os.path.join(EA.P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
            man, roots = {}, {}
            for r in rows:
                man[r["sid"]] = m0[r["sid"]] if r["stratum"] == "B0" else m2[r["sid"]]
                roots[r["sid"]] = EA.ORIG_W if r["stratum"] == "B0" else EA.P.ROOT_B2
        recs = []
        for r in rows:
            st = man[r["sid"]]
            o = np.load(os.path.join(roots[r["sid"]], "states", dname, r["sid"], "oracle.npz"))
            C = np.asarray(st["cand"], int); g = np.asarray(o["gain"], float) * 100
            src = st["sources"][0] if sset == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M")
            rec = {k: np.asarray(x) for k, x in np.load(os.path.join(R, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
            x = {"scene": r["scene"], "phase": r["phase"], "random": float(g[C].mean()), "oracle": float(g[C].max())}
            for m in ("V4", "B2"):
                pk = AL.pick_of(dname, st, np.asarray([float(np.mean(k)) for k in r["scores"][m]]), rec, "V2M")[0]
                x[m] = float(g[pk]); x[f"{m}_top1"] = float(g[pk] >= g[C].max() - EA.TAU)
            x["random_top1"] = 1.0 / len(C)
            recs.append(x)
        res[sset] = recs
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.8))
    out = {}
    for ax, sset, title in ((axs[0], "valsel", "Indoor — MP3D VALSEL (90 states, V4 zero-shot)"), (axs[1], "dev3", "Outdoor — Macarons++ Dev-3 (275 states)")):
        recs = res[sset]; out[sset] = {}
        phases = ["early", "middle", "late", "all"]
        keys = [("random", "random candidate (expected)", GRID), ("B2", "B2 pick (predicted depth)", AQUA), ("V4", "V4 pick (predicted depth)", BLUE), ("oracle", "oracle best (GT)", MUTED)]
        for pi, ph in enumerate(phases):
            rr = recs if ph == "all" else [x for x in recs if x["phase"] == ph]
            scenes = sorted({x["scene"] for x in rr})
            out[sset][ph] = {}
            for ki, (k, lab, col) in enumerate(keys):
                v = float(np.mean([np.mean([x[k] for x in rr if x["scene"] == sc_]) for sc_ in scenes]))
                out[sset][ph][k] = v
                ax.bar(pi + (ki - 1.5) * 0.2, v, width=0.19, color=col, edgecolor=SURF, hatch="//" if k == "V4" else None, label=lab if pi == 0 else None)
                ax.text(pi + (ki - 1.5) * 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6.8, rotation=90)
            out[sset][ph]["n"] = len(rr)
        ax.set_xticks(range(len(phases))); ax.set_xticklabels([f"{p}\n({out[sset][p]['n']} states)" for p in phases])
        ax.set_ylabel("true one-step coverage gain of the chosen camera (pp)"); ax.set_title(title, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_ylim(0, max(out[sset]["early"]["oracle"], 1) * 1.3)
        A_ = out[sset]["all"]
        ax.text(0.99, 0.97, f"all states: V4 pick = {A_['V4'] / A_['random']:.2f} x random, {A_['V4'] / A_['oracle'] * 100:.0f} % of oracle\n"
                f"B2 pick = {A_['B2'] / A_['random']:.2f} x random, {A_['B2'] / A_['oracle'] * 100:.0f} % of oracle", transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color=INK2)
    axs[0].legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, 0.86))
    fig.suptitle("Does the predicted future depth pick better cameras? Mean true gain of the chosen camera over all development states (scene-mean, then mean)", x=0.01, ha="left", fontsize=10.5, fontweight="semibold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(FIG, "viz_summary_pick_gain.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    json.dump(out, open(os.path.join(V, "summary_pick_gain.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    states = {"indoor_good": "Indoor example (MP3D VALSEL EU6Fwq7SyZv L1, start 50, t = 5; V4 zero-shot)",
              "outdoor_good": "Outdoor example (Macarons++ colosseum, start 4, t = 15)",
              "indoor_fail": "Indoor failure case (MP3D VALSEL pLe4wQe7qrG L0, start 22, t = 0)",
              "outdoor_fail": "Outdoor failure case (Macarons++ barts, start 4, t = 10)"}
    info = {}
    for nm, title in states.items():
        b = load(nm)
        pick, best, order = fig_overview(nm, b, title)
        fig_grid(nm, b, title, pick, best, order)
        g = b["gain"] * 100; C = b["cand"]
        info[nm] = {"pick": pick, "best": best, "pick_gain": float(g[pick]), "best_gain": float(g[best]), "random_gain": float(g[C].mean()), "n_cand": int(len(C))}
    s = summary()
    json.dump(info, open(os.path.join(V, "viz_states.json"), "w"), indent=1)
    print(json.dumps(info, indent=1)); print(json.dumps(s, indent=1))
