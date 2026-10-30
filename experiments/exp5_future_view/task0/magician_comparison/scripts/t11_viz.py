"""EXP5 Task 0 / T0.11 -- indoor visualisations of the MAGICIAN comparison (seen2scene env + ffmpeg; descriptive; reads
finished outputs only, nothing is re-planned). Layout follows the EXP4 ablation animations (abl_1/common/abl_viz.py).

  --covrep     per GT point (the frozen C_obs sample, 300k area-weighted points) the move at which it is first covered,
               replayed from the stored 10 % samples with the frozen mp3d_sim.Coverage test; every replayed curve must
               equal the stored C_obs curve bitwise (else STOP) -> MC/viz/covrep/<group>/<episode>.npz
  --figs       figures/indoor/: auc_area.png, auc_heatmap.png, paired_dots.png, behaviour.png, dataset_levels.png,
               cover_<episode>.png (top view, every GT point coloured by its first-covered move, red = observable
               but not covered after 100 moves)
  --anim EP    anim/anim_<episode>.{mp4,gif}: T1-C / V2K4 / MAG-G / MAG-B / V2K4-NV side by side -- top view of the GT
               surface below 2 m (ceiling removed; coloured = covered so far, gray = not yet), trajectory + current
               view direction, the current depth image, the coverage curves and the AUC so far
Top views are in z-up metres; GT is used only on the evaluation side, after the episodes.
"""
import argparse
import json
import os
import sys
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mc_common as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

S = M.S
VIZ = os.path.join(M.MC, "viz")
FIG = os.path.join(M.MC, "figures", "indoor")
ANIM = os.path.join(M.MC, "anim")
GROUPS = ("ref", "frozen", "C", "V2K4", "V2K1", "MAG-G", "MAG-B", "V2K4-NV")
PANEL = ("C", "V2K4", "MAG-G", "MAG-B", "V2K4-NV")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COL = {"ref": "#898781", "frozen": "#c3b59b", "C": "#8f6b4a", "V2K4": "#eb6834", "V2K1": "#f2a07b", "MAG-G": "#2a78d6",
       "MAG-B": "#0b0b0b", "V2K4-NV": "#1baf7a"}
LAB = {"ref": "Random ref.", "frozen": "Frozen S2S + DistP1", "C": "T1-C + DistP1", "V2K4": "V2 K4 + Greedy", "V2K1": "V2 K1 + Greedy",
       "MAG-G": "MAGICIAN Greedy", "MAG-B": "MAGICIAN Beam 10x10", "V2K4-NV": "V2 K4 + prefer-unvisited"}
MISS, NOTYET, HIDDEN = "#d6453d", "#e4e2db", "#f5f4f0"
TCM = matplotlib.colors.ListedColormap(plt.get_cmap("Purples")(np.linspace(0.42, 1, 256)))
CUT_M = 2.0
W = 13.33
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK2,
                     "savefig.dpi": 160, "savefig.bbox": "tight", "figure.facecolor": "white"})


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def ep_info(tag):
    return next(e for e in M.episodes() if e[0] == tag)


def cov_obj(scan, lv):
    crd = M.card(scan, lv)
    return S.Coverage(types.SimpleNamespace(key=crd["key"], k=crd["k_I"]), None), crd


# ------------------------------------------------------------------------------------------------ coverage replay
def covrep(g, tag):
    op = os.path.join(VIZ, "covrep", g, f"{tag}.npz")
    if os.path.exists(op):
        return dict(np.load(op))
    _, scan, lv, _ = ep_info(tag)
    cov, crd = cov_obj(scan, lv)
    st = M.stored_curve(g, tag)
    first = np.full(len(cov.pts), -1, np.int16)
    curve = []
    for t in range(101):
        rec = np.load(M.svc_path(g, tag, t))
        new = cov._near(cov.tree, np.asarray(rec["sample_points"], np.float64)) & ~cov.covered
        cov.covered |= new
        first[new] = t
        curve.append(cov.value())
    d = float(np.abs(np.asarray(curve) - np.asarray(st["coverage"])).max())
    if d > 1e-12:
        raise SystemExit(f"STOP: coverage replay differs for {g} {tag}: {d}")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    np.savez_compressed(op, first=first, visible=cov.visible, curve=np.asarray(curve))
    return {"first": first, "visible": cov.visible, "curve": np.asarray(curve)}


def level_points(scan, lv):
    cov, crd = cov_obj(scan, lv)
    P = S.rt_to_zup_m(cov.pts)
    h = P[:, 2] - crd["floor_m"]
    keep = h < min(CUT_M, 0.8 * crd["height_m"])
    return P, keep, crd


_RS = {}


def raster(scan, lv):
    """Top view of the GT points below the cut height: cells of 3 x the median spacing, the highest point per cell."""
    key = (scan, lv)
    if key not in _RS:
        from scipy.spatial import cKDTree
        P, keep, crd = level_points(scan, lv)
        Q = P[keep]
        cell = 3.0 * float(np.median(cKDTree(Q).query(Q[::20], k=2)[0][:, 1]))
        u, v = Q[:, 0], Q[:, 1]
        iu, iv = np.floor((u - u.min()) / cell).astype(np.int64), np.floor((v - v.min()) / cell).astype(np.int64)
        nu = int(iu.max()) + 1
        lin = iv * nu + iu
        o = np.lexsort((-Q[:, 2], lin))
        idx = o[np.r_[True, lin[o][1:] != lin[o][:-1]]]
        _RS[key] = {"sel": np.flatnonzero(keep)[idx], "pix": lin[idx], "shape": (int(iv.max()) + 1, nu),
                    "extent": (u.min(), u.min() + nu * cell, v.min(), v.min() + (int(iv.max()) + 1) * cell), "crd": crd}
    return _RS[key]


def raster_img(R, rgba):
    img = np.ones((R["shape"][0] * R["shape"][1], 4))
    img[:, 3] = 0.0
    img[R["pix"]] = rgba[R["sel"]]
    return img.reshape(R["shape"][0], R["shape"][1], 4)


def cam_track(g, tag, crd):
    st = M.stored_curve(g, tag)
    P = S.rt_to_zup_m(np.asarray(crd["pano_rt"], np.float64))
    pos = np.asarray([P[p[0]][:2] for p in st["executed"]])
    fwd = []
    for t in range(101):
        R = np.load(M.svc_path(g, tag, t))["R"]
        f = R[:, 2]                                                   # view +z in the runtime frame
        f = np.array([f[0], -f[2]])
        fwd.append(f / max(1e-9, np.linalg.norm(f)))
    return pos, np.asarray(fwd), st


# ------------------------------------------------------------------------------------------------ static figures
def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("fig", name, flush=True)


def house_curve(g):
    """House-cluster mean curve (mean over the levels of a house, then over the 11 houses) -- the same averaging as the
    tables, so the AUC of this curve equals the reported AUC@100."""
    by = {}
    for tag, scan, lv, st in M.episodes():
        by.setdefault(scan, []).append(np.asarray(M.stored_curve(g, tag)["coverage"]))
    return np.mean([np.mean(v, 0) for v in by.values()], 0)


def fig_auc_area():
    fig, ax = plt.subplots(1, 2, figsize=(W, 4.6), gridspec_kw={"width_ratios": [1.5, 1]})
    C = {g: house_curve(g) for g in GROUPS}
    for g in GROUPS:
        c = C[g]
        a = np.trapz(c, dx=1.0) / 100
        ls = "--" if g in ("ref", "frozen", "C") else "-"
        ax[0].plot(c, ls, color=COL[g], lw=2.2 if g in ("V2K4", "MAG-B", "V2K4-NV") else 1.5, label=f"{LAB[g]}  (AUC {a:.3f})")
    ax[0].fill_between(np.arange(101), 0, C["MAG-B"], color=COL["MAG-B"], alpha=0.06)
    ax[0].set_xlim(0, 100)
    ax[0].set_ylim(0, 0.75)
    ax[0].set_xlabel("move")
    ax[0].set_ylabel("C_obs (house-cluster mean)")
    ax[0].grid(color=GRID)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    ax[0].set_title("Indoor closed loop, 22 levels: area under curve / 100 = AUC@100", loc="left", fontsize=10.5)
    for g in ("V2K4", "V2K4-NV", "MAG-G", "MAG-B"):
        ax[1].plot(C[g] - C["MAG-B"], color=COL[g], lw=2, label=LAB[g])
    ax[1].axhline(0, color=INK2, lw=0.8)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("move")
    ax[1].set_ylabel("coverage minus MAGICIAN Beam")
    ax[1].grid(color=GRID)
    ax[1].legend(frameon=False, fontsize=8.5)
    ax[1].set_title("Gap to MAGICIAN Beam over the episode", loc="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "auc_area.png")


def fig_auc_heatmap():
    eps = M.episodes()
    A = np.array([[np.trapz(M.stored_curve(g, e[0])["coverage"], dx=1.0) / 100 for g in GROUPS] for e in eps])
    fig, ax = plt.subplots(figsize=(W * 0.8, 9.5))
    im = ax.imshow(A, cmap="Purples", vmin=0, vmax=0.9, aspect="auto")
    for i in range(len(eps)):
        j = int(np.argmax(A[i]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#eb6834", lw=2))
        for jj in range(len(GROUPS)):
            ax.text(jj, i, f"{A[i, jj]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if A[i, jj] > 0.5 else INK)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([LAB[g] for g in GROUPS], rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(eps)))
    ax.set_yticklabels([e[0] for e in eps], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="AUC@100")
    ax.set_title("AUC@100 per episode (orange box = best group of the row)", loc="left", fontsize=10.5)
    save(fig, "auc_heatmap.png")


def fig_paired():
    r = json.load(open(os.path.join(M.MC, "analysis", "t10.json")))
    pe_ = r["per_episode"]
    pairs = [("V2K4", "MAG-B", "① full systems: V2K4 − MAG-B"), ("V2K4", "MAG-G", "② V2K4 − MAG-G"),
             ("V2K4-NV", "MAG-G", "② same revisit rule: V2K4-NV − MAG-G"), ("MAG-B", "MAG-G", "③ beam: MAG-B − MAG-G"),
             ("V2K4-NV", "V2K4", "ablation: V2K4-NV − V2K4")]
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.4))
    for ax, k, lab in zip(axs, ("auc", "comp5_100"), ("ΔAUC@100", "ΔComp@5cm @100")):
        for i, (a, b, name) in enumerate(pairs):
            d = np.array([pe_[t][a][k] - pe_[t][b][k] for t in pe_])
            ax.scatter(d, np.full(len(d), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(d)), s=16, color=COL[a], alpha=0.6, lw=0)
            s = r["paired"][f"{a}:{b}"][k]
            ax.errorbar(s["mean"], i, xerr=[[s["mean"] - s["ci95"][0]], [s["ci95"][1] - s["mean"]]], fmt="D", color=INK, ms=6, capsize=4, lw=1.6)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p[2] for p in pairs] if ax is axs[0] else [""] * len(pairs), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel(lab)
        ax.grid(axis="x", color=GRID)
    fig.suptitle("Paired differences on the same 22 episodes: dots = episodes, diamond = house-cluster mean with 95 % CI",
                 x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "paired_dots.png")


def fig_behaviour():
    r = json.load(open(os.path.join(M.MC, "analysis", "t10.json")))
    keys = [("pose_revisit_ratio", "pose revisit ratio"), ("longest_zero_run", "longest stagnation (moves)"),
            ("distinct_stations", "distinct stations visited"), ("comp5_100", "Comp@5cm after 100 moves")]
    fig, axs = plt.subplots(1, 4, figsize=(W, 3.6))
    for ax, (k, lab) in zip(axs, keys):
        v = [r["summary"][g][k] for g in GROUPS]
        ax.barh(range(len(GROUPS)), [x["mean"] for x in v], color=[COL[g] for g in GROUPS], height=0.66)
        if v[0].get("ci95"):
            ax.errorbar([x["mean"] for x in v], range(len(GROUPS)), xerr=np.array([[x["mean"] - x["ci95"][0], x["ci95"][1] - x["mean"]] for x in v]).T,
                        fmt="none", ecolor=INK2, lw=1, capsize=2)
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels([LAB[g] for g in GROUPS] if ax is axs[0] else [""] * len(GROUPS), fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(lab, loc="left", fontsize=10)
        ax.grid(axis="x", color=GRID)
    fig.tight_layout()
    save(fig, "behaviour.png")


def fig_dataset():
    eps = M.episodes()
    n = len(eps)
    cols = 6
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(W * 1.25, rows * 3.0))
    for ax, (tag, scan, lv, st) in zip(axs.flat, eps):
        R = raster(scan, lv)
        crd = R["crd"]
        img = np.tile(matplotlib.colors.to_rgba("#bdb8ad"), (len(level_points(scan, lv)[0]), 1))
        ax.imshow(raster_img(R, img), origin="lower", extent=R["extent"], interpolation="nearest")
        P = S.rt_to_zup_m(np.asarray(crd["pano_rt"], np.float64))[:, :2]
        for i, nb in crd["graph"].items():
            for j in nb:
                if int(i) < j:
                    ax.plot(P[[int(i), j], 0], P[[int(i), j], 1], color="#2a78d6", lw=0.4, alpha=0.6)
        ax.scatter(P[:, 0], P[:, 1], s=4, color="#2a78d6", zorder=3)
        s0 = next(i for i, p in enumerate(crd["pano_idx"]) if p == st)
        ax.scatter(P[s0, 0], P[s0, 1], s=60, marker="*", color="#eb6834", edgecolor="white", lw=0.6, zorder=4)
        ext = np.array(crd["box_m"][1]) - np.array(crd["box_m"][0])
        ax.set_title(f"{scan}_L{lv}\n{crd['n_panos']} stations · {ext[0]:.0f}×{ext[1]:.0f} m · h {crd['height_m']:.1f} m", fontsize=8)
        ax.set_aspect("equal")
        clean(ax)
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.suptitle("Test set of the indoor closed loop: MP3D official val, 11 houses -> 22 levels (GT surface below 2 m seen from above; "
                 "blue = panorama stations + station graph; orange star = start)", x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "dataset_levels.png")


def fig_cover(tag):
    _, scan, lv, _ = ep_info(tag)
    R = raster(scan, lv)
    fig, axs = plt.subplots(1, len(PANEL), figsize=(W * 1.3, 4.2))
    for ax, g in zip(axs, PANEL):
        d = covrep(g, tag)
        rgba = np.tile(matplotlib.colors.to_rgba(HIDDEN), (len(d["first"]), 1))
        vis = d["visible"].astype(bool)
        rgba[vis] = matplotlib.colors.to_rgba(MISS)
        c = d["first"] >= 0
        rgba[c] = TCM(np.clip(d["first"][c], 0, 100) / 100.0)
        ax.imshow(raster_img(R, rgba), origin="lower", extent=R["extent"], interpolation="nearest")
        pos, fwd, st = cam_track(g, tag, R["crd"])
        ax.plot(pos[:, 0], pos[:, 1], color=COL[g], lw=1.4, path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
        ax.scatter(pos[0, 0], pos[0, 1], s=50, facecolor="white", edgecolor=INK, lw=1.6, zorder=5)
        a = np.trapz(st["coverage"], dx=1.0) / 100
        ax.set_title(f"{LAB[g]}\nCov@100 {st['coverage'][100]:.3f} · AUC {a:.3f}", loc="left", fontsize=9.5)
        ax.set_aspect("equal")
        clean(ax)
    sm = plt.cm.ScalarMappable(cmap=TCM, norm=matplotlib.colors.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.set_label("move at which the GT surface point is first covered (light = early, dark = late)", color=INK2)
    fig.legend([Patch(color=MISS), Patch(color=HIDDEN)], ["observable, not covered after 100 moves", "not observable from any legal pose"],
               loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"{tag}: GT surface below 2 m seen from above (ceiling removed); line = camera stations visited, white dot = start",
                 x=0.01, y=1.04, ha="left", fontsize=11, color=INK2)
    save(fig, f"cover_{tag}.png")


# ------------------------------------------------------------------------------------------------ animation
def anim(tag, fps=4, preview=None):
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    _, scan, lv, _ = ep_info(tag)
    R = raster(scan, lv)
    ds = []
    for g in PANEL:
        d = covrep(g, tag)
        pos, fwd, st = cam_track(g, tag, R["crd"])
        zb = [np.load(M.svc_path(g, tag, t))["zbuf"] for t in range(101)]
        ds.append({"g": g, **d, "pos": pos, "fwd": fwd, "cov": np.asarray(st["coverage"]), "zb": zb,
                   "auc": float(np.trapz(st["coverage"], dx=1.0) / 100)})
    ref = np.asarray(M.stored_curve("ref", tag)["coverage"])
    lo, hi = np.array(R["extent"][::2]), np.array(R["extent"][1::2])
    asp = (hi[1] - lo[1]) / (hi[0] - lo[0])
    n = len(PANEL)
    Wf = 17.5
    cw = Wf / n * 0.9
    fig = plt.figure(figsize=(Wf, cw * asp + cw * 0.56 + 3.4))
    gs = fig.add_gridspec(3, n, height_ratios=[cw * asp, cw * 0.56, 2.3])
    gray = np.asarray(matplotlib.colors.to_rgba(NOTYET))
    hid = np.asarray(matplotlib.colors.to_rgba(HIDDEN))
    full, base = [], []
    for d in ds:
        b = np.tile(hid, (len(d["first"]), 1))
        b[d["visible"].astype(bool)] = gray
        base.append(b)
        full.append(TCM(np.clip(d["first"], 0, 100) / 100.0))
    ims, lines, heads, titles, views = [], [], [], [], []
    for c, d in enumerate(ds):
        ax = fig.add_subplot(gs[0, c])
        ims.append(ax.imshow(raster_img(R, base[c]), origin="lower", extent=R["extent"], interpolation="nearest"))
        lines.append(ax.plot([], [], color=COL[d["g"]], lw=1.6, path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])[0])
        L = 0.06 * float(hi[0] - lo[0])
        heads.append(ax.quiver([d["pos"][0, 0]], [d["pos"][0, 1]], [0], [0], angles="xy", scale_units="xy", scale=1, color=INK,
                               width=0.012, headwidth=3, headlength=3.5, zorder=6))
        d["L"] = L
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        clean(ax)
        ax.set_anchor("S")
        titles.append(ax.set_title("", loc="left", fontsize=10))
        axv = fig.add_subplot(gs[1, c])
        axv.set_anchor("N")
        dcm = plt.get_cmap("magma_r").copy()
        dcm.set_bad("#d9d7d0")                                   # no depth reading (no hit within 10 m)
        views.append(axv.imshow(np.zeros((256, 456)), cmap=dcm, vmin=0, vmax=10))
        clean(axv)
    axc = fig.add_subplot(gs[2, :-1])
    axc.plot(ref, "--", color=COL["ref"], lw=1.2, label="Random ref.")
    cl = []
    for d in ds:
        axc.plot(d["cov"], color=COL[d["g"]], lw=1, alpha=0.18)
        cl.append(axc.plot([], [], color=COL[d["g"]], lw=2.4, label=LAB[d["g"]])[0])
    axc.set_xlim(0, 100)
    axc.set_ylim(0, min(1.0, 1.1 * max(float(d["cov"].max()) for d in ds) + 0.02))
    axc.set_xlabel("move")
    axc.set_ylabel("level-scale C_obs")
    axc.grid(color=GRID)
    axc.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=3)
    vline = axc.axvline(0, color=INK2, lw=0.8, ls=":")
    axb = fig.add_subplot(gs[2, -1])
    bars = axb.barh(range(n), [0] * n, height=0.62, color=[COL[d["g"]] for d in ds])
    axb.scatter([d["auc"] for d in ds], range(n), marker="|", s=260, color=INK, lw=1.6, zorder=3)
    btxt = [axb.text(0, i, "", va="center", fontsize=9, color=INK) for i in range(n)]
    axb.set_yticks(range(n))
    axb.set_yticklabels([d["g"] for d in ds], fontsize=9)
    axb.invert_yaxis()
    axb.set_xlim(0, 1.0)
    axb.set_title("AUC so far (| = AUC@100)", loc="left", fontsize=10)
    axb.grid(axis="x", color=GRID)
    sup = fig.suptitle("", x=0.01, ha="left", fontsize=11.5, color=INK2)

    def update(t):
        vline.set_xdata([t, t])
        for c, d in enumerate(ds):
            on = (d["first"] >= 0) & (d["first"] <= t)
            ims[c].set_data(raster_img(R, np.where(on[:, None], full[c], base[c])))
            lines[c].set_data(d["pos"][:t + 1, 0], d["pos"][:t + 1, 1])
            heads[c].set_offsets(d["pos"][t:t + 1])
            heads[c].set_UVC([d["fwd"][t, 0] * d["L"]], [d["fwd"][t, 1] * d["L"]])
            a_t = float(np.trapz(d["cov"][:t + 1], dx=1.0) / 100)
            titles[c].set_text(f"{LAB[d['g']]}\ncoverage {d['cov'][t]:.3f} · AUC so far {a_t:.3f}")
            z = d["zb"][t].astype(np.float64) / S.S_RT
            views[c].set_data(np.where(d["zb"][t] > 0, z, np.nan))
            cl[c].set_data(np.arange(t + 1), d["cov"][:t + 1])
            bars[c].set_width(a_t)
            btxt[c].set_position((a_t + 0.02, c))
            btxt[c].set_text(f"{a_t:.3f}")
        sup.set_text(f"{tag} · move {t:3d} / 100   (GT surface below 2 m from above: coloured = covered, light = early, dark = late, "
                     f"gray = not yet, pale = not observable; arrow = current view; below: current depth 0-10 m)")
        return []
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    if preview is not None:
        update(preview)
        os.makedirs(ANIM, exist_ok=True)
        fig.savefig(os.path.join(ANIM, f"anim_preview_{tag}_t{preview}.png"), dpi=70)
        plt.close(fig)
        return
    ani = FuncAnimation(fig, update, frames=range(101), blit=False)
    os.makedirs(ANIM, exist_ok=True)
    base_p = os.path.join(ANIM, f"anim_{tag}")
    ani.save(base_p + ".mp4", writer=FFMpegWriter(fps=fps, bitrate=3500), dpi=95)
    ani.save(base_p + ".gif", writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)
    print("anim", tag, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--covrep", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--anim", nargs="*")
    ap.add_argument("--preview", type=int)
    ap.add_argument("--cover", nargs="*")
    a = ap.parse_args()
    if a.covrep:
        for tag, *_ in M.episodes():
            for g in PANEL:
                covrep(g, tag)
        print("covrep done", flush=True)
    if a.figs:
        fig_auc_area()
        fig_auc_heatmap()
        fig_paired()
        fig_behaviour()
        fig_dataset()
    for tag in a.cover or []:
        fig_cover(tag)
    for tag in a.anim or []:
        anim(tag, preview=a.preview)


if __name__ == "__main__":
    main()
