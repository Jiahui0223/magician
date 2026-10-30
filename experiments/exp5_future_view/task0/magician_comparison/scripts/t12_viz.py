"""EXP5 Task 0 / T0.12 -- visualisations of the grid-protocol comparison (seen2scene env + ffmpeg; descriptive; reads
finished outputs only). Same drawing code and colours as t11_viz.py (imported, unchanged); 150 moves instead of 100.

  --covrep     per GT point the move at which it is first covered (frozen Coverage test on the stored 10 % samples);
               every replayed curve must equal the stored curve bitwise (else STOP) -> MC/viz/covrep_grid/<group>/<episode>.npz
  --figs       figures/grid/: grid_curves.png (house-cluster mean curves + per-episode AUC@150), grid_paired.png
               (paired differences, bootstrap CI), grid_vs_station.png (same 5 first starts: station graph @100 vs grid @100)
  --cover EP   figures/grid/cover_<episode>.png
  --anim EP    anim/anim_grid_<episode>.{mp4,gif}: Random / V2K4 / V2K4-NV / MAG-G / MAG-B side by side (as t11_viz.anim;
               faint dots = grid positions of the camera lattice, projected from above)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import t11_viz as V  # noqa: E402
import grid_env as GE  # noqa: E402

M, S, plt, pe, matplotlib = V.M, V.S, V.plt, V.pe, V.matplotlib
PANEL = ("G-RAND", "G-V2K4", "G-V2K4-NV", "G-MAG-G", "G-MAG-B")
BASE = {"G-RAND": "ref", "G-V2K4": "V2K4", "G-V2K4-NV": "V2K4-NV", "G-MAG-G": "MAG-G", "G-MAG-B": "MAG-B"}
COL = {g: V.COL[b] for g, b in BASE.items()}
LAB = {"G-RAND": "Random (prefer-unvisited)", **{g: V.LAB[b] for g, b in BASE.items() if g != "G-RAND"}}
FIG = os.path.join(M.MC, "figures", "grid")
NT = 151


def ep_info(tag):
    return next(e for e in M.grid_episodes() if e[0] == tag)


def covrep(g, tag):
    op = os.path.join(V.VIZ, "covrep_grid", g, f"{tag}.npz")
    if os.path.exists(op):
        return dict(np.load(op))
    _, scan, lv, _ = ep_info(tag)
    cov, crd = V.cov_obj(scan, lv)
    st = M.stored_curve(g, tag)
    first = np.full(len(cov.pts), -1, np.int16)
    curve = []
    for t in range(NT):
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


def cam_track(g, tag, crd):
    st = M.stored_curve(g, tag)
    gd = GE.grid_def(crd)
    pos = np.asarray([S.rt_to_zup_m(GE.position(gd, p))[:2] for p in st["executed"]])
    fwd = []
    for t in range(NT):
        R = np.load(M.svc_path(g, tag, t))["R"]
        f = np.array([R[0, 2], -R[2, 2]])
        fwd.append(f / max(1e-9, np.linalg.norm(f)))
    return pos, np.asarray(fwd), st


def lattice_xy(crd):
    gd = GE.grid_def(crd)
    n = gd["n"]
    return np.asarray([S.rt_to_zup_m(GE.position(gd, (i, 0, k)))[:2] for i in range(n[0]) for k in range(n[2])])


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print("fig", name, flush=True)


# ------------------------------------------------------------------------------------------------ static figures
def fig_curves():
    r = json.load(open(os.path.join(M.MC, "analysis", "t12.json")))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(V.W, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for g in PANEL:
        c = np.asarray(r["mean_curves"][g])
        s = r["summary"][g]
        a1.plot(c, color=COL[g], lw=2.2 if g != "G-RAND" else 1.4, ls="-" if g != "G-RAND" else "--",
                label=f"{LAB[g]}  AUC@150 {s['auc150']['mean']:.3f} · Cov@150 {s['cov150']['mean']:.3f}")
    a1.axvline(100, color=V.MUTED, lw=0.8, ls=":")
    a1.text(101, 0.97, "@100", color=V.MUTED, fontsize=8.5, transform=a1.get_xaxis_transform(), va="top")
    a1.set_xlim(0, 150)
    a1.set_ylim(0, None)
    a1.set_xlabel("move (one grid cell ~1.0 m, or one 30 / 45 degree rotation)")
    a1.set_ylabel("level-scale C_obs (house-cluster mean)")
    a1.grid(color=V.GRID)
    a1.legend(loc="lower right", frameon=False, fontsize=8.5)
    a1.set_title("Mean coverage, 5 levels x 2 starts, grid action space", loc="left", fontsize=11)
    tags = [t for t, *_ in M.grid_episodes()]
    for i, t in enumerate(tags):
        ys = [r["per_episode"][t][g]["auc150"] for g in PANEL]
        a2.plot(range(len(PANEL)), ys, color=V.GRID, lw=0.9, zorder=1)
    for j, g in enumerate(PANEL):
        ys = [r["per_episode"][t][g]["auc150"] for t in tags]
        a2.scatter([j] * len(ys), ys, s=22, color=COL[g], zorder=3)
        a2.scatter([j], [r["summary"][g]["auc150"]["mean"]], s=160, marker="_", color=V.INK, lw=2, zorder=4)
    a2.set_xticks(range(len(PANEL)))
    a2.set_xticklabels([BASE[g].replace("ref", "Random") for g in PANEL], fontsize=9)
    a2.set_ylabel("AUC@150 per episode")
    a2.grid(axis="y", color=V.GRID)
    a2.set_title("Per episode (gray lines join one episode)", loc="left", fontsize=11)
    save(fig, "grid_curves.png")


def fig_paired():
    r = json.load(open(os.path.join(M.MC, "analysis", "t12.json")))
    pairs = [("G-V2K4:G-MAG-B", "V2K4 - MAG-B"), ("G-V2K4:G-MAG-G", "V2K4 - MAG-G"), ("G-MAG-B:G-MAG-G", "MAG-B - MAG-G (beam)"),
             ("G-V2K4-NV:G-MAG-G", "V2K4-NV - MAG-G"), ("G-V2K4-NV:G-V2K4", "V2K4-NV - V2K4"), ("G-MAG-G:G-RAND", "MAG-G - Random")]
    tags = [t for t, *_ in M.grid_episodes()]
    fig, axs = plt.subplots(1, 3, figsize=(V.W, 4.4), sharey=True)
    for ax, (k, name) in zip(axs, [("auc", "AUC@100"), ("auc150", "AUC@150"), ("comp5_150", "Comp@5cm @150")]):
        for i, (pr, lab) in enumerate(pairs):
            x, y = pr.split(":")
            d = [r["per_episode"][t][x][k] - r["per_episode"][t][y][k] for t in tags]
            ax.scatter(d, np.full(len(d), i) + np.linspace(-0.18, 0.18, len(d)), s=14, color=V.MUTED, zorder=2)
            s = r["paired"][pr][k]
            ax.plot(s["ci95"], [i, i], color=V.INK, lw=2.2, zorder=3)
            ax.scatter([s["mean"]], [i], s=46, color=COL[x], edgecolor=V.INK, zorder=4)
            ax.text(0.99, i - 0.34, f"{s['mean']:+.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}] · {s['n_pos']}/10 > 0",
                    transform=ax.get_yaxis_transform(), fontsize=7.5, color=V.INK2, ha="right")
        ax.axvline(0, color=V.INK2, lw=0.8)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels([p[1] for p in pairs], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(axis="x", color=V.GRID)
    fig.suptitle("Paired differences on the grid (dots = episodes; bar = house-cluster bootstrap 95 % CI, 5 houses)", x=0.01, ha="left",
                 fontsize=11, color=V.INK2)
    save(fig, "grid_paired.png")


def fig_vs_station():
    r = json.load(open(os.path.join(M.MC, "analysis", "t12.json")))["station_first_starts"]
    gs = ["ref", "V2K4", "V2K4-NV", "MAG-G", "MAG-B"]
    fig, axs = plt.subplots(1, 3, figsize=(V.W, 3.9))
    for ax, (k, name) in zip(axs, [("auc", "AUC@100"), ("cov100", "Cov@100"), ("step_m", "mean move length (m)")]):
        x = np.arange(len(gs))
        a = [r["mean"][g][k] for g in gs]
        b = [r["grid_same_episodes_mean"]["G-RAND" if g == "ref" else f"G-{g}"][k] for g in gs]
        ax.bar(x - 0.2, a, 0.38, color=[V.COL[g] for g in gs], alpha=0.45, label="station graph (T0.6 / T0.10)")
        ax.bar(x + 0.2, b, 0.38, color=[V.COL[g] for g in gs], label="grid (T0.12)")
        for xi, (u, v) in enumerate(zip(a, b)):
            ax.text(xi - 0.2, u, f"{u:.2f}", ha="center", va="bottom", fontsize=7.5, color=V.INK2)
            ax.text(xi + 0.2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color=V.INK)
        ax.set_xticks(x)
        ax.set_xticklabels(["Random" if g == "ref" else g for g in gs], fontsize=8.5)
        ax.set_title(name, loc="left", fontsize=11)
        ax.grid(axis="y", color=V.GRID)
    fig.legend([V.Patch(color=V.MUTED, alpha=0.45), V.Patch(color=V.INK)], ["station graph (T0.6 / T0.10), pale, left bar",
               "grid (T0.12), solid, right bar"], loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Same 5 first starts: station graph (pale, left) vs grid (solid, right); descriptive only "
                 "(station-graph Random = the T0.3 reference policy)", x=0.01, ha="left", fontsize=10.5, color=V.INK2)
    save(fig, "grid_vs_station.png")


def fig_cover(tag):
    _, scan, lv, _ = ep_info(tag)
    R = V.raster(scan, lv)
    lat = lattice_xy(R["crd"])
    fig, axs = plt.subplots(1, len(PANEL), figsize=(V.W * 1.3, 4.2))
    for ax, g in zip(axs, PANEL):
        d = covrep(g, tag)
        rgba = np.tile(matplotlib.colors.to_rgba(V.HIDDEN), (len(d["first"]), 1))
        rgba[d["visible"].astype(bool)] = matplotlib.colors.to_rgba(V.MISS)
        c = d["first"] >= 0
        rgba[c] = V.TCM(np.clip(d["first"][c], 0, 150) / 150.0)
        ax.imshow(V.raster_img(R, rgba), origin="lower", extent=R["extent"], interpolation="nearest")
        ax.scatter(lat[:, 0], lat[:, 1], s=1.5, color=V.MUTED, alpha=0.5, lw=0)
        pos, fwd, st = cam_track(g, tag, R["crd"])
        ax.plot(pos[:, 0], pos[:, 1], color=COL[g], lw=1.4, path_effects=[pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()])
        ax.scatter(pos[0, 0], pos[0, 1], s=50, facecolor="white", edgecolor=V.INK, lw=1.6, zorder=5)
        a = np.trapz(st["coverage"], dx=1.0) / 150
        ax.set_title(f"{LAB[g]}\nCov@150 {st['coverage'][150]:.3f} · AUC@150 {a:.3f}", loc="left", fontsize=9.5)
        ax.set_aspect("equal")
        V.clean(ax)
    sm = plt.cm.ScalarMappable(cmap=V.TCM, norm=matplotlib.colors.Normalize(0, 150))
    cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.02)
    cb.set_label("move at which the GT surface point is first covered (light = early, dark = late)", color=V.INK2)
    fig.legend([V.Patch(color=V.MISS), V.Patch(color=V.HIDDEN)], ["observable from a station, not covered after 150 moves",
               "not observable from any station pose"], loc="lower right", frameon=False, fontsize=9)
    fig.suptitle(f"{tag} (grid): GT surface below 2 m from above; line = camera positions (top view), dots = grid lattice, white = start",
                 x=0.01, y=1.04, ha="left", fontsize=11, color=V.INK2)
    save(fig, f"cover_{tag}.png")


# ------------------------------------------------------------------------------------------------ animation
def anim(tag, fps=5, preview=None):
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    _, scan, lv, _ = ep_info(tag)
    R = V.raster(scan, lv)
    lat = lattice_xy(R["crd"])
    ds = []
    for g in PANEL:
        d = covrep(g, tag)
        pos, fwd, st = cam_track(g, tag, R["crd"])
        zb = [np.load(M.svc_path(g, tag, t))["zbuf"] for t in range(NT)]
        el = [GE.angles({"n_theta": GE.N_THETA, "n_azim": GE.N_AZIM}, p)[0] for p in st["executed"]]
        ds.append({"g": g, **d, "pos": pos, "fwd": fwd, "cov": np.asarray(st["coverage"]), "zb": zb, "el": el,
                   "auc": float(np.trapz(st["coverage"], dx=1.0) / 150)})
    lo, hi = np.array(R["extent"][::2]), np.array(R["extent"][1::2])
    asp = (hi[1] - lo[1]) / (hi[0] - lo[0])
    n = len(PANEL)
    Wf = 17.5
    cw = Wf / n * 0.9
    fig = plt.figure(figsize=(Wf, cw * asp + cw * 0.56 + 3.4))
    gs = fig.add_gridspec(3, n, height_ratios=[cw * asp, cw * 0.56, 2.3])
    gray = np.asarray(matplotlib.colors.to_rgba(V.NOTYET))
    hid = np.asarray(matplotlib.colors.to_rgba(V.HIDDEN))
    full, base = [], []
    for d in ds:
        b = np.tile(hid, (len(d["first"]), 1))
        b[d["visible"].astype(bool)] = gray
        base.append(b)
        full.append(V.TCM(np.clip(d["first"], 0, 150) / 150.0))
    ims, lines, heads, titles, views = [], [], [], [], []
    for c, d in enumerate(ds):
        ax = fig.add_subplot(gs[0, c])
        ims.append(ax.imshow(V.raster_img(R, base[c]), origin="lower", extent=R["extent"], interpolation="nearest"))
        ax.scatter(lat[:, 0], lat[:, 1], s=1.2, color=V.MUTED, alpha=0.45, lw=0)
        lines.append(ax.plot([], [], color=COL[d["g"]], lw=1.6, path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])[0])
        d["L"] = 0.06 * float(hi[0] - lo[0])
        heads.append(ax.quiver([d["pos"][0, 0]], [d["pos"][0, 1]], [0], [0], angles="xy", scale_units="xy", scale=1, color=V.INK,
                               width=0.012, headwidth=3, headlength=3.5, zorder=6))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_aspect("equal")
        V.clean(ax)
        ax.set_anchor("S")
        titles.append(ax.set_title("", loc="left", fontsize=10))
        axv = fig.add_subplot(gs[1, c])
        axv.set_anchor("N")
        dcm = plt.get_cmap("magma_r").copy()
        dcm.set_bad("#d9d7d0")
        views.append(axv.imshow(np.zeros((256, 456)), cmap=dcm, vmin=0, vmax=10))
        V.clean(axv)
    axc = fig.add_subplot(gs[2, :-1])
    cl = []
    for d in ds:
        axc.plot(d["cov"], color=COL[d["g"]], lw=1, alpha=0.18)
        cl.append(axc.plot([], [], color=COL[d["g"]], lw=2.4, label=LAB[d["g"]])[0])
    axc.set_xlim(0, 150)
    axc.set_ylim(0, min(1.0, 1.1 * max(float(d["cov"].max()) for d in ds) + 0.02))
    axc.set_xlabel("move")
    axc.set_ylabel("level-scale C_obs")
    axc.grid(color=V.GRID)
    axc.legend(loc="lower right", frameon=False, fontsize=8.5, ncol=3)
    vline = axc.axvline(0, color=V.INK2, lw=0.8, ls=":")
    axb = fig.add_subplot(gs[2, -1])
    bars = axb.barh(range(n), [0] * n, height=0.62, color=[COL[d["g"]] for d in ds])
    axb.scatter([d["auc"] for d in ds], range(n), marker="|", s=260, color=V.INK, lw=1.6, zorder=3)
    btxt = [axb.text(0, i, "", va="center", fontsize=9, color=V.INK) for i in range(n)]
    axb.set_yticks(range(n))
    axb.set_yticklabels([BASE[d["g"]].replace("ref", "Random") for d in ds], fontsize=9)
    axb.invert_yaxis()
    axb.set_xlim(0, 1.0)
    axb.set_title("AUC so far (| = AUC@150)", loc="left", fontsize=10)
    axb.grid(axis="x", color=V.GRID)
    sup = fig.suptitle("", x=0.01, ha="left", fontsize=11.5, color=V.INK2)

    def update(t):
        vline.set_xdata([t, t])
        for c, d in enumerate(ds):
            on = (d["first"] >= 0) & (d["first"] <= t)
            ims[c].set_data(V.raster_img(R, np.where(on[:, None], full[c], base[c])))
            lines[c].set_data(d["pos"][:t + 1, 0], d["pos"][:t + 1, 1])
            heads[c].set_offsets(d["pos"][t:t + 1])
            heads[c].set_UVC([d["fwd"][t, 0] * d["L"]], [d["fwd"][t, 1] * d["L"]])
            a_t = float(np.trapz(d["cov"][:t + 1], dx=1.0) / 150)
            titles[c].set_text(f"{LAB[d['g']]}\ncoverage {d['cov'][t]:.3f} · AUC so far {a_t:.3f} · pitch {d['el'][t]:+.0f}°")
            z = d["zb"][t].astype(np.float64) / S.S_RT
            views[c].set_data(np.where(d["zb"][t] > 0, z, np.nan))
            cl[c].set_data(np.arange(t + 1), d["cov"][:t + 1])
            bars[c].set_width(a_t)
            btxt[c].set_position((a_t + 0.02, c))
            btxt[c].set_text(f"{a_t:.3f}")
        sup.set_text(f"{tag} (grid, 1.0 m cells) · move {t:3d} / 150   (GT surface below 2 m from above: coloured = covered, light = early, "
                     f"dark = late, gray = not yet; dots = grid lattice; arrow = current view; below: current depth 0-10 m)")
        return []
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    os.makedirs(V.ANIM, exist_ok=True)
    if preview is not None:
        update(preview)
        fig.savefig(os.path.join(V.ANIM, f"anim_grid_preview_{tag}_t{preview}.png"), dpi=70)
        plt.close(fig)
        return
    ani = FuncAnimation(fig, update, frames=range(NT), blit=False)
    base_p = os.path.join(V.ANIM, f"anim_grid_{tag}")
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
        for tag, *_ in M.grid_episodes():
            for g in PANEL:
                covrep(g, tag)
        print("covrep done", flush=True)
    if a.figs:
        fig_curves()
        fig_paired()
        fig_vs_station()
    for tag in a.cover or []:
        fig_cover(tag)
    for tag in a.anim or []:
        anim(tag, preview=a.preview)


if __name__ == "__main__":
    main()
