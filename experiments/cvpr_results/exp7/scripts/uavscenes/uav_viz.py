"""EXP7 / UAVScenes -- visualisations of the closed-loop runs (seen2scene env, CPU; reads only episode records, tile metadata and the
downloaded mesh cache). Outputs in experiments/cvpr_results/exp7/fig/uavscenes/:
  traj_<tile>.{pdf,png}    top-down ortho image of the tile (highest vertex colour per 0.5 m pixel) with the executed camera positions of
                           Ours / MAGICIAN / Random for the same start (start 0 unless --start), the scene box, start marker, and the
                           altitude of every step as a second panel
  curves_uav.{pdf,png}     official coverage C[t] per tile (mean over the 10 starts, shaded = min..max) for every method present
  overview_uav.{pdf,png}   all tiles in one figure (ortho + Ours / MAGICIAN / Random paths for start 0)
Usage: python uav_viz.py [--start 0] [--version v0|v1]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
DL = os.path.join(W7, "downloads", "uavscenes", "derived")
OUT = os.path.join(R, "experiments", "cvpr_results", "exp7", "fig", "uavscenes")
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import render as RD  # noqa: E402

plt = RD.plt
TILES = ["uav_AMtown_11", "uav_AMtown_12", "uav_AMvalley_01", "uav_HKairport_00", "uav_HKairport_01", "uav_HKisland_00"]
METH = {"OURS": ("V2M", "V2M", "Ours", RD.STYLE["V3-T0-D1"]["color"]), "MAG-B": ("MAG-B", "MAG-B", "MAGICIAN", RD.STYLE["MAG-B"]["color"]),
        "RANDOM": ("random", "random", "Random Walk", RD.STYLE["RANDOM"]["color"])}


def ortho(tile, px=0.5):
    meta = json.load(open(os.path.join(W7, "uav_macpp", tile, "tile_meta.json")))
    z = np.load(os.path.join(DL, f"{meta['source_scene']}_mesh.npz"))
    V, C = z["V"], z["C"]
    e0, n0, e1, n1 = meta["box_enu_m"]
    m = meta["margin_m"]
    sel = (V[:, 0] >= e0 - m) & (V[:, 0] <= e1 + m) & (V[:, 1] >= n0 - m) & (V[:, 1] <= n1 + m)
    V, C = V[sel], C[sel]
    nx, ny = int((e1 - e0 + 2 * m) / px), int((n1 - n0 + 2 * m) / px)
    ix = np.clip(((V[:, 0] - (e0 - m)) / px).astype(int), 0, nx - 1)
    iy = np.clip(((V[:, 1] - (n0 - m)) / px).astype(int), 0, ny - 1)
    order = np.argsort(V[:, 2])                               # highest vertex wins (painter's order)
    img = np.zeros((ny, nx, 3), np.uint8)
    img[iy[order], ix[order]] = C[order]
    empty = np.ones((ny, nx), bool)
    empty[iy, ix] = False
    from scipy import ndimage
    near = ndimage.distance_transform_edt(empty, return_distances=True, return_indices=True)
    d, (ii, jj) = near[0], near[1]
    img = img[ii, jj]
    img[d > 4] = 255                                           # outside the mesh footprint (> 2 m from any vertex): white
    s, (cx, cy, g) = meta["s_runtime_per_m"], meta["origin_enu_m"]
    ext = [s * (e0 - m - cx), s * (e1 + m - cx), s * (n0 - m - cy), s * (n1 + m - cy)]   # runtime x and -z ( = north) extents
    return img, ext, meta


def positions(tile, method, start, version):
    td = tile if version == "v0" else tile + "_v1"
    grp, pre = METH[method][0], METH[method][1]
    p = os.path.join(W7, "episodes", "uavscenes", "raw", method, grp, "episodes", f"ep_{pre}_{td}_s{start}_r0.json")
    if not os.path.exists(p):
        return None, None
    e = json.load(open(p))
    keys = e.get("executed_keys") or [s["pose_key"] for s in e["steps"]]
    st = json.load(open(os.path.join(W7, "uav_macpp", td, "settings.json")))["camera"]
    lo, hi = 10.0 * np.asarray(st["x_min"]), 10.0 * np.asarray(st["x_max"])
    n = np.asarray([st["pose_l"], st["pose_w"], st["pose_h"]])
    P = np.asarray([json.loads(k)[:3] for k in keys], float)
    X = lo + (0.5 + P) * (hi - lo) / n
    c = np.asarray(e["coverage"], float)
    vo = os.path.join(W7, "uav_macpp", td, "visibility_official.json")     # evaluator-GT normalisation (2026-11-12), as the paper table
    if os.path.exists(vo):
        v = json.load(open(vo))
        c = c * v["visibility_ratio_settings"] / v["starts"][str(start)]["ratio"]
    return X, c


def draw_tile(ax, tile, start, version, legend=True):
    img, ext, meta = ortho(tile)
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    sb = meta["scene_box_runtime"]
    ax.plot([sb[0][0], sb[1][0], sb[1][0], sb[0][0], sb[0][0]], [-sb[0][2], -sb[0][2], -sb[1][2], -sb[1][2], -sb[0][2]], color="white", lw=0.8, ls="--")
    for m, (_, _, lab, col) in METH.items():
        X, c = positions(tile, m, start, version)
        if X is None:
            continue
        ax.plot(X[:, 0], -X[:, 2], color=col, lw=1.3, alpha=0.95, label=f"{lab} (Final {c[-1]:.2f})")
        ax.scatter(X[-1, 0], -X[-1, 2], color=col, s=14, marker="s", zorder=4)
    X, _ = positions(tile, "OURS", start, version)
    if X is not None:
        ax.scatter(X[0, 0], -X[0, 2], color="white", edgecolor="k", s=40, marker="*", zorder=5, label="start")
    ax.set_title(tile.replace("uav_", "") + ("" if version == "v0" else " (v1)"), fontsize=8.5)
    ax.set_xticks([]); ax.set_yticks([])
    if legend:
        ax.legend(fontsize=6, loc="lower left", framealpha=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--version", default="v0")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    tiles = TILES if a.version == "v0" else ["uav_AMvalley_01", "uav_HKisland_00"]
    suf = "" if a.version == "v0" else "_v1"
    for t in tiles:
        fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})
        draw_tile(axs[0], t, a.start, a.version, legend=False)
        for m, (_, _, lab, col) in METH.items():
            X, c = positions(t, m, a.start, a.version)
            if X is not None:
                axs[1].plot(np.arange(len(X)), X[:, 1] / 0.5, color=col, lw=1.1, label=lab)
        axs[1].set_xlabel("step"); axs[1].set_ylabel("camera height above tile ground (m)"); axs[1].grid(alpha=0.25, lw=0.4)
        axs[1].set_title(f"start {a.start}: altitude", fontsize=8.5)
        h, l = axs[0].get_legend_handles_labels()
        axs[1].legend(h, l, fontsize=6.5, loc="lower right", framealpha=0.9)
        RD.save(fig, os.path.join(OUT, f"traj_{t.replace('uav_', '')}{suf}_s{a.start}"))
    # coverage curves
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True, sharey=True)
    for ax, t in zip(axs.ravel(), TILES):
        ver = a.version if t in tiles else "v0"
        for m, (_, _, lab, col) in METH.items():
            cs = [positions(t, m, k, ver)[1] for k in range(10)]
            cs = [c for c in cs if c is not None]
            if not cs:
                continue
            C = np.asarray(cs)
            ax.plot(np.arange(C.shape[1]), C.mean(0), color=col, lw=1.4, label=f"{lab} ({len(cs)})")
            ax.fill_between(np.arange(C.shape[1]), C.min(0), C.max(0), color=col, alpha=0.12, lw=0)
        ax.set_title(t.replace("uav_", "") + ("" if ver == "v0" else " (v1)"), fontsize=8.5); ax.grid(alpha=0.25, lw=0.4); ax.set_ylim(0, 1.02)
    axs[1, 1].set_xlabel("step"); axs[0, 0].set_ylabel("official coverage"); axs[1, 0].set_ylabel("official coverage")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, [x.split(" (")[0] for x in l], loc="upper center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
    RD.save(fig, os.path.join(OUT, f"curves_uav{suf}"))
    # overview
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.0))
    for ax, t in zip(axs.ravel(), TILES):
        draw_tile(ax, t, a.start, a.version if t in tiles else "v0", legend=False)
    hs = [plt.Line2D([], [], color=col, lw=1.5) for (_, _, _, col) in METH.values()] + [plt.Line2D([], [], color="white", marker="*", mec="k", ls="")]
    fig.legend(hs, [v[2] for v in METH.values()] + ["start"], loc="lower center", ncol=4, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    RD.save(fig, os.path.join(OUT, f"overview_uav{suf}_s{a.start}"))
    print("written to", OUT)


if __name__ == "__main__":
    main()
