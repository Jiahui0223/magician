"""Geometry / ROI audit K3 (CPU, read-only): figures from the REAL colosseum renders of k1_render.py and the planner masks of
k2_view_masks.py -> figures/colosseum_real/.
  01_birdseye_roi.jpg      top-down render of the whole simulator mesh: as rendered / with everything outside the ROI greyed out
  02_camera_region.jpg     top-down zoom on the camera action space (ROI box, horizontal camera grid, the five example cameras) + two
                           oblique views
  03_camera_views.jpg      what candidate cameras see: real image | ROI surfaces in colour, the rest grey | which rays reach the ROI and
                           which are stopped by a surface outside it | where D1 expects new surface and what D-v2 removes
  view_*_as_rendered.jpg   the single real frames (no overlay)"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
K = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit", "k")
FIG = os.path.join(AUD, "figures", "colosseum_real")
C_ROI, C_CAMBOX, C_SEE, C_BLOCK, C_NOVEL, C_CUT, INK = "#00C389", "#FFB000", "#00C389", "#E8400C", "#00B7EB", "#FF2FB3", "#222222"


def load(name):
    z = np.load(os.path.join(K, f"render_{name}.npz"))
    return {k: z[k] for k in z.files} | {"stats": json.loads(str(z["stats"])), "extra": json.loads(str(z["extra"]))}


def grey_outside(rgb, cls):
    g = rgb.astype(np.float64) / 255.0
    lum = (0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2])[..., None]
    out = np.where((cls == 1)[..., None], g, np.broadcast_to(0.45 * lum + 0.55, g.shape))
    return np.where((cls == 0)[..., None], 1.0, out)


def tint(g, mask, color, alpha):
    g = g.copy()
    g[mask] = (1 - alpha) * g[mask] + alpha * np.asarray(matplotlib.colors.to_rgb(color))
    return g


def draw_lines(ax, L, color, lw=1.6, ls="-"):
    for seg in L:
        ax.plot(seg[:, 0], seg[:, 1], color=color, lw=lw, ls=ls, solid_capstyle="round")


def content_box(r, extra_lines=(), margin=0.03):
    H, W = r["cls"].shape
    ys, xs = np.nonzero(r["cls"] > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    for L in (r["roi_lines"],) + tuple(extra_lines):
        p = L.reshape(-1, 2)
        p = p[np.isfinite(p).all(1)]
        p = p[(p[:, 0] >= 0) & (p[:, 0] <= W) & (p[:, 1] >= 0) & (p[:, 1] <= H)]
        if len(p):
            x0, x1, y0, y1 = min(x0, p[:, 0].min()), max(x1, p[:, 0].max()), min(y0, p[:, 1].min()), max(y1, p[:, 1].max())
    mx, my = margin * W, margin * H
    return max(0, x0 - mx), min(W, x1 + mx), max(0, y0 - my), min(H, y1 + my)


def show(ax, img, r, title=None, roi=True, lw=1.6, box=None, fs=9):
    H, W = r["cls"].shape
    ax.imshow(img, extent=(0, W, H, 0), interpolation="lanczos")
    if roi:
        draw_lines(ax, r["roi_lines"], C_ROI, lw)
    x0, x1, y0, y1 = box if box is not None else (0, W, 0, H)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fs, color=INK, loc="left")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight", facecolor="white", pil_kwargs={"quality": 88})   # photographic content -> JPEG
    plt.close(fig)


def birdseye():
    r = load("top_whole_mesh")
    e = r["extra"]
    box = content_box(r, (r["cam_box_lines"],))
    asp = (box[3] - box[2]) / (box[1] - box[0])
    fig, axs = plt.subplots(2, 1, figsize=(13, 2 * 13 * asp + 1.0))
    show(axs[0], r["rgb"], r, "Simulator mesh rendered from above (the simulator's own textured renderer). "
                              f"Mesh {e['mesh_hi'][0] - e['mesh_lo'][0]:.0f} x {e['mesh_hi'][2] - e['mesh_lo'][2]:.0f} units; green = ROI box "
                              f"{e['roi_hi'][0] - e['roi_lo'][0]:.0f} x {e['roi_hi'][2] - e['roi_lo'][2]:.0f} x {e['roi_hi'][1] - e['roi_lo'][1]:.0f}; yellow dashed = camera action space", lw=2.2, box=box)
    draw_lines(axs[0], r["cam_box_lines"], C_CAMBOX, 1.6, "--")
    show(axs[1], grey_outside(r["rgb"], r["cls"]), r, "Same render: only surfaces inside the ROI box keep their colour (the coverage target). "
                                                      "Everything grey is rendered and can occlude, but never counts", lw=2.2, box=box)
    draw_lines(axs[1], r["cam_box_lines"], C_CAMBOX, 1.6, "--")
    fig.tight_layout()
    save(fig, "01_birdseye_roi.jpg")


def camera_region(views):
    r = load("top_camera_region")
    box = content_box(r, (r["cam_box_lines"],), margin=0.01)
    fig = plt.figure(figsize=(13.5, 13.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.75, 1.0])
    ax = fig.add_subplot(gs[0, :])
    show(ax, grey_outside(r["rgb"], r["cls"]), r, f"Top-down zoom (real render; colour = inside the ROI). Dots = the 14 x 14 horizontal camera grid of the action space, drawn on the ground.\n"
                                                  f"Green = above the ROI footprint ({int(r['n_heights_in_roi'])} of {int(r['n_heights'])} camera heights are inside the ROI box); white = outside the ROI at every height. "
                                                  f"Triangles A-E = the example cameras of the next figure",
         lw=2.4, box=box, fs=9)
    draw_lines(ax, r["cam_box_lines"], C_CAMBOX, 1.8, "--")
    p, ok, inr = r["foot_pts"], r["foot_ok"], r["foot_in_roi_footprint"]
    ax.scatter(p[ok & ~inr, 0], p[ok & ~inr, 1], s=26, color="white", edgecolor="#222222", linewidths=0.7, zorder=5)
    ax.scatter(p[ok & inr, 0], p[ok & inr, 1], s=26, color=C_ROI, edgecolor="#222222", linewidths=0.7, zorder=5)
    n = len(views)
    vp = r["view_pts"][n:]                                      # ground footprints of the example cameras
    for i, v in enumerate(views):
        ax.scatter(vp[i, 0], vp[i, 1], s=230, marker="^", color=C_ROI if v["in_roi"] else C_BLOCK, edgecolor="white", linewidths=1.4, zorder=7)
        ax.annotate(f"{chr(65 + i)} (y = {v['center'][1]:.0f})", (vp[i, 0], vp[i, 1]), textcoords="offset points", xytext=(11, -5), fontsize=12, color="white", weight="bold", zorder=8,
                    path_effects=[pe.withStroke(linewidth=3, foreground="#111111")])
    for i, (name, title) in enumerate((("oblique_from_north", "Oblique real render from the +z side"), ("oblique_from_east", "Oblique real render from the +x side"))):
        q = load(name)
        a2 = fig.add_subplot(gs[1, i])
        show(a2, grey_outside(q["rgb"], q["cls"]), q, f"{title} (colour = inside the ROI box)", lw=1.8, box=content_box(q, margin=0.02))
    fig.tight_layout()
    save(fig, "02_camera_region.jpg")


def camera_views(views):
    n = len(views)
    fig, axs = plt.subplots(n, 4, figsize=(19.5, 3.05 * n + 0.9))
    heads = ("real render (ROI box in green)", "colour = surface inside the ROI (coverage target), grey = outside",
             "rays heading into the ROI box", "where D1 expects new surface")
    for i, v in enumerate(views):
        r, m = v["r"], v["m"]
        H, W = r["cls"].shape
        st = r["stats"]["of_rays_meeting_the_roi"]
        c = v["center"]
        show(axs[i, 0], r["rgb"], r, f"{chr(65 + i)}: camera ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}), {'INSIDE' if v['in_roi'] else 'outside'} the ROI" + (f" — {heads[0]}" if i == 0 else ""), lw=1.3)
        show(axs[i, 1], grey_outside(r["rgb"], r["cls"]), r, heads[1] if i == 0 else None, lw=1.3)
        g = r["rgb"].astype(np.float64) / 255.0
        g = tint(g, r["cls"] == 1, C_SEE, 0.45)
        g = tint(g, r["cls"] == 2, C_BLOCK, 0.55)
        show(axs[i, 2], g, r, f"rays into the ROI box: {100 * st['see_target']:.0f}% reach it, {100 * st['blocked_by_outside_surface']:.0f}% stopped outside", roi=False)
        g2 = r["rgb"].astype(np.float64) / 255.0
        lum = (0.299 * g2[..., 0] + 0.587 * g2[..., 1] + 0.114 * g2[..., 2])[..., None]
        g2 = 0.35 * g2 + 0.65 * (0.5 * lum + 0.5)
        sc = H / 256.0
        nov, cut = np.zeros((H, W)), np.zeros((H, W))
        for u_, v_, a_, b_ in zip(m["u"], m["v"], m["novel_w"], m["cut_w"]):
            x0, y0 = int(round((u_ - 1.5) * sc)), int(round((v_ - 1.5) * sc))
            nov[y0:y0 + int(4 * sc), x0:x0 + int(4 * sc)] = a_ - b_
            cut[y0:y0 + int(4 * sc), x0:x0 + int(4 * sc)] = b_
        an, ac = np.clip(nov, 0, 1)[..., None] * 0.75, np.clip(cut, 0, 1)[..., None] * 0.75
        g2 = g2 * (1 - an) + np.asarray(matplotlib.colors.to_rgb(C_NOVEL)) * an
        g2 = g2 * (1 - ac) + np.asarray(matplotlib.colors.to_rgb(C_CUT)) * ac
        tot, rem = float(m["novel_w"].sum()), float(m["cut_w"].sum())
        show(axs[i, 3], np.clip(g2, 0, 1), r, f"D1's expected new surface: D-v2 removes {100 * rem / max(tot, 1e-9):.0f}%", roi=False)
    h = [plt.Line2D([], [], color=C_ROI, lw=2, label="ROI box (scene box = coverage target region = P1 domain)"),
         plt.Rectangle((0, 0), 1, 1, fc=C_SEE, alpha=0.6, label="visible surface inside the ROI"),
         plt.Rectangle((0, 0), 1, 1, fc=C_BLOCK, alpha=0.7, label="surface OUTSIDE the ROI in front of the ROI box (the ray would enter the box behind it)"),
         plt.Rectangle((0, 0), 1, 1, fc=C_CUT, alpha=0.8, label="D1 expects new surface, but the ray is stopped by an already observed surface (removed by D-v2)"),
         plt.Rectangle((0, 0), 1, 1, fc=C_NOVEL, alpha=0.8, label="D1 expects new surface, kept by D-v2")]
    fig.legend(handles=h, fontsize=9, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.045))
    fig.tight_layout()
    save(fig, "03_camera_views.jpg")
    for i, v in enumerate(views):
        Image.fromarray(v["r"]["rgb"]).save(os.path.join(FIG, f"view_{chr(65 + i)}_as_rendered.jpg"), quality=95)


def main():
    os.makedirs(FIG, exist_ok=True)
    vl = json.load(open(os.path.join(AUD, "results", "colosseum_real_views.json")))["views"]
    views = []
    for w in vl:
        tag = f"s{w['start']}_t{w['t']}_j{w['j']}"
        r = load(f"view_{tag}")
        m = np.load(os.path.join(K, f"masks_{tag}.npz"))
        views.append({"r": r, "m": {k: m[k] for k in m.files}, "center": r["center"], "in_roi": r["stats"]["camera_inside_roi"], **w})
    birdseye()
    camera_region(views)
    camera_views(views)
    summary = {"views": [{"label": chr(65 + i), "start": v["start"], "step": v["t"], "candidate": v["j"], "camera": [float(x) for x in v["center"]], "camera_inside_roi": v["in_roi"],
                          **v["r"]["stats"], "d1_expected_new_surface_weighted_rays": float(v["m"]["novel_w"].sum()), "removed_by_dv2_weighted_rays": float(v["m"]["cut_w"].sum())}
                         for i, v in enumerate(views)],
               "birdseye": {n: load(n)["stats"] for n in ("top_whole_mesh", "top_camera_region", "oblique_from_north", "oblique_from_east")}}
    json.dump(summary, open(os.path.join(AUD, "results", "colosseum_real_views_summary.json"), "w"), indent=1)
    for v in summary["views"]:
        print(v["label"], np.round(v["camera"], 0), "inside" if v["camera_inside_roi"] else "outside", {k: round(x, 3) for k, x in v["of_rays_meeting_the_roi"].items()},
              "D1 new", round(v["d1_expected_new_surface_weighted_rays"]), "removed", round(v["removed_by_dv2_weighted_rays"]))


if __name__ == "__main__":
    main()
