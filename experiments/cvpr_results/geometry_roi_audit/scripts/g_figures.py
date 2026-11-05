"""Geometry / ROI audit figures (seen2scene env, CPU, read-only; real coordinates, runtime units = OBJ x 10, y-up).
figures/scene_roi_overview.{png,pdf}     All-14: mesh area inside / outside the ROI box, cameras inside the box, GT target in the P1 domain
figures/colosseum_roi_rays.{png,pdf}     colosseum top and side view: simulator mesh (target / non-target), ROI box = P1 domain, camera
                                         action space, the 10 audited rays (camera, first observed point, predicted hit)
figures/colosseum_ray_sections.{png,pdf} vertical sections through 3 of those rays: exact GT mesh section, observed cells, ROI box,
                                         camera, ray, blocking position, predicted hit"""
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AUD = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(AUD)))
WORK = os.path.join(REPO, "work_dirs", "cvpr_results", "geometry_roi_audit")
POSE = os.path.join(REPO, "work_dirs", "exp6_v3", "H", "pose_tables")
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
FIG = os.path.join(AUD, "figures")
INK, MUTED = "#222222", "#666666"
C_TARGET, C_OUT, C_OBS, C_PRED, C_CAM, C_ROI = "#4D4D4D", "#C8C8C8", "#0072B2", "#D55E00", "#222222", "#009E73"
ORDER = ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer", "barts", "colosseum", "eiffel", "pantheon", "sestino_museum", "alhambra", "bridge", "neuschwanstein"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors=MUTED, labelsize=8)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def overview():
    a1 = json.load(open(os.path.join(AUD, "results", "a1_gt_target.json")))
    a2 = json.load(open(os.path.join(AUD, "results", "a2_p1_roi.json")))
    y = np.arange(len(ORDER))[::-1]
    lab = [s.replace("_museum", "") for s in ORDER]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.6), sharey=True)
    ax = axs[0]
    tgt = np.array([a1[s]["area_share"]["target_faces_of_mesh"] for s in ORDER])
    ax.barh(y, tgt, color=C_TARGET, height=0.62, label="coverage target (faces inside the ROI box)")
    ax.barh(y, 1 - tgt, left=tgt, color=C_OUT, height=0.62, label="outside the ROI box (rendered, never counted)")
    for yi, v in zip(y, tgt):
        ax.text(min(v + 0.02, 0.9), yi, f"{100 * v:.0f}%" if v >= 0.01 else f"{100 * v:.1f}%", va="center", fontsize=7, color="white" if v > 0.9 else INK)
    ax.set_title("simulator mesh surface area", fontsize=9, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax = axs[1]
    sp = np.array([a2[s]["cameras"]["action_space_in_roi_box"] for s in ORDER])
    ex = np.array([a2[s]["cameras"]["executed_in_roi_box"] for s in ORDER])
    ax.barh(y + 0.17, sp, color="#999999", height=0.3, label="all positions of the action space")
    ax.barh(y - 0.17, ex, color=C_OBS, height=0.3, label="positions executed by D-v2 (5 starts)")
    ax.set_title("camera positions inside the ROI box", fontsize=9, color=INK, loc="left")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    ax = axs[2]
    gd = np.array([a2[s]["gt_points"]["in_p1_domain"] for s in ORDER])
    ax.barh(y, gd, color=C_ROI, height=0.62)
    for yi, v in zip(y, gd):
        ax.text(v + 0.0005, yi, f"{100 * v:.2f}%", va="center", fontsize=7, color=INK)
    ax.set_xlim(0.94, 1.008)
    ax.set_title("official GT target points inside the P1 valid domain", fontsize=9, color=INK, loc="left")
    for ax in axs:
        style(ax)
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save(fig, "scene_roi_overview")


def colosseum():
    top = json.load(open(os.path.join(AUD, "results", "top_rays_colosseum.json")))
    z = np.load(os.path.join(WORK, "a1", "colosseum.npz"))
    lo, hi = np.asarray(top["roi_box"]["lo"]), np.asarray(top["roi_box"]["hi"])
    dlo, dhi = np.asarray(top["p1_domain_rt"]["lo"]), np.asarray(top["p1_domain_rt"]["hi"])
    ms, inb = z["mesh_sample"], z["mesh_sample_target_face"]
    C = np.unique(np.load(os.path.join(POSE, "colosseum.npz"))["C"], axis=0)
    cin = ((C >= lo) & (C <= hi)).all(-1)
    wlo, whi = C.min(0) - 18.0, C.max(0) + 18.0                      # window = camera action space + margin (the mesh extends far beyond)
    wlo[1] = min(lo[1], C[:, 1].min()) - 8.0
    win = ((ms >= wlo) & (ms <= whi)).all(-1)
    fig, axs = plt.subplots(1, 2, figsize=(13.5, 6.2), gridspec_kw={"width_ratios": [1.0, 1.0]})
    for ax, (i, j, xl, yl, title) in zip(axs, ((0, 2, "x", "z", "top view"), (0, 1, "x", "y (height)", "side view"))):
        sel = np.flatnonzero(win)
        ax.scatter(ms[sel][~inb[sel], i], ms[sel][~inb[sel], j], s=0.6, color=C_OUT, rasterized=True, zorder=1, linewidths=0)
        ax.scatter(ms[sel][inb[sel], i], ms[sel][inb[sel], j], s=0.6, color=C_TARGET, rasterized=True, zorder=2, linewidths=0)
        ax.add_patch(Rectangle((lo[i], lo[j]), hi[i] - lo[i], hi[j] - lo[j], fill=False, ec=C_ROI, lw=2.0, zorder=5))
        ax.add_patch(Rectangle((dlo[i], dlo[j]), dhi[i] - dlo[i], dhi[j] - dlo[j], fill=False, ec=C_ROI, lw=0.8, ls=":", zorder=5))
        ax.scatter(C[~cin, i], C[~cin, j], s=5, color="#999999", marker="o", zorder=3, linewidths=0)
        ax.scatter(C[cin, i], C[cin, j], s=5, color=C_ROI, marker="o", zorder=3, linewidths=0)
        for n, c in enumerate(top["rays"]):
            cam, r = np.asarray(c["cam"]), c["ray"]
            ph, ob = np.asarray(r["pred_hit"]), np.asarray(r["observed_point"])
            ax.plot([cam[i], ph[i]], [cam[j], ph[j]], color=C_PRED, lw=0.9, alpha=0.9, zorder=6)
            ax.scatter(cam[i], cam[j], s=34, marker="^", color=C_CAM, edgecolor="white", linewidths=0.6, zorder=8)
            ax.scatter(ob[i], ob[j], s=30, marker="X", color=C_OBS, edgecolor="white", linewidths=0.5, zorder=9)
            ax.scatter(ph[i], ph[j], s=22, marker="o", color=C_PRED, edgecolor="white", linewidths=0.5, zorder=9)
            ax.annotate(str(n + 1), (cam[i], cam[j]), textcoords="offset points", xytext=(4, 4), fontsize=7, color=INK, zorder=10)
        ax.set_xlabel(xl, fontsize=8, color=MUTED)
        ax.set_ylabel(yl, fontsize=8, color=MUTED)
        ax.set_title(f"colosseum, {title} (runtime units; mesh cropped to the camera region)", fontsize=9, color=INK, loc="left")
        ax.set_xlim(wlo[i], whi[i])
        ax.set_ylim(wlo[j], whi[j])
        ax.set_aspect("equal")
        style(ax)
    h = [plt.Line2D([], [], marker="o", ls="", ms=4, color=C_TARGET, label="mesh: coverage target (inside ROI)"),
         plt.Line2D([], [], marker="o", ls="", ms=4, color=C_OUT, label="mesh: outside the ROI (not a target)"),
         plt.Line2D([], [], color=C_ROI, lw=2, label="ROI box = scene box (P1 valid domain dotted)"),
         plt.Line2D([], [], marker="o", ls="", ms=4, color="#999999", label="camera positions outside the ROI"),
         plt.Line2D([], [], marker="o", ls="", ms=4, color=C_ROI, label="camera positions inside the ROI"),
         plt.Line2D([], [], marker="^", ls="", ms=6, color=C_CAM, label="candidate camera of an audited ray"),
         plt.Line2D([], [], marker="X", ls="", ms=6, color=C_OBS, label="first observed point on the ray"),
         plt.Line2D([], [], marker="o", ms=5, color=C_PRED, label="ray to the predicted first hit (removed by D-v2)")]
    fig.legend(handles=h, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "colosseum_roi_rays")
    return top, lo, hi


def sections(top, lo, hi):
    import trimesh
    obj = sorted(glob.glob(os.path.join(DATA, "colosseum", "*.obj")))[0]
    m = trimesh.load(obj, force="mesh", process=False, skip_materials=True)
    m.vertices = np.asarray(m.vertices, np.float64) * 10.0
    # three rays from different episodes / cameras
    pick, seen = [], set()
    for c in top["rays"]:
        if c["episode"] not in seen:
            pick.append(c)
            seen.add(c["episode"])
        if len(pick) == 3:
            break
    fig, axs = plt.subplots(1, len(pick), figsize=(4.6 * len(pick), 5.0))
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, pick):
        cam, r = np.asarray(c["cam"]), c["ray"]
        d = np.asarray(r["dir"])
        hd = np.array([d[0], 0.0, d[2]])
        hn = np.linalg.norm(hd)
        hd = hd / hn
        nrm = np.array([-hd[2], 0.0, hd[0]])
        segs = trimesh.intersections.mesh_plane(m, plane_normal=nrm, plane_origin=cam)
        s_end = r["t_pred"] * hn + 22
        if len(segs):
            S = (segs - cam[None, None]) @ hd
            Y = segs[..., 1]
            keep = (S.max(1) >= -8) & (S.min(1) <= s_end)
            ax.add_collection(LineCollection(np.stack([S[keep], Y[keep]], -1), colors="#9A9A9A", linewidths=0.9, zorder=2))
        sec = np.load(os.path.join(WORK, "c", f"sections_{c['episode']}.npz"))
        key = f"t{c['t']}_j{c['candidate']}"
        if key in sec.files:
            P = sec[key].astype(np.float64)
            ax.scatter((P - cam[None]) @ hd, P[:, 1], s=3, color=C_OBS, zorder=3, linewidths=0, rasterized=True)
        # ROI box in this vertical plane: the interval of the horizontal line cam + s * hd inside the box footprint
        s_in, s_out = -np.inf, np.inf
        for a_ in (0, 2):
            if abs(hd[a_]) < 1e-12:
                continue
            t1, t2 = (lo[a_] - cam[a_]) / hd[a_], (hi[a_] - cam[a_]) / hd[a_]
            s_in, s_out = max(s_in, min(t1, t2)), min(s_out, max(t1, t2))
        ax.add_patch(Rectangle((s_in, lo[1]), s_out - s_in, hi[1] - lo[1], fill=True, fc=C_ROI, alpha=0.07, ec=C_ROI, lw=1.8, zorder=1))
        ax.plot([0, r["t_pred"] * hn], [cam[1], cam[1] + r["t_pred"] * d[1]], color=C_PRED, lw=1.2, zorder=5)
        ax.scatter(0, cam[1], s=60, marker="^", color=C_CAM, edgecolor="white", zorder=8)
        ob = np.asarray(r["observed_point"])
        ax.scatter((ob - cam) @ hd, ob[1], s=60, marker="X", color=C_OBS, edgecolor="white", zorder=9)
        ax.scatter(r["t_block_bundle"] * hn, cam[1] + r["t_block_bundle"] * d[1], s=44, marker="D", color="#56B4E9", edgecolor="white", zorder=9)
        ax.scatter(r["t_pred"] * hn, cam[1] + r["t_pred"] * d[1], s=50, marker="o", color=C_PRED, edgecolor="white", zorder=9)
        if r["t_true_gt"] is not None:
            ax.scatter(r["t_true_gt"] * hn, cam[1] + r["t_true_gt"] * d[1], s=70, marker="*", color="#CC79A7", edgecolor="white", zorder=9)
        ax.set_xlim(-8, s_end)
        ax.set_ylim(min(lo[1], cam[1]) - 6, max(hi[1], cam[1]) + 6)
        ax.set_aspect("equal")
        ax.set_xlabel("horizontal distance from the camera along the ray", fontsize=8, color=MUTED)
        ax.set_ylabel("height y (runtime units)", fontsize=8, color=MUTED)
        ax.set_title(f"start {c['start']}, step {c['t']}, candidate {c['candidate']}; camera ({cam[0]:.0f}, {cam[1]:.0f}, {cam[2]:.0f}) "
                     f"{'inside' if c['cam_in_box'] else 'outside'} the ROI\ndepth along the ray: observed {r['observed_point_depth']:.1f}, blocked {r['t_block_bundle']:.1f}, "
                     f"predicted {r['t_pred']:.1f}" + (f", true {r['t_true_gt']:.1f}" if r["t_true_gt"] is not None else ""), fontsize=8, color=INK, loc="left")
        style(ax)
    h = [plt.Line2D([], [], color="#9A9A9A", lw=1.2, label="simulator mesh in this vertical plane (offline)"),
         plt.Line2D([], [], marker="o", ls="", ms=3, color=C_OBS, label="depth points the agent has observed (within 0.5 of the plane)"),
         plt.Rectangle((0, 0), 1, 1, fc=C_ROI, alpha=0.15, ec=C_ROI, label="ROI box = coverage target region = P1 domain"),
         plt.Line2D([], [], marker="^", ls="", ms=7, color=C_CAM, label="candidate camera"),
         plt.Line2D([], [], marker="X", ls="", ms=7, color=C_OBS, label="first observed point on the ray"),
         plt.Line2D([], [], marker="D", ls="", ms=6, color="#56B4E9", label="D-v2 blocking depth (whole bundle blocked)"),
         plt.Line2D([], [], marker="o", ms=6, color=C_PRED, label="predicted first hit (removed)"),
         plt.Line2D([], [], marker="*", ls="", ms=9, color="#CC79A7", label="true first surface (offline)")]
    fig.legend(handles=h, fontsize=7.5, frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    save(fig, "colosseum_ray_sections")


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    overview()
    if os.path.exists(os.path.join(AUD, "results", "top_rays_colosseum.json")):
        t, lo, hi = colosseum()
        sections(t, lo, hi)
