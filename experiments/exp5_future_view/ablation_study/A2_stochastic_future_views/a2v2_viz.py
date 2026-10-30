"""EXP5 Ablation A2 figures v2 (seen2scene env, CPU) -- one figure per case x model x candidate (model pick, oracle best).

Layout (3 rows x 6 columns; user design 2026-10-28):
            current camera        sample 1 .. 4 (candidate)            candidate camera (truth)
  row 1     rendered RGB          per-sample numbers                   rendered RGB
  row 2     observed depth        predicted depth (64 x 114 rays)      true depth on the same rays
  row 3     observed 3D points    3D points of the predicted depth     3D points of the true depth
Colour: gray = surface already observed (planner map P1; for the truth outside the planning region: within the
coverage eps of an observed point); viridis (yellow near -> purple far) = depth of unobserved surface, one scale for
the 4 samples and the truth; white = no surface (within range / inside the planning region for predictions).
Depth = z along the optical axis (MP3D metres, Macarons++ MAGICIAN scene units). Inputs: S2 v2.npz / v2.json, S1
oracle.npz, cases/<d>_<sid>.npz (observed points), cases_v2/<d>_<sid>.npz (renders). Output figures/v2/
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402
import ab_style as ST  # noqa: E402

import numpy as np  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

plt = ST.plt
FIG = os.path.join(HERE, "figures", "v2")
GH, GW = 64, 114
RANGE_RT = 70.0                          # frozen MAGICIAN sensor range (runtime units, both datasets)
UNIT = {"indoor": (7.0, "m"), "outdoor": (1.0, "scene units")}
GRAY = "#a9a8a2"
CAT_OBS, CAT_NEW = (2, 4, 5, 7), (3, 6, 9)        # futureview.geometry.CAT: KNOWN/BLOCKED/NEAR/INF_BAND; HIT/INF_EMPTY/INF_OTHER
CMAP = plt.get_cmap("viridis_r")
LAB = {"V2M": "V2-Mix", "V2O": "V2 Original"}
ANN = {}


def zup(p):
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


def load(dataset, sid):
    d = A.state_dir(dataset, sid)
    st = next(s for s in A.load_manifest(dataset)["states"] if s["sid"] == sid)
    src = st["sources"][0]["source"]
    rec = np.load(A.svc_path(dataset, src, {"tag": st["tag"], "scene": st["scene"], "start": st["start"]}, st["t"]))
    if dataset == "indoor":
        eps = float(np.load(os.path.join(A.REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage",
                                         f"{st['scene']}.npz"))["eps_rt"])
    else:
        eps = float(np.load(os.path.join(A.W, "gt_points", f"{st['scene']}_s{st['start']}.npz"))["eps"])
    return (st, rec, np.load(os.path.join(d, "v2.npz")), json.load(open(os.path.join(d, "v2.json"))),
            np.load(os.path.join(d, "oracle.npz")), np.load(os.path.join(A.W, "cases", f"{dataset}_{sid}.npz")),
            np.load(os.path.join(A.W, "cases_v2", f"{dataset}_{sid}.npz")), eps)


def ray_geometry(z, rec, j):
    """Frame (z-up runtime) ray origin / directions of candidate j and dz = cos(angle to the optical axis)."""
    k_M, origin = float(z["k_M"]), z["origin"]
    o, d = z["o"][j].astype(np.float64), z["d"][j].astype(np.float64)
    fwd = zup(rec["legal_R"][j][:, 2])
    fwd = fwd / np.linalg.norm(fwd)
    dn = d / np.linalg.norm(d, axis=1, keepdims=True)
    return o, d, dn @ fwd, k_M, origin


def points(o, d, t, k_M, origin):
    return (o[None] + d * t[:, None]) * k_M + origin


def truth(z, o_, j, rec, obs_tree, eps):
    """Per ray: class (0 none, 1 observed, 2 unobserved in region, 3 unobserved outside region), depth z, 3D point."""
    o, d, dz, k_M, origin = ray_geometry(z, rec, j)
    ts = o_["tstar"][j].astype(np.float64)
    cat = z["lab_cat"][j]
    fin = np.isfinite(ts)
    zz = np.where(fin, ts * k_M * dz, np.inf)
    P = points(o, d, np.where(fin, ts, 0.0), k_M, origin)
    cls = np.zeros(len(ts), np.int8)
    cls[np.isin(cat, CAT_OBS) & fin] = 1
    cls[np.isin(cat, CAT_NEW) & fin] = 2
    rest = (cls == 0) & fin & (zz < RANGE_RT)
    if rest.any():
        dd, _ = obs_tree.query(P[rest], k=1, distance_upper_bound=eps)
        cls[np.flatnonzero(rest)] = np.where(np.isfinite(dd), 1, 3)
    return cls, zz, P


def sample(z, m, j, k, rec):
    o, d, dz, k_M, origin = ray_geometry(z, rec, j)
    t = z[f"{m}_t"][j][k].astype(np.float64)
    hu = z[f"{m}_hit_u"][j][k].astype(bool)
    fin = np.isfinite(t)
    cls = np.zeros(len(t), np.int8)
    cls[fin & ~hu] = 1
    cls[hu] = 2
    zz = np.where(fin, t * k_M * dz, np.inf)
    return cls, zz, points(o, d, np.where(fin, t, 0.0), k_M, origin)


def depth_image(cls, zz, u, vmin, vmax):
    img = np.ones((len(cls), 3))
    img[cls == 1] = plt.matplotlib.colors.to_rgb(GRAY)
    m = (cls == 2) | (cls == 3)
    img[m] = CMAP(np.clip((zz[m] / u - vmin) / max(vmax - vmin, 1e-9), 0, 1))[:, :3]
    return img.reshape(GH, GW, 3)


def stats(scls, sz, gcls, gz, u):
    pred = scls == 2
    real = gcls == 2
    both = pred & real
    return {"pred_rays": int(pred.sum()), "precision": float(both.sum() / pred.sum()) if pred.any() else None,
            "recall": float(both.sum() / real.sum()) if real.any() else None,
            "med_abs_dz": float(np.median(np.abs(sz[both] - gz[both])) / u) if both.any() else None,
            "phantom_on_observed": float((pred & (gcls == 1)).sum() / max(1, (gcls == 1).sum()))}


def scatter3d(ax, P, cls, zz, u, vmin, vmax, lo, hi, view, cam=None, s=2.0, obs=None):
    if obs is not None and len(obs):
        ax.scatter(*obs.T, s=0.25, color=GRAY, alpha=0.35, lw=0, rasterized=True)
    keep = np.all((P >= lo) & (P <= hi), 1)
    g = keep & (cls == 1)
    c = keep & ((cls == 2) | (cls == 3))
    if g.any():
        ax.scatter(*P[g].T, s=s, color=GRAY, lw=0, rasterized=True)
    if c.any():
        ax.scatter(*P[c].T, s=s, c=zz[c] / u, cmap=CMAP, vmin=vmin, vmax=vmax, lw=0, rasterized=True)
    if cam is not None:
        ax.scatter(*cam, s=40, marker="^", color="#d03b3b", zorder=5)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(np.maximum(hi - lo, 1e-6))
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()


def figure(dataset, case, sid, m, j, role, data):
    st, rec, z, v2, o_, cz, rz, eps = data
    u, uname = UNIT[dataset]
    obs = cz["obs_pts"].astype(np.float64)
    tree = cKDTree(obs)
    gcls, gz, gP = truth(z, o_, j, rec, tree, eps)
    smp = [sample(z, m, j, k, rec) for k in range(4)]
    col = np.concatenate([gz[(gcls == 2) | (gcls == 3)]] + [s[1][s[0] == 2] for s in smp]) / u
    col = col[np.isfinite(col)]
    vmin, vmax = (float(np.quantile(col, 0.02)), float(np.quantile(col, 0.98))) if len(col) else (0.0, 1.0)
    vmax = max(vmax, vmin + 1e-3)
    gain = o_["gain"] * 100
    C = np.asarray(st["cand"])
    sk = np.asarray(v2["models"][m]["distp1_k"])
    mean_s = np.asarray(v2["models"][m]["distp1"])
    rank = lambda x, jj: int(1 + np.sum(x[C] > x[jj]))                         # noqa: E731
    cam = (zup(rec["legal_center"][j])) / u
    cur = zup(rec["cam_center"]) / u
    # 3D crop and view: truth + samples around the candidate camera
    allp = np.concatenate([gP[gcls > 0]] + [s[2][s[0] > 0] for s in smp]) / u
    if len(allp) < 10:                                   # camera looking at nothing (e.g. out of the building)
        allp = np.concatenate([allp, cam[None] - 1.0, cam[None] + 1.0])
    lo, hi = np.quantile(allp, 0.02, 0), np.quantile(allp, 0.98, 0)
    lo, hi = np.minimum(lo, cam), np.maximum(hi, cam)
    pad = 0.08 * (hi - lo) + 1e-3
    lo, hi = lo - pad, hi + pad
    fw = zup(rec["legal_R"][j][:, 2])
    view = (24.0, float(np.degrees(np.arctan2(fw[1], fw[0]))) + 180.0 - 40.0)
    fig = plt.figure(figsize=(19.5, 9.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.35], hspace=0.10, wspace=0.04, left=0.01, right=0.99, top=0.885, bottom=0.10)
    # row 1
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rz["rgb_cur"])
    ax.set_title(f"current camera (t = {st['t']}): rendered RGB", loc="left", fontsize=9)
    ax.axis("off")
    for k in range(4):
        ax = fig.add_subplot(gs[0, 1 + k])
        ax.axis("off")
        s = stats(smp[k][0], smp[k][1], gcls, gz, u)
        f = lambda x, p=0: "—" if x is None else (f"{100 * x:.0f} %" if p else f"{x:.2f}")   # noqa: E731
        txt = (f"sample {k + 1}\n\nDistP1 = {sk[j, k]:.1f}\n"
               f"predicted new-surface rays: {s['pred_rays']}\n"
               f"of them really new surface: {f(s['precision'], 1)}\n"
               f"true new surface recovered: {f(s['recall'], 1)}\n"
               f"median |depth error|: {f(s['med_abs_dz'])} {uname}\n"
               f"phantom on observed rays: {f(s['phantom_on_observed'], 1)}")
        ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=9, color=ST.INK, family="DejaVu Sans",
                bbox=dict(boxstyle="round,pad=0.5", fc="#f6f5f2", ec=ST.GRID))
        ANN[f"v2|{dataset}|{case}|{m}|j{j}|k{k}"] = float(sk[j, k])
    ax = fig.add_subplot(gs[0, 5])
    ax.imshow(rz[f"rgb_c{j}"])
    ax.set_title(f"candidate j={j} ({role}): rendered RGB", loc="left", fontsize=9)
    ax.axis("off")
    # row 2
    ax = fig.add_subplot(gs[1, 0])
    zc = rz["z_cur"].astype(np.float64)
    zc = np.where(zc > 0, zc / u, np.nan)
    zv = zc[np.isfinite(zc)]
    ax.imshow(zc, cmap="Greys", vmin=np.quantile(zv, 0.02) if len(zv) else 0, vmax=np.quantile(zv, 0.98) if len(zv) else 1,
              interpolation="nearest")
    ax.set_title("current depth (all observed; gray: light = near)", loc="left", fontsize=9)
    ax.axis("off")
    for k in range(4):
        ax = fig.add_subplot(gs[1, 1 + k])
        ax.imshow(depth_image(smp[k][0], smp[k][1], u, vmin, vmax), interpolation="nearest", aspect=256 / 456 * GW / GH)
        ax.set_title(f"sample {k + 1}: predicted depth", loc="left", fontsize=9)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, 5])
    ax.imshow(depth_image(gcls, gz, u, vmin, vmax), interpolation="nearest", aspect=256 / 456 * GW / GH)
    outside = (gcls == 3).reshape(GH, GW)
    if outside.any():
        ax.contour(outside.astype(float), levels=[0.5], colors="#d03b3b", linewidths=1.0, linestyles="--")
    ax.set_title("true depth (red dashes: outside V2 region)", loc="left", fontsize=9)
    ax.axis("off")
    # row 3
    ax = fig.add_subplot(gs[2, 0], projection="3d")
    big_lo = np.minimum(np.minimum(lo, cur), cam) - pad
    big_hi = np.maximum(np.maximum(hi, cur), cam) + pad
    ob = obs / u
    ob = ob[np.all((ob >= big_lo) & (ob <= big_hi), 1)]
    if len(ob) > 40000:
        ob = ob[np.random.default_rng(0).choice(len(ob), 40000, replace=False)]
    scatter3d(ax, np.zeros((0, 3)), np.zeros(0, np.int8), np.zeros(0), u, vmin, vmax, big_lo, big_hi, view, cam=cam, obs=ob)
    ax.scatter(*cur, s=40, marker="o", color=ST.INK, zorder=5)
    ax.plot(*np.stack([cam, cam + fw / np.linalg.norm(fw) * 0.12 * np.linalg.norm(big_hi - big_lo)]).T, color="#d03b3b", lw=1.5)
    ax.set_title("observed so far (gray); ● current, ▲ candidate", loc="left", fontsize=9)
    for k in range(4):
        ax = fig.add_subplot(gs[2, 1 + k], projection="3d")
        scatter3d(ax, smp[k][2] / u, smp[k][0], smp[k][1], u, vmin, vmax, lo, hi, view, cam=cam)
        ax.set_title(f"sample {k + 1}: 3D of the predicted depth", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[2, 5], projection="3d")
    scatter3d(ax, gP / u, gcls, gz, u, vmin, vmax, lo, hi, view, cam=cam)
    ax.set_title("truth: 3D of the true depth", loc="left", fontsize=9)
    # colour bar + legend
    cax = fig.add_axes([0.30, 0.055, 0.40, 0.013])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=CMAP), cax=cax, orientation="horizontal")
    cb.set_label(f"depth of UNOBSERVED surface ({uname}): yellow = near, purple = far   |   gray = already observed   |   "
                 f"white = no surface", fontsize=9)
    pk = {mm: int(rz[f"pick_{mm}"]) for mm in ("V2M", "V2O")}
    regret = gain[C].max() - gain[pk[m]]
    fig.suptitle(f"{'MP3D' if dataset == 'indoor' else 'Macarons++'} {st['scene']}  start {st['start']}  t = {st['t']}   "
                 f"[case: {case}]\n"
                 f"candidate j = {j} = {role}.   Oracle gain {gain[j]:.3f} pp (rank {rank(gain, j)}/{len(C)}),   "
                 f"GT-depth DistP1 {v2['gtd_distp1'][j]:.1f},   {LAB[m]} mean DistP1 {mean_s[j]:.1f} (rank {rank(mean_s, j)}/{len(C)}).   "
                 f"{LAB[m]} picked j = {pk[m]}, oracle best j = {int(rz['best'])}, regret {regret:.3f} pp",
                 x=0.01, ha="left", fontsize=10.5)
    os.makedirs(FIG, exist_ok=True)
    tagr = "best" if role == "oracle best" else ("gtdpick" if role == "GTD pick" else "pick")
    fn = f"{dataset}_{case}_{m}_{tagr}_j{j}.png"
    fig.savefig(os.path.join(FIG, fn), dpi=130)
    plt.close(fig)
    ANN[f"v2|{dataset}|{case}|{m}|j{j}|gain"] = float(gain[j])
    ANN[f"v2|{dataset}|{case}|{m}|j{j}|gtd"] = float(v2["gtd_distp1"][j])
    return fn


def main():
    ST.setup()
    cases = json.load(open(os.path.join(HERE, "results", "cases.json")))
    out = []
    for dataset, cs in cases.items():
        for case, c in cs.items():
            if not c["sid"] or not os.path.exists(os.path.join(A.W, "cases_v2", f"{dataset}_{c['sid']}.npz")):
                continue
            data = load(dataset, c["sid"])
            rz = data[6]
            best = int(rz["best"])
            for m in ("V2M", "V2O"):
                pk = int(rz[f"pick_{m}"])
                roles = [(pk, f"{LAB[m]} pick")] + ([(best, "oracle best")] if best != pk else [])
                gp = int(rz["pick_GTD"]) if "pick_GTD" in rz.files else -1
                if gp >= 0 and gp not in (pk, best):
                    roles.append((gp, "GTD pick"))
                for j, role in roles:
                    out.append(figure(dataset, case, c["sid"], m, j, role, data))
                    print(out[-1], flush=True)
    json.dump(ANN, open(os.path.join(FIG, "annotations.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
