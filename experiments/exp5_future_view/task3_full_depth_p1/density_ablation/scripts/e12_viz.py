"""EXP5 Task 3 / E1.2 -- Figures 3-5 (seen2scene env, 1 GPU). The four density maps of ONE fixed state are rebuilt one
after the other from the same stored observations with the planner rules (e12_fixed); same camera, crop and colours.

fig3_<d>_<scene>_s<start>_t<t>.png  rows = D10 / D50 / D75 / D100: fine P1 slice at the camera height | zoom | 8^3 coarse
                                    slice (3D-encoder input)
fig4_<...>.png                      candidate camera (the I100 = B2 pick): rows = D10..D100: t_known | UNKNOWN segments per
                                    ray | EMPTY end depth | phantom rays (known surface > tol in front of GT); last row GT
                                    depth + GT new-surface truth (evaluation only)
fig5_<...>.png                      scoring disagreement (S10 vs S100, same D100 predictions): top-down candidates (colour =
                                    oracle gain; picks marked), per-candidate DistP1 under sparse / dense scoring vs oracle
                                    gain; for the two picks: predicted depth (k = 0), novelty under the sparse and the dense
                                    band_dil, pixels excluded only by the dense map (red = truly new, grey = truly old), GT depth
Usage: python e12_viz.py fig34|fig5 DATASET STRATUM SID [...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e12_fixed as X  # noqa: E402
import textwrap  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

FIGD = os.path.join(os.path.dirname(HERE), "figures")
C_EMPTY, C_BAND, C_UNK = (0.62, 0.77, 0.96), (0.05, 0.05, 0.05), (0.92, 0.91, 0.89)
GH, GW = 64, 114


def find(dataset, stratum, sid):
    root = X.ORIG_W if stratum == "B0" else X.P.ROOT_B2
    for st in json.load(open(os.path.join(root, "manifests", f"states_{dataset}.json")))["states"]:
        if st["sid"] == sid:
            return st, root
    raise KeyError(sid)


class StateMaps:
    """Maps of every density at one state (built sequentially; geometry kept, fusion freed)."""

    def __init__(self, dataset, stratum, st, root, core, dev, dens=X.DENS, want_h=()):
        np, torch, MC, PC, PR, RY = X.np, X.torch, X.MC, X.PC, X.PR, X.RY
        self.dataset, self.st, self.root = dataset, st, root
        ep = X.S2.episode_of(dataset, st)
        self.rp = X.S2.StateReplay(dataset, ep, "V2M", {}, dev)
        t = st["t"]
        self.recs = [MC.load_service_record(X.A.svc_path(dataset, "V2M", ep, q)) for q in range(t + 1)]
        r = t - t % 10
        if dataset == "indoor":
            fr, dims, dom, _, _ = self.rp.F.region_geometry(self.rp.card, self.recs[r]["cam_center"])
        else:
            fr, dims, dom, _ = self.rp.frame
        self.region = (fr, dims, dom)
        self.geo, self.h = {}, {}
        self.eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(self.rp.seed_key(tr, kk)))
        for x in dens:
            fus = X.D.fusion_class(x)(*self.region, device=dev)
            for q in range(r + 1):
                fus.integrate_record(self.recs[q])
            if x in want_h:
                g = PC.Geometry.from_fusion(*self.region, fus, dev)
                self.h[x] = core.pred.encode(g.coarse())
                del g
            for q in range(r + 1, t + 1):
                fus.integrate_record(self.recs[q])
            self.geo[x] = PC.Geometry.from_fusion(*self.region, fus, dev)
            del fus
            torch.cuda.empty_cache()
        self.rec = self.recs[-1]
        self.orc = np.load(os.path.join(root, "states", dataset, st["sid"], "oracle.npz"))
        self.k_M = float(fr.k_M)
        self.z0 = self.rp.z0


def slice_img(geo, kz):
    G, np = X.G, X.np
    s = geo.struct[:, :, kz].cpu().numpy()
    dm = geo.domain[:, :, kz].cpu().numpy()
    img = np.ones(s.shape + (3,))
    img[(s == G.UNKNOWN) & dm] = C_UNK
    img[s == G.EMPTY] = C_EMPTY
    img[s == G.BAND] = C_BAND
    return img, s


def coarse_img(geo, kz):
    np = X.np
    c = geo.coarse().cpu().numpy().astype(np.float32) / 255.0
    ce, cb = c[0][:, :, kz // 8], c[1][:, :, kz // 8]
    img = np.ones(ce.shape + (3,)) * (1 - ce[..., None]) + np.asarray(C_EMPTY)[None, None] * ce[..., None]
    b = np.clip(cb * 8, 0, 1)[..., None]
    return img * (1 - b) + np.asarray(C_BAND)[None, None] * b


def cam_slice(M):
    np, G = X.np, X.G
    g10 = M.geo[10]
    oc = g10.fr.runtime_to_model(np.asarray(M.rec["cam_center"]).reshape(1, 3))[0] / G.VS
    s0 = g10.struct
    kz = int(np.clip(oc[2], 0, s0.shape[2] - 1))
    if ((s0[:, :, kz] == G.EMPTY) | (s0[:, :, kz] == G.BAND)).sum().item() < 500:
        Pz = g10.fr.runtime_to_model(np.asarray(M.rec["sample_points"], np.float64))[:, 2] / G.VS
        kz = int(np.clip(np.median(Pz) + 6, 0, s0.shape[2] - 1))
    return kz, oc


def crop_of(imgs):
    np, G = X.np, X.G
    known = np.zeros(imgs[100][1].shape, bool)
    for x in imgs:
        known |= (imgs[x][1] == G.EMPTY) | (imgs[x][1] == G.BAND)
    xs, ys = np.nonzero(known)
    return max(0, xs.min() - 15), min(known.shape[0], xs.max() + 15), max(0, ys.min() - 15), min(known.shape[1], ys.max() + 15)


def ray_maps(M, x, R, T, ztrue, u):
    np, MC, torch = X.np, X.MC, X.torch
    geo = M.geo[x]
    v = geo.view(R, T)
    dz = v["dz"].double().cpu().numpy()
    tk = v["pr"]["t_known"]
    zk = tk.double().cpu().numpy() * M.k_M * dz
    tb, te = X.struct_march(geo, v["o"], v["d"], v["tcap"])
    ze = te.double().cpu().numpy() * M.k_M * dz
    gt = np.isfinite(ztrue) & (ztrue < MC.SENSOR_RANGE)
    tol = MC.TRUNC * M.k_M + 0.01 * np.where(gt, ztrue, 0.0)
    known = np.isfinite(zk)
    ph = known & gt & (zk < np.where(gt, ztrue, 0) - tol)
    nseg = v["pr"]["n_seg"].cpu().numpy()
    return {"zk": np.where(known, zk / u, np.nan).reshape(GH, GW), "ze": np.where(np.isfinite(ze), ze / u, np.nan).reshape(GH, GW),
            "nseg": nseg.reshape(GH, GW).astype(float), "ph": ph.reshape(GH, GW),
            "tk_valid": float(known[gt].mean()) if gt.any() else None, "ph_rate": float(ph[known & gt].mean()) if (known & gt).any() else None,
            "nseg_mean": float(nseg.mean()), "v": v}


def fig34(dataset, stratum, sid, core, dev, pick=None):
    np, MC, plt = X.np, X.MC, ST.plt
    st, root = find(dataset, stratum, sid)
    M = StateMaps(dataset, stratum, st, root, core, dev)
    u, uname = (7.0, "m") if dataset == "indoor" else (1.0, "scene units")
    rows = json.load(open(STATES))[dataset]
    row = next(r for r in rows if r["sid"] == sid)
    C = np.asarray(st["cand"])
    j = int(pick) if pick is not None else int(row["I100_pick"])
    kz, oc = cam_slice(M)
    imgs = {x: slice_img(M.geo[x], kz) for x in X.DENS}
    x0, x1, y0, y1 = crop_of(imgs)
    bx, by = np.nonzero(imgs[100][1] == X.G.BAND)
    cx, cy = (int(np.median(bx)), int(np.median(by))) if len(bx) else (int(oc[0]), int(oc[1]))
    zr = 45
    # ---------------- fig 3
    fig = plt.figure(figsize=(15, 17))
    gs = fig.add_gridspec(4, 3, hspace=0.18, wspace=0.05, left=0.01, right=0.99, top=0.94, bottom=0.01)
    for i, x in enumerate(X.DENS):
        img, s = imgs[x]
        nb = int((M.geo[x].struct == X.G.BAND).sum())
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(np.transpose(img[x0:x1, y0:y1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[x0, x1, y0, y1])
        ax.scatter([oc[0]], [oc[1]], s=40, color="#d03b3b")
        ax.add_patch(plt.Rectangle((cx - zr, cy - zr), 2 * zr, 2 * zr, fill=False, ec="#d03b3b", lw=1.2))
        ax.set_title(f"D{x}: fine P1 slice z={kz} (BAND {nb:,} voxels)", loc="left", fontsize=10)
        ax = fig.add_subplot(gs[i, 1])
        zx0, zx1, zy0, zy1 = max(0, cx - zr), min(img.shape[0], cx + zr), max(0, cy - zr), min(img.shape[1], cy + zr)
        ax.imshow(np.transpose(img[zx0:zx1, zy0:zy1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[zx0, zx1, zy0, zy1])
        ax.set_title(f"D{x}: zoom (red box)", loc="left", fontsize=10)
        ax = fig.add_subplot(gs[i, 2])
        ci = coarse_img(M.geo[x], kz)
        ax.imshow(np.transpose(ci[x0 // 8:x1 // 8 + 1, y0 // 8:y1 // 8 + 1], (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(f"D{x}: 3D-encoder input (8^3 blocks; dark = BAND fraction x8)", loc="left", fontsize=10)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])
    nm = "MP3D" if dataset == "indoor" else "Macarons++"
    fig.suptitle(f"Task 3 / E1.2 Figure 3 — the same observation history fused at four BAND densities ({nm} {st['scene']} s{st['start']} t={st['t']}, "
                 f"{'B0' if stratum == 'B0' else 'B2'} trajectory).\nblue = EMPTY (carved from the full depth, identical rule), black = BAND, light grey = UNKNOWN, "
                 f"red dot = current camera. Same slice, crop and colours in every row.", x=0.005, ha="left", fontsize=11)
    tag = f"{dataset}_{st['scene']}_s{st['start']}_t{st['t']}"
    fig.savefig(os.path.join(FIGD, f"fig3_{tag}.png"), dpi=100)
    plt.close(fig)
    # ---------------- fig 4 (candidate j)
    vj = M.geo[100].view(M.rec["legal_R"][j], M.rec["legal_T"][j])
    dzj = vj["dz"].double().cpu().numpy()
    tsj = M.orc["tstar"][j].astype(np.float64)
    zj = tsj * M.k_M * dzj
    okj = np.isfinite(tsj) & (zj < MC.SENSOR_RANGE)
    zj = np.where(okj, zj, np.inf)
    maps = {x: ray_maps(M, x, M.rec["legal_R"][j], M.rec["legal_T"][j], zj, u) for x in X.DENS}
    vals = zj[np.isfinite(zj)] / u
    vmin, vmax = np.quantile(vals, 0.02), np.quantile(vals, 0.98)
    cmd = plt.get_cmap("viridis_r").copy()
    cmd.set_bad("white")
    fig = plt.figure(figsize=(19, 17))
    gs = fig.add_gridspec(5, 4, hspace=0.42, wspace=0.05, left=0.01, right=0.99, top=0.93, bottom=0.01)
    nmax = max(1.0, max(np.nanmax(m["nseg"]) for m in maps.values()))
    for i, x in enumerate(X.DENS):
        mm = maps[x]
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(mm["zk"], cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(textwrap.fill(f"D{x}: t_known (model input) — valid on {100 * (mm['tk_valid'] or 0):.0f}% of the rays with a true surface", 70), loc="left", fontsize=9)
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(mm["nseg"], cmap=ST.SEQ, vmin=0, vmax=min(nmax, 8), interpolation="nearest")
        ax.set_title(textwrap.fill(f"D{x}: UNKNOWN segments per ray (mean {mm['nseg_mean']:.2f}; white 0 -> dark blue >= 8)", 70), loc="left", fontsize=9)
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(mm["ze"], cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(textwrap.fill(f"D{x}: EMPTY end depth (first non-EMPTY sample; same colours)", 70), loc="left", fontsize=9)
        ax = fig.add_subplot(gs[i, 3])
        img = np.ones((GH, GW, 3))
        img[mm["ph"]] = (0.82, 0.23, 0.23)
        img[~np.isfinite(zj).reshape(GH, GW)] = (0.93, 0.93, 0.93)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(textwrap.fill(f"D{x}: phantom rays (red; {100 * (mm['ph_rate'] or 0):.1f}% of known-surface rays); grey = no true surface", 70), loc="left", fontsize=9)
    truth_new = X.np.zeros(GH * GW, bool)
    st_rows = rows
    ax = fig.add_subplot(gs[4, 0])
    ax.imshow(np.where(np.isfinite(zj), zj / u, np.nan).reshape(GH, GW), cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(textwrap.fill(f"reference (NOT an input): TRUE depth of candidate j={j} ({uname}; yellow near -> purple far)", 70), loc="left", fontsize=9)
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, M.recs, dataset
    tr, _ = E.truth_of(st, M.orc, None, M.k_M)
    if j in tr:
        truth_new = tr[j][0]
    ax = fig.add_subplot(gs[4, 1])
    img = np.ones((GH, GW, 3))
    img[okj.reshape(GH, GW)] = to_rgb(ST.KNOWN_GRAY)
    img[(truth_new & okj).reshape(GH, GW)] = (0.11, 0.69, 0.48)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(textwrap.fill("reference: GT surface NEW (green) vs already observed (grey) — farther / closer than the coverage eps to past samples", 70), loc="left", fontsize=9)
    ax = fig.add_subplot(gs[4, 2])
    ax.axis("off")
    ax.text(0, 0.95, f"{nm} {st['scene']} s{st['start']} t={st['t']}\ncandidate j = {j} (I100 = B2 pick)\n\n" + "\n".join(
        f"D{x}: t_known valid {100 * (maps[x]['tk_valid'] or 0):.0f}%   phantom {100 * (maps[x]['ph_rate'] or 0):.1f}%   UNKNOWN segments {maps[x]['nseg_mean']:.2f}"
        for x in X.DENS), va="top", fontsize=10)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Task 3 / E1.2 Figure 4 — ray features of the SAME candidate camera at four BAND densities ({nm} {st['scene']} s{st['start']} t={st['t']}). "
                 f"Depth colours shared by all depth panels.", x=0.005, ha="left", fontsize=11)
    fig.savefig(os.path.join(FIGD, f"fig4_{tag}.png"), dpi=100)
    plt.close(fig)
    return {"fig3": f"fig3_{tag}.png", "fig4": f"fig4_{tag}.png", "candidate": j,
            "stats": {str(x): {k: maps[x][k] for k in ("tk_valid", "ph_rate", "nseg_mean")} for x in X.DENS}}


def fig5(dataset, stratum, sid, core, dev):
    np, MC, torch, PC, plt = X.np, X.MC, X.torch, X.PC, ST.plt
    st, root = find(dataset, stratum, sid)
    M = StateMaps(dataset, stratum, st, root, core, dev, dens=(10, 100), want_h=(100,))
    rec, k_M, z0 = M.rec, M.k_M, M.z0
    out, _, views = PC.score_state(core, M.geo[100], M.h[100], rec["legal_R"], rec["legal_T"], M.eps, z0, keep_hits=True)
    hits = [hh["t"] for hh in out["hits"]]
    s100 = np.asarray([float(np.mean(x)) for x in out["distp1_k"]])
    s10 = np.asarray([float(np.mean(x)) for x in X.distp1_k(views, hits, M.geo[10], k_M, z0)])
    row = next(r for r in json.load(open(STATES))[dataset] if r["sid"] == sid)
    C = np.asarray(st["cand"])
    g = M.orc["gain"][C] * 100
    import ab_load as AL
    srec = {k: np.asarray(x) for k, x in np.load(os.path.join(X.A.REPO, next(e for e in st["sources"] if e["source"] == "V2M")["svc"])).items()
            if k in ("legal_idx", "executed_pose_idx_history")}
    p10, p100 = AL.pick_of(dataset, st, s10, srec, "V2M")[0], AL.pick_of(dataset, st, s100, srec, "V2M")[0]
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, M.recs, dataset
    tr, eps_rt = E.truth_of(st, M.orc, None, k_M)
    u, uname = (7.0, "m") if dataset == "indoor" else (1.0, "scene units")
    nm = "MP3D" if dataset == "indoor" else "Macarons++"
    fig = plt.figure(figsize=(21, 13))
    gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.08, left=0.02, right=0.99, top=0.91, bottom=0.03, height_ratios=[1.25, 1, 1])
    # top-down
    kz, oc = cam_slice(M)
    img, sl = slice_img(M.geo[100], kz)
    imgs = {100: (img, sl), 10: slice_img(M.geo[10], kz)}
    x0, x1, y0, y1 = crop_of(imgs)
    ax = fig.add_subplot(gs[0, 0:2])
    ax.imshow(np.transpose(img[x0:x1, y0:y1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[x0, x1, y0, y1], alpha=0.55)
    cmap = ST.SEQ
    gn = (g - g.min()) / max(1e-9, g.max() - g.min())
    for i, c in enumerate(C):
        pc = M.geo[10].fr.runtime_to_model(np.asarray(rec["legal_center"][c], np.float64).reshape(1, 3))[0] / X.G.VS
        d_y, _, _ = MC.pixel_rays(np.asarray(rec["legal_R"][c], np.float64), np.asarray(rec["legal_T"][c], np.float64), np.asarray([MC.W / 2]), np.asarray([MC.H / 2]))
        dm = MC.to_z_up(d_y)[0]
        ax.arrow(pc[0], pc[1], 18 * dm[0], 18 * dm[1], width=0.8, head_width=4, color=cmap(0.25 + 0.75 * gn[i]), length_includes_head=True)
        ax.text(pc[0] + 2, pc[1] + 2, str(int(c)), fontsize=8, color=ST.INK)
        if c == p10:
            ax.scatter([pc[0]], [pc[1]], s=170, facecolors="none", edgecolors=ST.BLUE[10], lw=2.2)
        if c == p100:
            ax.scatter([pc[0]], [pc[1]], s=300, facecolors="none", edgecolors="#d03b3b", lw=2.2)
        if g[i] >= g.max() - X.np.float64(1e-2):
            ax.scatter([pc[0]], [pc[1]], marker="*", s=160, color="#1baf7a", zorder=5)
    ax.scatter([oc[0]], [oc[1]], s=50, color=ST.INK, marker="s")
    ax.set_title("candidates (arrow = view direction, darker = larger oracle gain); blue ring = sparse-scoring pick (S10 = B2),\n"
                 "red ring = dense-scoring pick (S100), green star = oracle best, black square = current camera", loc="left", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(gs[0, 2:4])
    xs = np.arange(len(C))
    ax.bar(xs - 0.27, s10[C] / max(1e-9, s10[C].max()), 0.26, color=ST.BLUE[9], label="DistP1, sparse scoring (S10)")
    ax.bar(xs, s100[C] / max(1e-9, s100[C].max()), 0.26, color="#d03b3b", label="DistP1, dense scoring (S100)")
    ax.bar(xs + 0.27, g / max(1e-9, g.max()), 0.26, color="#1baf7a", label="oracle coverage gain")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(c)) for c in C])
    ax.set_xlabel("candidate")
    ax.set_ylabel("normalised to the max of each series")
    ax.legend(fontsize=8)
    ST.clean(ax)
    ax.set_title("same D100 predictions, two scoring maps", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[0, 4:6])
    ax.axis("off")
    loc = {int(c): i for i, c in enumerate(C)}
    txt = (f"{nm} {st['scene']} s{st['start']} t={st['t']} ({row['phase']})\n\nsparse-scoring pick {p10}: gain {g[loc[p10]]:.3f} pp, regret {g.max() - g[loc[p10]]:.3f}\n"
           f"dense-scoring pick {p100}: gain {g[loc[p100]]:.3f} pp, regret {g.max() - g[loc[p100]]:.3f}\noracle best: {int(C[np.argmax(g)])} ({g.max():.3f} pp)\n\n"
           f"stored rows: S10 regret {row['S10_regret']:.3f}, S100 regret {row['S100_regret']:.3f}\ncoverage eps = {eps_rt:.3f} runtime units")
    ax.text(0, 0.95, txt, va="top", fontsize=10)
    cmd = plt.get_cmap("viridis_r").copy()
    cmd.set_bad("white")
    for r_, (c, lab) in enumerate(((p10, "sparse-scoring pick"), (p100, "dense-scoring pick"))):
        v = views[c]
        dzn = v["dz"].double().cpu().numpy()
        tp = hits[c][0]
        zp = tp.double().cpu().numpy() * k_M * dzn
        ts = M.orc["tstar"][c].astype(np.float64)
        zg = ts * k_M * dzn
        ok = np.isfinite(ts) & (zg < MC.SENSOR_RANGE)
        vals = np.concatenate([zp[np.isfinite(zp)], zg[ok]]) / u
        vmin, vmax = np.quantile(vals, 0.02), np.quantile(vals, 0.98)
        h10, _, d10 = X.voxel_flags(M.geo[10], v, tp)
        h100, _, d100 = X.voxel_flags(M.geo[100], v, tp)
        new_t = tr[c][0] if c in tr else np.zeros(len(ts), bool)
        panels = [(np.where(np.isfinite(zp), zp / u, np.nan), f"{lab} {c}: predicted depth (D100 input, sample k=0)"),
                  ("nov10", f"novelty with the SPARSE band_dil: coloured = new ({int((h10 & ~d10).sum())} rays), grey = excluded"),
                  ("nov100", f"novelty with the DENSE band_dil: coloured = new ({int((h100 & ~d100).sum())} rays), grey = excluded"),
                  ("only", f"excluded ONLY by the dense map: red = GT surface truly new ({int((d100 & ~d10 & new_t & ok).sum())}), dark grey = truly old ({int((d100 & ~d10 & ~(new_t & ok)).sum())})"),
                  (np.where(ok, zg / u, np.nan), f"reference: TRUE depth of candidate {c}"),
                  ("truth", "reference: GT surface new (green) / already observed (grey)")]
        for k_, (p, title) in enumerate(panels):
            ax = fig.add_subplot(gs[1 + r_, k_])
            if isinstance(p, str):
                im = np.ones((GH * GW, 3))
                if p in ("nov10", "nov100"):
                    hh, dd = (h10, d10) if p == "nov10" else (h100, d100)
                    col = cmd((np.clip(zp / u, vmin, vmax) - vmin) / max(1e-9, vmax - vmin))[:, :3]
                    im[hh & ~dd] = col[hh & ~dd]
                    im[dd] = to_rgb(ST.KNOWN_GRAY)
                elif p == "only":
                    o = d100 & ~d10
                    im[o & ~(new_t & ok)] = (0.35, 0.35, 0.35)
                    im[o & new_t & ok] = (0.82, 0.23, 0.23)
                else:
                    im[ok] = to_rgb(ST.KNOWN_GRAY)
                    im[new_t & ok] = (0.11, 0.69, 0.48)
                ax.imshow(im.reshape(GH, GW, 3), interpolation="nearest")
            else:
                ax.imshow(p.reshape(GH, GW), cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_title(textwrap.fill(title, 52), loc="left", fontsize=8.5)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"Task 3 / E1.2 Figure 5 — scoring disagreement: the SAME D100 predictions scored with the sparse (10%) and the dense (100%) band_dil "
                 f"({nm} {st['scene']} s{st['start']} t={st['t']}). Depth colours: yellow near -> purple far ({uname}).", x=0.005, ha="left", fontsize=11)
    tag = f"{dataset}_{st['scene']}_s{st['start']}_t{st['t']}"
    fig.savefig(os.path.join(FIGD, f"fig5_{tag}.png"), dpi=100)
    plt.close(fig)
    return {"fig5": f"fig5_{tag}.png", "pick_sparse": int(p10), "pick_dense": int(p100),
            "picks_match_analysis": [int(p10) == row["S10_pick"], int(p100) == row["S100_pick"]]}


STATES = os.path.join(os.path.dirname(HERE), "analysis", "e12_states.json")
ST = None


def main():
    global ST
    kind, args = sys.argv[1], sys.argv[2:]
    out_p = os.path.join(FIGD, "figures.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for i in range(0, len(args), 3):
        d, stratum, sid = args[i:i + 3]
        X.setup(stratum)
        import ab_style as ST_
        ST = ST_
        ST.setup()
        dev = X.torch.device("cuda")
        core = X.S2.load_cores(dev)["V2M"]
        res = fig34(d, stratum, sid, core, dev) if kind == "fig34" else fig5(d, stratum, sid, core, dev)
        out[f"{kind}|{d}|{sid}"] = res
        print(kind, d, sid, json.dumps(res, default=float), flush=True)
        X.torch.cuda.empty_cache()
    json.dump(out, open(out_p, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
