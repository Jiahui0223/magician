"""EXP5 Ablation A2 -- what V2 actually receives as input (seen2scene env, 1 GPU; user question 2026-10-28).

V2 sees no depth image and no RGB. Its inputs are built from the planner map P1 (frozen m_p1):
  (1) 3D encoder: the coarse grid of P1 (8 x 8 x 8 voxel blocks: fraction EMPTY, fraction BAND, fraction in domain, height)
  (2) per candidate ray (64 x 114): 11 projection channels of the ray cast through the fine P1 (class, t_known, start
      d_lo / end d_hi / length of the UNKNOWN part, #UNKNOWN segments, direction z, camera z, range)
  (3) ray context: encoder features sampled at 32 points along each ray
Figure, for one fixed state:
  row 1  the map: observed 10 % points, a horizontal slice of the fine P1 at camera height (EMPTY / BAND / UNKNOWN), a
         zoom, and the same slice of the coarse grid (what the 3D encoder sees)
  row 2  per-ray channels of the CURRENT camera (the one that just observed): true depth (reference, not an input),
         t_known (input), #UNKNOWN segments (input), and -- not an input channel -- where the first free-space run of
         the ray ends in P1 (to show that the wall position is in the map but not handed to the ray channels)
Output figures/v2/input_<dataset>_<case>.png + results/input_view.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import ab_common as A  # noqa: E402
import s2_infer as S2  # noqa: E402
import ab_style as ST  # noqa: E402

np, torch = S2.np, S2.torch
MC, G = S2.MC, S2.G
plt = ST.plt
GH, GW = 64, 114
C_EMPTY, C_BAND, C_UNK = (0.62, 0.77, 0.96), (0.05, 0.05, 0.05), (0.92, 0.91, 0.89)


@torch.no_grad()
def free_space_end(geo, v):
    """Per ray: distance (model units) where the first EMPTY run along the ray ends (first non-EMPTY sample after the
    first EMPTY sample), inf if the ray has no EMPTY sample. Not a V2 input -- shows what the map contains."""
    pr, o, d = v["pr"], v["o"], v["d"]
    s = geo.struct
    dims = s.shape
    out = torch.full((d.shape[0],), float("inf"), device=d.device, dtype=d.dtype)
    for a in range(0, d.shape[0], 512):
        sl = slice(a, min(d.shape[0], a + 512))
        ts = G.sample_ts(pr["tmin"][sl], pr["M"])
        p = o[None, None] + d[sl, None] * ts[..., None]
        ijk = torch.floor(p / G.VS).long()
        inside = (ts <= pr["tmax"][sl, None]) & (ijk >= 0).all(-1) & (ijk[..., 0] < dims[0]) & (ijk[..., 1] < dims[1]) & (ijk[..., 2] < dims[2])
        code = torch.full(ts.shape, -1, dtype=torch.int8, device=d.device)
        q = ijk[inside]
        code[inside] = s[q[:, 0], q[:, 1], q[:, 2]]
        emp = code == G.EMPTY
        has = emp.any(1)
        first = torch.argmax(emp.int(), 1)
        idx = torch.arange(ts.shape[1], device=d.device)[None]
        after = (idx > first[:, None]) & ~emp
        has_end = has & after.any(1)
        j = torch.argmax(after.int(), 1)
        out[sl] = torch.where(has_end, ts.gather(1, j[:, None])[:, 0], out[sl])
    return out


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    cores = S2.load_cores(dev)
    cases = json.load(open(os.path.join(HERE, "results", "cases.json")))
    res = {}
    for dataset, case in (("outdoor", "success"), ("indoor", "stochastic_nondegenerate")):
        u, uname = (7.0, "m") if dataset == "indoor" else (1.0, "scene units")
        sid = cases[dataset][case]["sid"]
        st = next(s for s in A.load_manifest(dataset)["states"] if s["sid"] == sid)
        src = st["sources"][0]["source"]
        ep = S2.episode_of(dataset, st)
        rz = np.load(os.path.join(A.W, "cases_v2", f"{dataset}_{sid}.npz"))
        recs = [np.load(A.svc_path(dataset, src, ep, t)) for t in range(st["t"] + 1)]
        rec = recs[-1]
        for t, rec2, geo, h, eps in S2.StateReplay(dataset, ep, src, cores, dev).run([st["t"]]):
            k_M, vs_u = float(geo.fr.k_M), G.VS * float(geo.fr.k_M) / u
            v = geo.view(rec["R"], rec["T"])
            pr, dz = v["pr"], v["dz"].double().cpu().numpy()
            zf = lambda x: np.where(np.isfinite(x), x * k_M * dz / u, np.nan)          # noqa: E731
            tk = zf(pr["t_known"].double().cpu().numpy()).reshape(GH, GW)
            fse = zf(free_space_end(geo, v).double().cpu().numpy()).reshape(GH, GW)
            dlo = zf(pr["d_lo"].double().cpu().numpy()).reshape(GH, GW)
            nseg = pr["n_seg"].cpu().numpy().reshape(GH, GW)
            s = geo.struct.cpu().numpy()
            dom = geo.domain.cpu().numpy()
            coarse = geo.coarse().cpu().numpy().astype(np.float32) / 255.0
            oc = geo.fr.runtime_to_model(np.asarray(rec["cam_center"]).reshape(1, 3))[0] / G.VS
            P = geo.fr.runtime_to_model(np.concatenate([r["sample_points"] for r in recs]).astype(np.float64)) / G.VS
        uu, vv = MC.scoring_pixel_grid()
        zt = rz["z_cur"][np.round(vv).astype(int), np.round(uu).astype(int)].astype(np.float64)
        zt = np.where((zt > 0) & (zt < MC.SENSOR_RANGE), zt / u, np.nan).reshape(GH, GW)
        seen = np.isfinite(zt)
        m = seen & np.isfinite(fse)
        stats = {"rays_true_surface": int(seen.sum()), "t_known_found": float(np.mean(np.isfinite(tk[seen]))),
                 "free_space_end_found": float(np.mean(np.isfinite(fse[seen]))),
                 "free_space_end_minus_true_median": float(np.median(fse[m] - zt[m])) if m.any() else None,
                 "free_space_end_abs_err_p90": float(np.quantile(np.abs(fse[m] - zt[m]), 0.9)) if m.any() else None,
                 "n_seg_ge2": float(np.mean(nseg[seen] >= 2)), "d_lo_median": float(np.nanmedian(dlo[seen])),
                 "voxel": vs_u, "coarse_cell": 8 * vs_u, "unit": uname,
                 "slice_counts": None}
        # ---- slice of the fine map at camera height
        kz = int(np.clip(oc[2], 0, s.shape[2] - 1))
        if ((s[:, :, kz] == G.EMPTY) | (s[:, :, kz] == G.BAND)).sum() < 500:        # aerial view: slice at the observed surface
            Pz = geo.fr.runtime_to_model(np.asarray(rec["sample_points"], np.float64))[:, 2] / G.VS
            kz = int(np.clip(np.median(Pz) + 6, 0, s.shape[2] - 1))
        sl, dm = s[:, :, kz], dom[:, :, kz]
        img = np.ones(sl.shape + (3,))
        img[(sl == G.UNKNOWN) & dm] = C_UNK
        img[sl == G.EMPTY] = C_EMPTY
        img[sl == G.BAND] = C_BAND
        stats["slice_counts"] = {"EMPTY": int((sl == G.EMPTY).sum()), "BAND": int((sl == G.BAND).sum()),
                                 "UNKNOWN": int(((sl == G.UNKNOWN) & dm).sum())}
        xs, ys = np.nonzero((sl == G.EMPTY) | (sl == G.BAND))
        x0, x1 = max(0, xs.min() - 20), min(sl.shape[0], xs.max() + 20)
        y0, y1 = max(0, ys.min() - 20), min(sl.shape[1], ys.max() + 20)
        bx, by = np.nonzero(sl == G.BAND)
        cx, cy = (int(np.median(bx)), int(np.median(by))) if len(bx) else (int(oc[0]), int(oc[1]))
        zr = 60
        res[f"{dataset}|{case}"] = dict(stats, sid=sid, scene=st["scene"], t=st["t"])
        # ---- figure
        fig = plt.figure(figsize=(20, 10.5))
        gs = fig.add_gridspec(2, 4, hspace=0.28, wspace=0.08, left=0.01, right=0.99, top=0.88, bottom=0.03)
        ax = fig.add_subplot(gs[0, 0])
        band = np.abs(P[:, 2] - kz) < 3
        ax.scatter(P[band, 0], P[band, 1], s=0.6, color=ST.INK2, lw=0)
        ax.scatter([oc[0]], [oc[1]], marker="o", s=40, color="#d03b3b")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_title(f"observed 10% points within +-3 voxels of the slice height\n(top view; slice z = {kz}, camera z = {oc[2]:.0f} voxels)", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(np.transpose(img[x0:x1, y0:y1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[x0, x1, y0, y1])
        ax.scatter([oc[0]], [oc[1]], marker="o", s=40, color="#d03b3b")
        ax.add_patch(plt.Rectangle((cx - zr, cy - zr), 2 * zr, 2 * zr, fill=False, ec="#d03b3b", lw=1.2))
        ax.set_title(f"fine P1 slice (voxel {vs_u:.3f} {uname}): blue EMPTY (full depth),\nblack BAND (10% points), gray UNKNOWN",
                     loc="left", fontsize=9)
        ax = fig.add_subplot(gs[0, 2])
        zx0, zx1, zy0, zy1 = max(0, cx - zr), min(sl.shape[0], cx + zr), max(0, cy - zr), min(sl.shape[1], cy + zr)
        ax.imshow(np.transpose(img[zx0:zx1, zy0:zy1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[zx0, zx1, zy0, zy1])
        ax.set_title("zoom (red box): EMPTY stops ~3 voxels before the\nwall; BAND = short dashes (one per 10% point)",
                     loc="left", fontsize=9)
        ax = fig.add_subplot(gs[0, 3])
        ck = kz // 8
        ce, cb = coarse[0][:, :, ck], coarse[1][:, :, ck]
        cimg = np.ones(ce.shape + (3,))
        cimg = cimg * (1 - ce[..., None]) + np.asarray(C_EMPTY)[None, None] * ce[..., None]
        cimg = cimg * (1 - np.clip(cb * 8, 0, 1)[..., None]) + np.asarray(C_BAND)[None, None] * np.clip(cb * 8, 0, 1)[..., None]
        ax.imshow(np.transpose(cimg[x0 // 8:x1 // 8 + 1, y0 // 8:y1 // 8 + 1], (1, 0, 2)), origin="lower", interpolation="nearest",
                  extent=[x0 // 8, x1 // 8 + 1, y0 // 8, y1 // 8 + 1])
        ax.set_title(f"what the 3D encoder sees: the same slice in 8x8x8 blocks\n(cell {8 * vs_u:.2f} {uname}; blue = EMPTY fraction, "
                     f"dark = BAND fraction x8)", loc="left", fontsize=9)
        for a in fig.axes:
            a.set_xticks([])
            a.set_yticks([])
        cmd = plt.get_cmap("viridis_r").copy()
        cmd.set_bad("white")
        vals = np.concatenate([x[np.isfinite(x)] for x in (zt, fse, tk)])
        vmin, vmax = np.quantile(vals, 0.02), np.quantile(vals, 0.98)
        ax = fig.add_subplot(gs[1, 0])
        ax.imshow(zt, cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title("current camera: TRUE depth on the 64 x 114 rays\n(reference; NOT an input)", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[1, 1])
        ax.imshow(tk, cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"INPUT channel t_known: first P1 known surface on the ray\npresent on {100 * stats['t_known_found']:.0f}% of the rays "
                     f"(white = none)", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[1, 2])
        ax.imshow(np.clip(nseg, 0, 3), cmap=plt.matplotlib.colors.ListedColormap(["white", "#c6d7ee", "#6da7ec", "#1c5cab"]),
                  vmin=-0.5, vmax=3.5, interpolation="nearest")
        ax.set_title(f"INPUT #UNKNOWN segments (white 0 ... dark 3+)\n{100 * stats['n_seg_ge2']:.0f}% of rays have >= 2",
                     loc="left", fontsize=9)
        ax = fig.add_subplot(gs[1, 3])
        ax.imshow(fse, cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"NOT an input: where P1's free space ends\n(median {stats['free_space_end_minus_true_median']:.2f} {uname} "
                     f"vs true depth)", loc="left", fontsize=9)
        for a in fig.axes[-4:]:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(f"{'MP3D' if dataset == 'indoor' else 'Macarons++'} {st['scene']} start {st['start']} t = {st['t']}: what V2 receives. "
                     f"No depth image, no RGB: (1) the P1 map in 8x8x8 blocks -> 3D encoder; (2) 11 channels per candidate ray from "
                     f"casting through the fine P1; (3) encoder features along the ray.\nRow 2 uses the CURRENT camera (it measured "
                     f"every pixel of this view). Depth colours: yellow near -> purple far ({uname}).", x=0.005, ha="left", fontsize=10.5)
        fig.savefig(os.path.join(HERE, "figures", "v2", f"input_{dataset}_{case}.png"), dpi=120)
        plt.close(fig)
        print(dataset, case, json.dumps(res[f"{dataset}|{case}"]), flush=True)
    json.dump(res, open(os.path.join(HERE, "results", "input_view.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
