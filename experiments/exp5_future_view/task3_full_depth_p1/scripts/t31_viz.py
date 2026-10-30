"""EXP5 Task 3 / E1 -- Sparse-P1 vs FullDepth-P1 on the same history (seen2scene env, 1 GPU), in the style of
ablation_study/A2_stochastic_future_views/figures/v2/input_<d>_<case>.png.

For a fixed state of the Task 2 V2-Mix trajectory, both maps are rebuilt from the same stored observations with the
planner's rules. Figure (3 rows x 5 columns):
  row 1  Sparse-P1 (B0):     fine horizontal slice | zoom | coarse 8^3 slice (3D encoder input) | t_known of the current
                             camera | t_known of the candidate (the V2-Mix pick)
  row 2  FullDepth-P1 (B1):  the same
  row 3  reference, not inputs: numbers | true depth current | true depth candidate | rays where P1's first known surface
         lies in front of the true first surface (B0 orange, B1 red; current and candidate)
Output figures/compare_<dataset>_<scene>_s<start>_t<t>.png + figures/compare.json
Usage: python t31_viz.py DATASET SCENE START T [DATASET SCENE START T ...]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "ablation_study", "common"))
import ab_common as A  # noqa: E402
import s2_infer as S2  # noqa: E402
import ab_style as ST  # noqa: E402

sys.path.insert(0, HERE)
import p1_full as PF  # noqa: E402

np, torch = S2.np, S2.torch
MC, G, PC, m_p1 = S2.MC, S2.G, S2.PC, S2.m_p1
plt = ST.plt
GH, GW = 64, 114
FIG = os.path.join(os.path.dirname(HERE), "geometry_diagnostics", "figures")
C_EMPTY, C_BAND, C_UNK = (0.62, 0.77, 0.96), (0.05, 0.05, 0.05), (0.92, 0.91, 0.89)


def find_state(dataset, scene, start, t):
    for st in A.load_manifest(dataset)["states"]:
        if st["scene"] == scene and st["start"] == start and st["t"] == t and any(e["source"] == "V2M" for e in st["sources"]):
            return st
    raise KeyError((dataset, scene, start, t))


def build_maps(dataset, st, dev):
    """Both P1 maps at state t of the V2-Mix trajectory (planner rules: indoor region of the last refresh, all records)."""
    ep = S2.episode_of(dataset, st)
    rp = S2.StateReplay(dataset, ep, "V2M", {}, dev)
    recs = [MC.load_service_record(A.svc_path(dataset, "V2M", ep, t)) for t in range(st["t"] + 1)]
    if dataset == "indoor":
        tr = 10 * (st["t"] // 10)
        fr, dims, dom, win, box = rp.F.region_geometry(rp.card, recs[tr]["cam_center"])
    else:
        fr, dims, dom, _ = rp.frame
    out = {}
    for m, cls in (("s", PF._P1), ("f", PF.P1FusionFull)):
        fu = cls(fr, dims, dom, device=dev)
        for r in recs:
            fu.integrate_record(r)
        out[m] = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        del fu
    return out, recs


def ray_maps(geo, R, T, ztrue, u):
    v = geo.view(R, T)
    dz = v["dz"].double().cpu().numpy()
    k_M = float(geo.fr.k_M)
    zk = v["pr"]["t_known"].double().cpu().numpy() * k_M * dz
    tol = MC.TRUNC * k_M + 0.01 * np.where(np.isfinite(ztrue), ztrue, 0.0)
    gt = np.isfinite(ztrue) & (ztrue < MC.SENSOR_RANGE)
    known = np.isfinite(zk)
    pseudo = known & gt & (zk < ztrue - tol)
    return (np.where(known, zk / u, np.nan).reshape(GH, GW), pseudo.reshape(GH, GW),
            {"t_known_valid": float(known[gt].mean()) if gt.any() else None,
             "pseudo_front": float(pseudo[known & gt].mean()) if (known & gt).any() else None})


def slice_img(geo, kz):
    s = geo.struct[:, :, kz].cpu().numpy()
    dm = geo.domain[:, :, kz].cpu().numpy()
    img = np.ones(s.shape + (3,))
    img[(s == G.UNKNOWN) & dm] = C_UNK
    img[s == G.EMPTY] = C_EMPTY
    img[s == G.BAND] = C_BAND
    return img, s


def coarse_img(geo, kz):
    c = geo.coarse().cpu().numpy().astype(np.float32) / 255.0
    ce, cb = c[0][:, :, kz // 8], c[1][:, :, kz // 8]
    img = np.ones(ce.shape + (3,)) * (1 - ce[..., None]) + np.asarray(C_EMPTY)[None, None] * ce[..., None]
    b = np.clip(cb * 8, 0, 1)[..., None]
    return img * (1 - b) + np.asarray(C_BAND)[None, None] * b


def figure(dataset, scene, start, t, dev):
    u, uname = (7.0, "m") if dataset == "indoor" else (1.0, "scene units")
    st = find_state(dataset, scene, start, t)
    geos, recs = build_maps(dataset, st, dev)
    rec = recs[-1]
    v2 = json.load(open(os.path.join(A.state_dir(dataset, st["sid"]), "v2.json")))
    C = np.asarray(st["cand"])
    sc = np.asarray(v2["models"]["V2M"]["distp1"])[C]
    j = int(C[int(np.flatnonzero(sc == sc.max())[0])])
    orc = np.load(os.path.join(A.state_dir(dataset, st["sid"]), "oracle.npz"))
    uu, vv = MC.scoring_pixel_grid()
    ui, vi = np.round(uu).astype(int), np.round(vv).astype(int)
    zc = np.where(np.asarray(rec["mask"], bool)[vi, ui], np.asarray(rec["zbuf"], np.float64)[vi, ui], np.inf)
    vj = geos["s"].view(rec["legal_R"][j], rec["legal_T"][j])
    zj = orc["tstar"][j].astype(np.float64) * float(geos["s"].fr.k_M) * vj["dz"].double().cpu().numpy()
    maps, stats = {}, {}
    for m in ("s", "f"):
        a = ray_maps(geos[m], rec["R"], rec["T"], zc, u)
        b = ray_maps(geos[m], rec["legal_R"][j], rec["legal_T"][j], zj, u)
        maps[m] = (a, b)
        stats[m] = {"cur": a[2], "cand": b[2], "band": int((geos[m].struct == G.BAND).sum()),
                    "empty": int((geos[m].struct == G.EMPTY).sum())}
    oc = geos["s"].fr.runtime_to_model(np.asarray(rec["cam_center"]).reshape(1, 3))[0] / G.VS
    s0 = geos["s"].struct
    kz = int(np.clip(oc[2], 0, s0.shape[2] - 1))
    if ((s0[:, :, kz] == G.EMPTY) | (s0[:, :, kz] == G.BAND)).sum().item() < 500:
        Pz = geos["s"].fr.runtime_to_model(np.asarray(rec["sample_points"], np.float64))[:, 2] / G.VS
        kz = int(np.clip(np.median(Pz) + 6, 0, s0.shape[2] - 1))
    imgs = {m: slice_img(geos[m], kz) for m in ("s", "f")}
    known = (imgs["s"][1] == G.EMPTY) | (imgs["s"][1] == G.BAND) | (imgs["f"][1] == G.BAND)
    xs, ys = np.nonzero(known)
    x0, x1, y0, y1 = max(0, xs.min() - 15), min(known.shape[0], xs.max() + 15), max(0, ys.min() - 15), min(known.shape[1], ys.max() + 15)
    bx, by = np.nonzero(imgs["s"][1] == G.BAND)
    cx, cy = (int(np.median(bx)), int(np.median(by))) if len(bx) else (int(oc[0]), int(oc[1]))
    zr = 50
    cmd = plt.get_cmap("viridis_r").copy()
    cmd.set_bad("white")
    vals = np.concatenate([x[np.isfinite(x)] for x in (zc / u, zj / u)])
    vmin, vmax = np.quantile(vals, 0.02), np.quantile(vals, 0.98)
    fig = plt.figure(figsize=(21, 12.5))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.06, left=0.01, right=0.99, top=0.9, bottom=0.03)
    for r, (m, lab) in enumerate((("s", "Sparse-P1 (B0)"), ("f", "FullDepth-P1 (B1)"))):
        img = imgs[m][0]
        ax = fig.add_subplot(gs[r, 0])
        ax.imshow(np.transpose(img[x0:x1, y0:y1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[x0, x1, y0, y1])
        ax.scatter([oc[0]], [oc[1]], s=40, color="#d03b3b")
        ax.add_patch(plt.Rectangle((cx - zr, cy - zr), 2 * zr, 2 * zr, fill=False, ec="#d03b3b", lw=1.2))
        ax.set_title(f"{lab}: fine P1 slice z={kz}\nblue EMPTY, black BAND ({stats[m]['band']:,} voxels), gray UNKNOWN", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[r, 1])
        zx0, zx1, zy0, zy1 = max(0, cx - zr), min(img.shape[0], cx + zr), max(0, cy - zr), min(img.shape[1], cy + zr)
        ax.imshow(np.transpose(img[zx0:zx1, zy0:zy1], (1, 0, 2)), origin="lower", interpolation="nearest", extent=[zx0, zx1, zy0, zy1])
        ax.set_title(f"{lab}: zoom (red box)", loc="left", fontsize=9)
        ax = fig.add_subplot(gs[r, 2])
        ci = coarse_img(geos[m], kz)
        ax.imshow(np.transpose(ci[x0 // 8:x1 // 8 + 1, y0 // 8:y1 // 8 + 1], (1, 0, 2)), origin="lower", interpolation="nearest")
        ax.set_title(f"{lab}: 3D encoder input (8^3 blocks)\nblue EMPTY fraction, dark BAND fraction x8", loc="left", fontsize=9)
        (zk_c, _, sc_c), (zk_j, _, sc_j) = maps[m]
        ax = fig.add_subplot(gs[r, 3])
        ax.imshow(zk_c, cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{lab}: INPUT t_known, current camera\nvalid on {100 * sc_c['t_known_valid']:.0f}% of the rays with a true surface",
                     loc="left", fontsize=9)
        ax = fig.add_subplot(gs[r, 4])
        ax.imshow(zk_j, cmap=cmd, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{lab}: INPUT t_known, candidate j={j} (V2M pick)\nvalid on {100 * sc_j['t_known_valid']:.0f}% of the rays with a "
                     f"true surface", loc="left", fontsize=9)
    ax = fig.add_subplot(gs[2, 0])
    ax.axis("off")
    txt = (f"{'MP3D' if dataset == 'indoor' else 'Macarons++'} {scene} start {start} t = {t}\n\n"
           f"BAND voxels:  B0 {stats['s']['band']:,}   B1 {stats['f']['band']:,}\n"
           f"EMPTY voxels: B0 {stats['s']['empty']:,}   B1 {stats['f']['empty']:,}\n\n"
           f"t_known valid (current):   B0 {100 * stats['s']['cur']['t_known_valid']:.0f}%  B1 {100 * stats['f']['cur']['t_known_valid']:.0f}%\n"
           f"t_known valid (candidate): B0 {100 * stats['s']['cand']['t_known_valid']:.0f}%  B1 {100 * stats['f']['cand']['t_known_valid']:.0f}%\n\n"
           f"pseudo surface in front of the true one\n(share of known-surface rays):\n"
           f"  current:   B0 {100 * (stats['s']['cur']['pseudo_front'] or 0):.1f}%  B1 {100 * (stats['f']['cur']['pseudo_front'] or 0):.1f}%\n"
           f"  candidate: B0 {100 * (stats['s']['cand']['pseudo_front'] or 0):.1f}%  B1 {100 * (stats['f']['cand']['pseudo_front'] or 0):.1f}%")
    ax.text(0.0, 0.95, txt, va="top", ha="left", fontsize=10, family="DejaVu Sans")
    for c, (zt, lab) in enumerate(((zc, "current camera"), (zj, f"candidate j={j}"))):
        ax = fig.add_subplot(gs[2, 1 + c])
        ax.imshow(np.where(np.isfinite(zt) & (zt < MC.SENSOR_RANGE), zt / u, np.nan).reshape(GH, GW), cmap=cmd, vmin=vmin, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(f"reference (NOT an input): TRUE depth, {lab}", loc="left", fontsize=9)
    for c, k in enumerate((0, 1)):
        ax = fig.add_subplot(gs[2, 3 + c])
        img = np.ones((GH, GW, 3))
        img[maps["s"][k][1]] = (0.93, 0.52, 0.35)
        img[maps["f"][k][1]] = (0.82, 0.23, 0.23)
        img[maps["s"][k][1] & maps["f"][k][1]] = (0.35, 0.05, 0.05)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"rays whose P1 known surface lies IN FRONT of the true one\n({'current' if k == 0 else f'candidate j={j}'}): "
                     f"orange B0 only, red B1 only, dark both", loc="left", fontsize=9)
    for a in fig.axes:
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Task 3 / E1 — Sparse-P1 (B0) vs FullDepth-P1 (B1) on the SAME observation history ({'MP3D' if dataset == 'indoor' else 'Macarons++'} "
                 f"{scene} s{start} t={t}, V2-Mix trajectory). Depth colours: yellow near -> purple far ({uname}); white = none.",
                 x=0.005, ha="left", fontsize=11)
    os.makedirs(FIG, exist_ok=True)
    fn = f"compare_{dataset}_{scene}_s{start}_t{t}.png"
    fig.savefig(os.path.join(FIG, fn), dpi=110)
    plt.close(fig)
    return fn, {"sid": st["sid"], "candidate": j, **{m: stats[m] for m in ("s", "f")}}


def main():
    dev = torch.device("cuda")
    args = sys.argv[1:]
    pj = os.path.join(FIG, "compare.json")
    out = json.load(open(pj)) if os.path.exists(pj) else {}
    for i in range(0, len(args), 4):
        d, s, k, t = args[i], args[i + 1], int(args[i + 2]), int(args[i + 3])
        fn, st = figure(d, s, k, t, dev)
        out[fn] = st
        print(fn, json.dumps(st), flush=True)
        torch.cuda.empty_cache()
    os.makedirs(FIG, exist_ok=True)
    json.dump(out, open(pj, "w"), indent=1)


if __name__ == "__main__":
    main()
