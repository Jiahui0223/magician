"""EXP5 Ablation A2 -- diagnostic of the 'gray' (observed) pixels (seen2scene env, 1 GPU; user question 2026-10-28).

Why is the current depth dense while the observed (gray) part of a candidate view is speckled?  Two separate checks:
  (1) coordinate transform: every observed point of records 0..t (the sim's 10 % samples, world y-up runtime units)
      projected into the current and the candidate camera with the frozen MAGICIAN camera model (m_common: view =
      X R + T, NDC of pixel_rays inverted), z-buffered, compared with the rendered depth of that camera at the pixel
      (cases_v2 renders): |dz| / z <= 2 % = consistent, deeper = occluded in this view, shallower = inconsistent
  (2) the planner's map: the 64 x 114 scoring rays of the current and the candidate camera cast against P1 (the
      frozen v2_plan_core Geometry.view of the rebuilt planner state): ray reaches a P1 known surface without passing
      UNKNOWN (gray), ray passes UNKNOWN voxels (P1 hole / unseen), or neither
Output figures/v2/diag_gray_<dataset>_<case>.png + results/diag_gray.json
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


def project(P, R, T):
    """World (y-up runtime) points -> pixel (u, v) and view depth z (frozen MAGICIAN camera, inverse of pixel_rays)."""
    R = np.asarray(R, np.float64).reshape(3, 3)
    T = np.asarray(T, np.float64).reshape(3)
    view = np.asarray(P, np.float64) @ R + T
    z = view[:, 2]
    nx, ny = view[:, 0] * MC.S_PROJ / z, view[:, 1] * MC.S_PROJ / z
    u = (MC.W / MC.M_SHORT - nx) * (MC.M_SHORT - 1) / 2.0
    v = (MC.H / MC.M_SHORT - ny) * (MC.M_SHORT - 1) / 2.0
    return u, v, z


def splat(P, R, T, zref):
    """z-buffered projection of P; per pixel the nearest point; classification against the rendered depth zref."""
    u, v, z = project(P, R, T)
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (z > 0.5) & (ui >= 0) & (ui < MC.W) & (vi >= 0) & (vi < MC.H)
    ui, vi, z = ui[ok], vi[ok], z[ok]
    flat = vi * MC.W + ui
    order = np.lexsort((z, flat))
    flat, z = flat[order], z[order]
    first = np.r_[True, flat[1:] != flat[:-1]]
    flat, z = flat[first], z[first]
    zr = zref.reshape(-1)[flat]
    rel = np.where(zr > 0, (z - zr) / np.maximum(zr, 1e-6), np.inf)
    cls = np.where(np.abs(rel) <= 0.02, 1, np.where(rel > 0.02, 2, 3))          # 1 consistent, 2 occluded, 3 inconsistent
    img = np.zeros(MC.H * MC.W, np.int8)
    img[flat] = cls
    return img.reshape(MC.H, MC.W), {"projected_pixels": int(len(flat)), "consistent": float(np.mean(cls == 1)),
                                     "occluded": float(np.mean(cls == 2)), "inconsistent": float(np.mean(cls == 3))}


def p1_view(geo, R, T, zbuf=None):
    """Planner casting of the 64 x 114 rays against P1: 1 = the ray reaches a P1 known (BAND) surface (t_known finite),
    2 = no known surface but UNKNOWN voxels on the ray (P1 hole or genuinely unseen), 0 = neither (EMPTY up to range),
    -1 = invalid ray. Also the UNKNOWN length (voxels) in front of the known surface."""
    v = geo.view(R, T)
    pr = v["pr"]
    known = torch.isfinite(pr["t_known"])
    unk = pr["n_u"] > 0
    cls = torch.zeros_like(pr["n_u"], dtype=torch.int8)
    cls[known] = 1
    cls[~known & unk] = 2
    cls[~pr["valid"]] = -1
    cls = cls.cpu().numpy()
    lu = (pr["n_u"].cpu().numpy() * 0.5)
    out = {"rays": int(len(cls)), "known_surface": float(np.mean(cls == 1)), "no_known_surface_unknown": float(np.mean(cls == 2)),
           "no_surface": float(np.mean(cls == 0)), "invalid": float(np.mean(cls == -1)),
           "unknown_shell_vox_median": float(np.median(lu[cls == 1])) if (cls == 1).any() else None}
    if zbuf is not None:                       # rays whose pixel this camera itself measured (true depth < range)
        u, vv = MC.scoring_pixel_grid()
        zb = zbuf[np.round(vv).astype(int), np.round(u).astype(int)]
        seen = (zb > 0) & (zb < MC.SENSOR_RANGE)
        out["self_seen_rays"] = int(seen.sum())
        out["self_seen_p1_known"] = float(np.mean(cls[seen] == 1)) if seen.any() else None
        out["self_seen_p1_hole"] = float(np.mean(cls[seen] == 2)) if seen.any() else None
    return cls.reshape(GH, GW), out


def figure(dataset, st, rows, path):
    u = 7.0 if dataset == "indoor" else 1.0
    fig, axs = plt.subplots(2, 4, figsize=(19, 7.2), layout="constrained")
    cm_pts = plt.matplotlib.colors.ListedColormap(["white", "#7d7c76", "#c6d7ee", "#d03b3b"])
    cm_p1 = plt.matplotlib.colors.ListedColormap(["#f6d5d5", "white", "#7d7c76", "#9ec5f4"])
    for i, (lab, rgb, z, pimg, pst, cimg, cst) in enumerate(rows):
        axs[i, 0].imshow(rgb)
        axs[i, 0].set_title(f"{lab}: rendered RGB", loc="left", fontsize=9.5)
        zz = np.where(z > 0, z / u, np.nan)
        zv = zz[np.isfinite(zz)]
        axs[i, 1].imshow(zz, cmap="Greys", vmin=np.quantile(zv, 0.02), vmax=np.quantile(zv, 0.98), interpolation="nearest")
        axs[i, 1].set_title(f"{lab}: true depth (full resolution; light = near)", loc="left", fontsize=9.5)
        from scipy.ndimage import maximum_filter
        axs[i, 2].imshow(maximum_filter(pimg, size=2), cmap=cm_pts, vmin=-0.5, vmax=3.5, interpolation="nearest")
        axs[i, 2].set_title(f"observed points (records 0..t) projected here ({pst['projected_pixels']} px, dots 2x2):\n"
                            f"{100 * pst['consistent']:.1f}% on the true surface (gray), {100 * pst['occluded']:.1f}% behind it (blue),\n"
                            f"{100 * pst['inconsistent']:.1f}% in front (red)", loc="left", fontsize=8.5)
        axs[i, 3].imshow(cimg, cmap=cm_p1, vmin=-1.5, vmax=2.5, interpolation="nearest", aspect=MC.H / MC.W * GW / GH)
        extra = ""
        if cst.get("self_seen_p1_known") is not None:
            extra = (f"\nrays THIS camera measured itself: {100 * cst['self_seen_p1_known']:.0f}% found in P1, "
                     f"{100 * cst['self_seen_p1_hole']:.0f}% fall through holes")
        axs[i, 3].set_title(f"planner's map P1 along the 64 x 114 scoring rays:\n{100 * cst['known_surface']:.0f}% hit a P1 known surface (gray), "
                            f"{100 * cst['no_known_surface_unknown']:.0f}% none -> UNKNOWN (blue){extra}", loc="left", fontsize=8.5)
        for a in axs[i]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"{'MP3D' if dataset == 'indoor' else 'Macarons++'} {st['scene']} start {st['start']} t = {st['t']}: "
                 f"are the observed (gray) pixels transformed correctly, and what does the planner's map P1 know?\n"
                 f"row 1 = current camera (took the observation at t); row 2 = {rows[1][0]} (the V2-Mix pick)",
                 x=0.005, ha="left", fontsize=10.5)
    fig.savefig(path, dpi=125)
    plt.close(fig)


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    cores = S2.load_cores(dev)
    cases = json.load(open(os.path.join(HERE, "results", "cases.json")))
    todo = [("outdoor", "success"), ("indoor", "stochastic_nondegenerate"), ("indoor", "success"),
            ("outdoor", "stochastic_nondegenerate")]
    out = {}
    for dataset, case in todo:
        sid = cases[dataset][case]["sid"]
        st = next(s for s in A.load_manifest(dataset)["states"] if s["sid"] == sid)
        src = st["sources"][0]["source"]
        ep = S2.episode_of(dataset, st)
        rz = np.load(os.path.join(A.W, "cases_v2", f"{dataset}_{sid}.npz"))
        j = int(rz["pick_V2M"])
        recs = [np.load(A.svc_path(dataset, src, ep, t)) for t in range(st["t"] + 1)]
        P = np.concatenate([r["sample_points"] for r in recs]).astype(np.float64)
        rec = recs[-1]
        rp = S2.StateReplay(dataset, ep, src, cores, dev)
        for t, rec2, geo, h, eps in rp.run([st["t"]]):
            c_cur, s_cur = p1_view(geo, rec["R"], rec["T"], zbuf=rec["zbuf"])
            c_can, s_can = p1_view(geo, rec["legal_R"][j], rec["legal_T"][j])
        p_cur, ps_cur = splat(P, rec["R"], rec["T"], rz["z_cur"])
        p_can, ps_can = splat(P, rec["legal_R"][j], rec["legal_T"][j], rz[f"z_c{j}"])
        out[f"{dataset}|{case}"] = {"sid": sid, "scene": st["scene"], "t": st["t"], "n_observed_points": int(len(P)),
                                    "candidate": j, "current": {"points": ps_cur, "p1": s_cur},
                                    "candidate_view": {"points": ps_can, "p1": s_can}}
        rows = [("current camera", rz["rgb_cur"], rz["z_cur"], p_cur, ps_cur, c_cur, s_cur),
                (f"candidate j = {j}", rz[f"rgb_c{j}"], rz[f"z_c{j}"], p_can, ps_can, c_can, s_can)]
        figure(dataset, st, rows, os.path.join(HERE, "figures", "v2", f"diag_gray_{dataset}_{case}.png"))
        print(dataset, case, json.dumps(out[f"{dataset}|{case}"]), flush=True)
        torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(HERE, "results", "diag_gray.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
