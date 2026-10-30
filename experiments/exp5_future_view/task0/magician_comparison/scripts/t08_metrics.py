"""EXP5 Task 0 / T0.8 -- unified metrics of the frozen T0.6 trajectories (magician_comparison protocol v1.0 §2-§4).
Evaluation side only (reads the GT mesh); no model inference; read-only on T0.1-T0.7 outputs. Magician env.

  visible  : per level, GT vertices (face-referenced, inside box_m) within 5 cm of the full-depth back-projection of
             ANY legal pose (all panoramas x 3 elev x 8 azim; frozen sensor on the whole-house mesh, as t06_sim)
             -> MC/metric_replay/visible/<level>.npz
  episodes : per (group, episode): coverage reproduction check (frozen Coverage on the stored 10 % samples vs the
             stored curve, |d| <= 1e-12, else STOP); per frame t = 0..100 the NN distance of every GT vertex to the
             observed cloud so far (primary: all valid pixels; secondary: the 10 % samples) -> Comp@5cm, Comp distance
             (mean / median, cm), Comp@5cm over visible vertices; behaviour metrics -> MC/metric_replay/episodes/<g>/<tag>.json
Usage: python t08_metrics.py visible|episodes --part i/n
"""
import argparse
import json
import os
import sys
import time
import types

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as M  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "adapters"))
import grid_env as GE  # noqa: E402

S = M.S
TAU_M = 0.05
OUT = os.path.join(M.MC, "metric_replay")


def visible(scan, lv, bp, dev):
    import torch
    key = f"{scan}_L{lv}"
    op = os.path.join(OUT, "visible", f"{key}.npz")
    if os.path.exists(op):
        return
    crd = M.card(scan, lv)
    verts = M.gt_vertices(scan, crd)
    v, f = S.mesh_zup(scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    vis = np.zeros(len(verts), bool)
    t0 = time.time()
    for i, X in enumerate(crd["pano_rt"]):
        for e in crd["elevs"]:
            for a in range(crd["n_azim"]):
                R, T = sen.RT(X, e, 360.0 * a / crd["n_azim"])
                fr = sen.render(R, T)
                rec = {"R": R[0].cpu().numpy(), "T": T[0].cpu().numpy(), "zbuf": fr["zbuf"][0, ..., 0].cpu().numpy(),
                       "mask": fr["mask"][0, ..., 0].cpu().numpy()}
                allp, m = bp.points(rec)
                if m.any():
                    d, _ = cKDTree(S.rt_to_zup_m(allp[m])).query(verts, k=1, distance_upper_bound=TAU_M, workers=8)
                    vis |= np.isfinite(d)
    os.makedirs(os.path.dirname(op), exist_ok=True)
    np.savez(op, visible=vis, n_vertices=len(verts), n_poses=len(crd["pano_rt"]) * len(crd["elevs"]) * crd["n_azim"],
             seconds=time.time() - t0)
    print(key, "visible", int(vis.sum()), "/", len(verts), "%.0f s" % (time.time() - t0), flush=True)


def longest_run(x):
    best = cur = 0
    for b in x:
        cur = cur + 1 if b else 0
        best = max(best, cur)
    return best


def episode(group, tag, scan, lv, bp):
    op = os.path.join(OUT, "episodes", group, f"{tag}.json")
    if os.path.exists(op):
        return
    t0 = time.time()
    crd = M.card(scan, lv)
    st = M.stored_curve(group, tag)
    cov = S.Coverage(types.SimpleNamespace(key=crd["key"], k=crd["k_I"]), None)        # cached GT points / visibility
    verts = M.gt_vertices(scan, crd)
    vis = np.load(os.path.join(OUT, "visible", f"{crd['key']}.npz"))["visible"]
    assert len(vis) == len(verts)
    dmin = np.full(len(verts), np.inf)
    dmin_s = np.full(len(verts), np.inf)
    rows, recov = [], []
    NF = len(st["coverage"])                                   # 101 (T0.6 / T0.10) or 151 (T0.12 grid)
    for t in range(NF):
        rec = np.load(M.svc_path(group, tag, t))
        assert list(rec["pose_idx"]) == list(st["executed"][t]), (group, tag, t)
        recov.append(cov.update(rec["sample_points"]))
        allp, m = bp.points(rec)
        if m.any():
            d, _ = cKDTree(S.rt_to_zup_m(allp[m])).query(verts, k=1, workers=8)
            dmin = np.minimum(dmin, d)
        if len(rec["sample_points"]):
            d, _ = cKDTree(S.rt_to_zup_m(np.asarray(rec["sample_points"], np.float64))).query(verts, k=1, workers=8)
            dmin_s = np.minimum(dmin_s, d)
        fin = np.isfinite(dmin)
        rows.append({"t": t, "n_valid_px": int(m.sum()),
                     "comp5": float((dmin <= TAU_M).mean()), "comp5_vis": float((dmin[vis] <= TAU_M).mean()) if vis.any() else None,
                     "dist_mean_cm": float(dmin.mean() * 100) if fin.all() else None,
                     "dist_median_cm": float(np.median(dmin) * 100) if fin.all() else None,
                     "comp5_s": float((dmin_s <= TAU_M).mean()),
                     "dist_mean_s_cm": float(dmin_s.mean() * 100) if np.isfinite(dmin_s).all() else None})
    diff = float(np.abs(np.asarray(recov) - np.asarray(st["coverage"])).max())
    if diff > 1e-12:
        raise SystemExit(f"STOP: coverage reproduction failed for {group} {tag}: max |d| = {diff}")
    # behaviour. Station graph: executed pose = [pano, 0, 0, elev, azim]; grid (T0.12): [l, w, h, elev, azim] -> the
    # "station" is the grid position, its region the .house region of the nearest panorama (evaluation side)
    ex = [tuple(x) for x in st["executed"]]
    c = np.asarray(st["coverage"])
    reg = M.regions(scan, crd)
    if group in M.GRID_GROUPS:
        g = GE.grid_def(crd)
        Pp = np.asarray(crd["pano_rt"], np.float64)
        posid = {p: i for i, p in enumerate(sorted({x[:3] for x in ex}))}
        P = np.asarray([GE.position(g, p) for p in sorted(posid)]) / S.S_RT
        near = [int(np.argmin(np.linalg.norm(Pp - GE.position(g, p), axis=1))) for p in sorted(posid)]
        reg = [reg[j] for j in near]
        ex = [(posid[x[:3]],) + x[3:] for x in ex]
    else:
        P = np.asarray(crd["pano_rt"], np.float64) / S.S_RT
    moves = [(a[0], b[0]) for a, b in zip(ex[:-1], ex[1:]) if a[0] != b[0]]
    seen_st, st_rev = {ex[0][0]}, 0
    for a, b in moves:
        st_rev += b in seen_st
        seen_st.add(b)
    pose_rev = [ex[t + 1] in ex[:t + 1] for t in range(NF - 1)]
    gain = np.diff(c)
    zero = gain <= 1e-12
    rseq = [reg[x[0]] for x in ex]
    out = {"group": group, "tag": tag, "house": scan, "level": lv, "coverage_reproduction_max_abs_diff": diff,
           "cov": {f"cov{k}": float(c[k]) for k in (10, 20, 50, 100, 150) if k < NF}, "auc": float(np.trapz(c[:101], dx=1.0) / 100),
           "auc150": float(np.trapz(c, dx=1.0) / 150) if NF == 151 else None,
           "n_valid_frames": int(sum(r["n_valid_px"] > 0 for r in rows)),
           "mean_valid_px": float(np.mean([r["n_valid_px"] for r in rows])),
           "n_moves": len(moves), "distinct_stations": len({x[0] for x in ex}), "station_revisits": st_rev,
           "station_revisit_ratio": st_rev / len(moves) if moves else None,
           "distinct_poses": len(set(ex)), "pose_revisit_steps": int(sum(pose_rev)), "pose_revisit_ratio": float(np.mean(pose_rev)),
           "path_m": float(sum(np.linalg.norm(P[a] - P[b]) for a, b in moves)),
           "zero_gain_steps": int(zero.sum()), "longest_zero_run": longest_run(zero),
           "gain_on_revisit_steps": float(gain[np.asarray(pose_rev)].sum()), "gain_on_new_steps": float(gain[~np.asarray(pose_rev)].sum()),
           "distinct_regions": len(set(rseq)), "region_changes": int(sum(a != b for a, b in zip(rseq[:-1], rseq[1:]))),
           "decision_s_mean": float(np.mean(st["decision_s"])) if st["decision_s"] else None, "wall_s": st["wall_s"],
           "n_gt_vertices": int(len(verts)), "n_visible_vertices": int(vis.sum()), "curve": rows, "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(op), exist_ok=True)
    json.dump(out, open(op + ".tmp", "w"), indent=1)
    os.replace(op + ".tmp", op)
    print(group, tag, "comp5 %.3f dist %.1f cm cov %.3f (%.0f s)" % (rows[-1]["comp5"], rows[-1]["dist_mean_cm"] or -1, c[100], time.time() - t0), flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["visible", "episodes"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--groups", default=",".join(M.GROUPS), help="episodes mode: groups to process (T0.10: MAG-G,MAG-B,V2K4-NV)")
    ap.add_argument("--grid", action="store_true", help="episodes mode: the 10 T0.12 grid episodes instead of the 22 T0.6 ones")
    a = ap.parse_args()
    pi, pn = (int(x) for x in a.part.split("/"))
    dev = torch.device("cuda")
    bp = M.BackProjector()
    eps = M.grid_episodes() if a.grid else M.episodes()
    if a.mode == "visible":
        levels = sorted({(scan, lv) for _, scan, lv, _ in eps})
        for scan, lv in levels[pi::pn]:
            visible(scan, lv, bp, dev)
    else:
        jobs = [(g, tag, scan, lv) for tag, scan, lv, st in eps for g in a.groups.split(",")]
        for g, tag, scan, lv in jobs[pi::pn]:
            episode(g, tag, scan, lv, bp)


if __name__ == "__main__":
    main()
