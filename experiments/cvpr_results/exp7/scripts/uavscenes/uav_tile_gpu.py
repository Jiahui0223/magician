"""EXP7 / UAVScenes -- second stage of a UAV-v0 tile (magician env, 1 GPU; simulator / protocol side, GT allowed here).

Camera pre-selection, starts and visibility of one tile folder written by uav_build_tile.py:
  1. the official MAGICIAN loader (load_scene, scale 10) must load the OBJ; vertex count / bounds are recorded.
  2. camera positions: official pose formula (cell centres of the settings camera box, runtime units).
  3. reachable positions: breadth-first search over +-1 position moves from the top-layer positions whose vertical segment to
     the sky is free; a move is allowed iff the official segment test (magician_utils.line_segment_mesh_intersection on the
     GT mesh, the simulator's legality rule) is free.
  4. clearance = distance to the nearest of 3 M area-weighted surface samples (runtime units).
  5. occupied_pose.pt (Macarons++ format): occupied = not reachable or clearance < 2.5 runtime (5 m) -- informative only
     (MAGICIAN's test loop does not read it; the simulator's legality is the segment test alone).
  6. starts: seeded (sha256 of "EXP7-UAV-v0|starts|<tile>") uniform choice of N distinct reachable positions with clearance
     >= 5 runtime (10 m); elevation index in {0, 1, 2} (-60, -30, 0 deg), azimuth index in 0..9.
  6b. starts must also have open sky above (vertical segment to +1000 runtime free): no start under terrain / overhangs.
  7. visibility_ratio = fraction of 100k area-weighted GT surface points (faces fully inside the scene box, seeded) within
     eps = 1 runtime unit (the coverage eps) of the union of all back-projected depth points (all pixels, z < 70, frozen
     sensor) of the 50 views of every reachable position; the union is deduplicated per 0.25-unit cell (cell centres, error
     <= 0.22 unit) -- the role of the Macarons++ settings value, same proximity rule as the official coverage. (A first version
     queried per-view 10 % samples with KD-trees at ~0.2 s / view; a z-buffer variant was stricter than the coverage rule;
     both replaced before any closed loop.)
Writes settings.json (settings.pending.json + visibility_ratio + start_positions), occupied_pose.pt, tile_gpu_meta.json and
<visuals>/uavscenes/<tile>_cameras.png. Usage: python -I uav_tile_gpu.py TILE [--n-starts 10]"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
ROOT = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")
VIS = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "visuals", "uavscenes")
MCLEAN = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69"
sys.path.insert(0, MCLEAN)
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task0", "scripts"))


def seg_blocked_batch(ray, A, B):
    """Official line_segment_mesh_intersection for many segments at once (same rule: any hit closer than the length)."""
    D = B - A
    L = np.linalg.norm(D, axis=1)
    out = np.zeros(len(A), bool)
    ok = L >= 1e-6
    if not ok.any():
        return out
    loc, iray, _ = ray.intersects_location(ray_origins=A[ok], ray_directions=D[ok] / L[ok, None])
    if len(loc):
        idx = np.flatnonzero(ok)[iray]
        d = np.linalg.norm(loc - A[idx], axis=1)
        hit = d < L[idx]
        out[np.unique(idx[hit])] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tile")
    ap.add_argument("--n-starts", type=int, default=10)
    a = ap.parse_args()
    import torch
    import trimesh
    from scipy.spatial import cKDTree
    from macarons.utility.macarons_utils import load_scene
    from macarons.utility.magician_utils import line_segment_mesh_intersection
    import mp3d_sim as S
    t0 = time.time()
    dev = torch.device("cuda")
    td = os.path.join(ROOT, a.tile)
    st = json.load(open(os.path.join(td, "settings.pending.json")))
    obj = os.path.join(td, f"{a.tile}.obj")
    mesh_p3d = load_scene(obj, 10.0, dev)                                  # official loader
    v_off = mesh_p3d.verts_list()[0].cpu().numpy().astype(np.float64)
    tm = trimesh.load(obj)
    if isinstance(tm, trimesh.Scene):
        tm = tm.dump(concatenate=True)
    tm.vertices *= 10.0
    V, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64)
    ray = tm.ray
    cam = st["camera"]
    lo, hi = 10.0 * np.asarray(cam["x_min"]), 10.0 * np.asarray(cam["x_max"])
    n = np.asarray([cam["pose_l"], cam["pose_w"], cam["pose_h"]])
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    X = lo + (0.5 + idx) * (hi - lo) / n
    key = {tuple(r): i for i, r in enumerate(idx)}
    # 3. reachability
    top = np.flatnonzero(idx[:, 1] == n[1] - 1)
    sky = X[top] + np.array([0.0, 1000.0, 0.0])
    seed_free = ~seg_blocked_batch(ray, X[top], sky)
    edges = []
    for i, r in enumerate(idx):
        for ax in range(3):
            q = r.copy(); q[ax] += 1
            if q[ax] < n[ax]:
                edges.append((i, key[tuple(q)]))
    edges = np.asarray(edges)
    blocked = seg_blocked_batch(ray, X[edges[:, 0]], X[edges[:, 1]])
    # spot-check the batched rule against the official per-segment function
    rng0 = np.random.default_rng(0)
    chk = rng0.choice(len(edges), size=min(300, len(edges)), replace=False)
    off = np.asarray([bool(line_segment_mesh_intersection(X[edges[j, 0]], X[edges[j, 1]], ray)) for j in chk])
    assert np.array_equal(off, blocked[chk]), "batched segment test differs from the official function"
    adj = [[] for _ in range(len(idx))]
    for (i, j), b in zip(edges, blocked):
        if not b:
            adj[i].append(j); adj[j].append(i)
    reach = np.zeros(len(idx), bool)
    stack = list(top[seed_free])
    reach[stack] = True
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not reach[j]:
                reach[j] = True; stack.append(j)
    # 4. clearance
    rs = np.random.default_rng(int(hashlib.sha256(f"EXP7-UAV-v0|surface|{a.tile}".encode()).hexdigest()[:15], 16))
    area = tm.area_faces
    fi = rs.choice(len(F), 3_000_000, p=area / area.sum())
    u, v = rs.random(len(fi)), rs.random(len(fi))
    su = np.sqrt(u)
    tri = V[F[fi]]
    P = (1 - su)[:, None] * tri[:, 0] + (su * (1 - v))[:, None] * tri[:, 1] + (su * v)[:, None] * tri[:, 2]
    clear, _ = cKDTree(P).query(X, k=1, workers=-1)
    # 5. occupied poses
    occupied = ~(reach & (clear >= 2.5))
    torch.save({"X_idx": torch.as_tensor(idx, dtype=torch.int64), "occupied": torch.as_tensor(occupied)}, os.path.join(td, "occupied_pose.pt"))
    # 6. starts
    sky_free = ~seg_blocked_batch(ray, X, X + np.array([0.0, 1000.0, 0.0]))    # open air above (not under terrain / overhang)
    cand = np.flatnonzero(reach & (clear >= 5.0) & sky_free)
    rg = np.random.default_rng(int(hashlib.sha256(f"EXP7-UAV-v0|starts|{a.tile}".encode()).hexdigest()[:15], 16))
    pick = rg.choice(cand, size=min(a.n_starts, len(cand)), replace=False)
    starts = [[int(idx[i][0]), int(idx[i][1]), int(idx[i][2]), int(rg.integers(0, 3)), int(rg.integers(0, 10))] for i in pick]
    # 7. visibility
    sc = st["scene"]
    smin, smax = 10.0 * np.asarray(sc["x_min"]), 10.0 * np.asarray(sc["x_max"])
    inside = np.all((V[F] >= smin) & (V[F] <= smax), axis=(1, 2))
    Fi = F[inside]
    ai = area[inside]
    rv = np.random.default_rng(int(hashlib.sha256(f"EXP7-UAV-v0|gtvis|{a.tile}".encode()).hexdigest()[:15], 16))
    gi = rv.choice(len(Fi), 100_000, p=ai / ai.sum())
    u, v = rv.random(len(gi)), rv.random(len(gi))
    su = np.sqrt(u)
    tri = V[Fi[gi]]
    G = (1 - su)[:, None] * tri[:, 0] + (su * (1 - v))[:, None] * tri[:, 1] + (su * v)[:, None] * tri[:, 2]
    sen = S.Sensor(V, F, dev)
    elevs = [-90.0 + 180.0 * (1 + i) / (cam["pose_n_theta"] + 1) for i in range(cam["pose_n_theta"])]
    azims = [360.0 * i / cam["pose_n_azim"] for i in range(cam["pose_n_azim"])]
    VXS = 0.25                                                             # union of observed points, deduplicated per 0.25 cell
    off = torch.as_tensor(np.floor(V.min(0) / VXS) - 2, dtype=torch.float64, device=dev)
    keys, acc, n_views = None, [], 0
    for i in np.flatnonzero(reach):
        for e in elevs:
            for az in azims:
                Rm, Tm = sen.RT(X[i], e, az)
                fr = sen.render(Rm, Tm)
                n_views += 1
                depth = torch.clamp(fr["zbuf"], min=0.5, max=750.0)
                pm = (fr["mask"].view(-1) & (depth.view(-1) < S.RANGE))
                if not pm.any():
                    continue
                p = sen.unproject(depth, fr["fov"])[0][pm].double()
                c = (torch.floor(p / VXS) - off).long()
                acc.append((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])
        if len(acc) >= 200:
            keys = torch.unique(torch.cat(([keys] if keys is not None else []) + acc)); acc = []
    if acc:
        keys = torch.unique(torch.cat(([keys] if keys is not None else []) + acc))
    kk = keys.cpu().numpy()
    cen = (np.stack([kk >> 42, (kk >> 21) & ((1 << 21) - 1), kk & ((1 << 21) - 1)], 1) + off.cpu().numpy() + 0.5) * VXS
    d, _ = cKDTree(cen).query(G, k=1, distance_upper_bound=1.0, workers=8)
    vis = np.isfinite(d)
    vr = float(vis.mean())
    st["scene"]["visibility_ratio"] = vr
    st["camera"]["start_positions"] = starts
    json.dump(st, open(os.path.join(td, "settings.json"), "w"))
    meta = {"official_loader_vertices": int(len(v_off)), "official_loader_bbox": [v_off.min(0).tolist(), v_off.max(0).tolist()],
            "trimesh_vertices": int(len(V)), "trimesh_faces": int(len(F)), "positions": int(len(idx)), "top_seed_free": int(seed_free.sum()),
            "edges": int(len(edges)), "edges_blocked": int(blocked.sum()), "segment_rule_spotcheck": int(len(chk)),
            "reachable": int(reach.sum()), "reachable_sky_free": int((reach & sky_free).sum()), "reachable_clear_2p5": int((reach & (clear >= 2.5)).sum()), "start_candidates_clear_5": int(len(cand)),
            "starts": starts, "start_clearance": [float(clear[i]) for i in pick], "gt_points_vis": int(len(G)), "views_rendered": n_views,
            "visibility_ratio": vr, "wall_s": time.time() - t0}
    json.dump(meta, open(os.path.join(td, "tile_gpu_meta.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    sel = np.random.default_rng(1).choice(len(P), 200000, replace=False)
    for ax, (u_, v_, lab) in zip(axs, ((0, 2, "x (east) / z (-north)"), (0, 1, "x (east) / y (up)"))):
        ax.scatter(P[sel, u_], P[sel, v_], s=0.1, c=P[sel, 1], cmap="terrain")
        ax.scatter(X[~reach, u_], X[~reach, v_], s=4, c="#999999", label="not reachable")
        ax.scatter(X[reach, u_], X[reach, v_], s=4, c="#0072B2", label="reachable")
        sx = X[[key[tuple(s_[:3])] for s_ in starts]]
        ax.scatter(sx[:, u_], sx[:, v_], s=40, c="#D55E00", marker="*", label="starts")
        ax.set_title(f"{a.tile}: {lab}"); ax.set_aspect("equal")
    axs[0].legend(fontsize=7, loc="upper right")
    os.makedirs(VIS, exist_ok=True)
    fig.savefig(os.path.join(VIS, f"{a.tile}_cameras.png"), dpi=110, bbox_inches="tight")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
