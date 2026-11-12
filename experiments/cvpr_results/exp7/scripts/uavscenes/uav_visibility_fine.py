"""EXP7 / UAVScenes -- recompute a tile's visibility ratio with a finer deduplication of the observation union (magician env, 1 GPU).
Reason: on uav_AMvalley_01_v1 the official coverage exceeded 1 (max 1.039), i.e. the visibility_ratio written by uav_tile_gpu.py
(union deduplicated per 0.25-unit cell, cell-centre error <= 0.22) under-estimates the visible fraction. Same definition and inputs as
uav_tile_gpu.py step 7 (same seeded 100k GT points on faces fully inside the scene box, same reachable positions by the official segment
rule from open sky, all 50 views, all pixels with z < 70, eps = 1 runtime unit) with the union deduplicated per `--vxs` (default 0.05,
error <= 0.043). settings.json is NOT modified (the closed loops never read the ratio; it only normalises the evaluator's coverage);
the result goes to <tile>/visibility_fine.json and the evaluation rescales coverage by old_ratio / new_ratio (exact: coverage =
covered fraction / visibility_ratio). Usage: python -I uav_visibility_fine.py TILE [--vxs 0.05]"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
ROOT = os.path.join(R, "work_dirs", "cvpr_results", "exp7", "uav_macpp")
sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69")
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task0", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "exp7", "scripts", "uavscenes"))
from uav_tile_gpu import seg_blocked_batch  # noqa: E402


def union_and_gt(tile, vxs):
    """Observation union of all reachable views (cell centres, deduplicated per vxs) + the seeded 100k area-weighted GT points."""
    import torch
    import trimesh
    import mp3d_sim as S
    a = argparse.Namespace(tile=tile, vxs=vxs)
    dev = torch.device("cuda")
    td = os.path.join(ROOT, a.tile)
    st = json.load(open(os.path.join(td, "settings.json")))
    tm = trimesh.load(os.path.join(td, f"{a.tile}.obj"))
    tm = tm.dump(concatenate=True) if isinstance(tm, trimesh.Scene) else tm
    tm.vertices *= 10.0
    V, F = np.asarray(tm.vertices, np.float64), np.asarray(tm.faces, np.int64)
    ray = tm.ray
    cam = st["camera"]
    lo, hi = 10.0 * np.asarray(cam["x_min"]), 10.0 * np.asarray(cam["x_max"])
    n = np.asarray([cam["pose_l"], cam["pose_w"], cam["pose_h"]])
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    X = lo + (0.5 + idx) * (hi - lo) / n
    key = {tuple(r): i for i, r in enumerate(idx)}
    top = np.flatnonzero(idx[:, 1] == n[1] - 1)
    seed_free = ~seg_blocked_batch(ray, X[top], X[top] + np.array([0.0, 1000.0, 0.0]))
    edges = np.asarray([(i, key[tuple(r + d)]) for i, r in enumerate(idx) for d in np.eye(3, dtype=int) if tuple(r + d) in key])
    blocked = seg_blocked_batch(ray, X[edges[:, 0]], X[edges[:, 1]])
    adj = [[] for _ in range(len(idx))]
    for (i, j), b in zip(edges, blocked):
        if not b:
            adj[i].append(j); adj[j].append(i)
    reach = np.zeros(len(idx), bool)
    stack = list(top[seed_free]); reach[stack] = True
    while stack:
        i = stack.pop()
        for j in adj[i]:
            if not reach[j]:
                reach[j] = True; stack.append(j)
    sc = st["scene"]
    smin, smax = 10.0 * np.asarray(sc["x_min"]), 10.0 * np.asarray(sc["x_max"])
    inside = np.all((V[F] >= smin) & (V[F] <= smax), axis=(1, 2))
    Fi, ai = F[inside], tm.area_faces[inside]
    rv = np.random.default_rng(int(hashlib.sha256(f"EXP7-UAV-v0|gtvis|{a.tile}".encode()).hexdigest()[:15], 16))
    gi = rv.choice(len(Fi), 100_000, p=ai / ai.sum())
    u, v = rv.random(len(gi)), rv.random(len(gi))
    su = np.sqrt(u)
    tri = V[Fi[gi]]
    G = (1 - su)[:, None] * tri[:, 0] + (su * (1 - v))[:, None] * tri[:, 1] + (su * v)[:, None] * tri[:, 2]
    sen = S.Sensor(V, F, dev)
    elevs = [-90.0 + 180.0 * (1 + i) / (cam["pose_n_theta"] + 1) for i in range(cam["pose_n_theta"])]
    azims = [360.0 * i / cam["pose_n_azim"] for i in range(cam["pose_n_azim"])]
    VXS = a.vxs
    off = torch.as_tensor(np.floor(V.min(0) / VXS) - 2, dtype=torch.float64, device=dev)
    keys, acc = None, []
    for i in np.flatnonzero(reach):
        for e in elevs:
            for az in azims:
                Rm, Tm = sen.RT(X[i], e, az)
                fr = sen.render(Rm, Tm)
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
    return cen, G, int(reach.sum()), sc["visibility_ratio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tile")
    ap.add_argument("--vxs", type=float, default=0.05)
    a = ap.parse_args()
    from scipy.spatial import cKDTree
    t0 = time.time()
    td = os.path.join(ROOT, a.tile)
    VXS = a.vxs
    cen, G, nreach, vr_settings = union_and_gt(a.tile, VXS)
    kk = cen
    d, _ = cKDTree(cen).query(G, k=1, distance_upper_bound=1.0, workers=8)
    vr = float(np.isfinite(d).mean())
    res = {"tile": a.tile, "vxs": VXS, "reachable": nreach, "union_cells": int(len(kk)), "visibility_ratio_fine": vr,
           "visibility_ratio_settings": vr_settings, "wall_s": time.time() - t0}
    json.dump(res, open(os.path.join(td, "visibility_fine.json"), "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
