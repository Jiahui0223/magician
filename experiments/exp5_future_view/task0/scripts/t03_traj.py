"""EXP5 Task 0.3 -- indoor reference trajectories (magician env, simulator / GT side; protocol M).

For one level (scan, level) and a start panorama: seeded reference policy for `steps` moves (uniform among the
legal actions whose target pose is unvisited, else among all legal actions; seed exp5-T0|<key>|<start>|t|ref),
whitelisted service record per step (frozen format), level-scale C_obs per step. Outputs W/preprocessing/traj/<key>_S<start>/
{svc_t<t>.npz, traj.json}. Re-running with the same arguments reproduces the files bitwise (checked by --verify).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402


def run(scan, level, start, steps, out_root, verify=False):
    import torch
    dev = torch.device("cuda")
    t0 = time.time()
    L = S.Level(scan, level)                         # whole level: graph, poses and the C_obs universe
    L.build_graph()
    v, f = S.mesh_zup(scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    S.write_scene_card(L)
    s_local = next(i for i, p in enumerate(L.panos) if p["idx"] == int(start))
    pose = (s_local, 0, 0, 1, 0)                     # start: elevation 0, azimuth bin 0
    od = os.path.join(out_root, f"{L.key}_S{int(start)}")
    os.makedirs(od, exist_ok=True)
    hist, visited, rows = [], {pose}, []
    for t in range(steps + 1):
        hist.append(pose)
        R, T = sen.RT(L.pano_rt[pose[0]], S.ELEVS[pose[3]], 360.0 * pose[4] / S.N_AZIM)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(L.key, start, t, "sample"))
        c = cov.update(pts.cpu().numpy())
        legal = S.legal_actions(L, pose)
        rec = S.service_record(L.key, start, t, fr, pts, flat, pose, hist, legal, L, sen)
        p = os.path.join(od, f"svc_t{t}.npz")
        if verify:
            old = np.load(p)
            assert all(np.array_equal(old[k], rec[k]) for k in rec), ("not reproducible", t)
        else:
            np.savez(p, **rec)
        rows.append({"t": t, "pose": list(pose), "cov": c, "n_legal": len(legal), "n_points": int(len(pts))})
        if t == steps:
            break
        rng = np.random.default_rng(S.derived_seed(L.key, start, t, "ref") % (2 ** 63))
        fresh = [a for a in legal if a not in visited]
        pool = fresh if fresh else legal
        pose = pool[int(rng.integers(len(pool)))]
        visited.add(pose)
    out = {"scene": L.describe(), "start_pano": int(start), "start_local": s_local, "steps": steps,
           "n_edges": len(L.edges), "n_poses": len(S.all_poses(L)), "visible_points": int(cov.visible.sum()),
           "gt_points": int(len(cov.pts)), "eps_rt": cov.eps_rt, "rows": rows, "seconds": time.time() - t0}
    if not verify:
        json.dump(out, open(os.path.join(od, "traj.json"), "w"), indent=1, default=float)
    print(L.key, start, "cov", [round(r["cov"], 3) for r in rows[:: max(1, steps // 10)]], f"{out['seconds']:.0f}s", flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True, help="MP3D panorama index")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(S.W, "preprocessing", "traj"))
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    run(a.scan, a.level, a.start, a.steps, a.out, a.verify)
