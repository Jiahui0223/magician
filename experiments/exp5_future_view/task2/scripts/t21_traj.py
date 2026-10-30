"""EXP5 Task 2 / T2.1a step 2 -- MP3D training / selection trajectories on the T0.12 grid (magician env, GPU; simulator /
GT side, training houses and selection houses only).

Same grid (adapters/grid_env.py), legality (official single segment on the GT mesh), sensor (frozen MAGICIAN sensor,
456 x 256, 60 deg), 10 % sampling seeds and service-record format as the T0.12 simulator (scripts/t12_sim.py, imported
unchanged); 150 moves. Policies (W2/mp3d/manifests/traj_jobs.json):
  rnv  seeded uniform choice among the legal poses not executed before (all legal poses if none)
  ogr  GT one-step greedy: the legal candidate whose next 10 % sample covers the most not-yet-covered GT points
       (100k area-weighted GT points of the level box, eps = EPS_MODEL x k as the C_obs evaluator); unvisited poses
       first; ties seeded. GT here only shapes the training trajectories (training / selection houses).
Start: the job's panorama snapped to the grid (t12_sim.start_pose), elevation 0, the job's seeded azimuth.
Output W2/mp3d/traj/<level>/<traj>/{svc/svc_t<t>.npz, traj.json}.
Usage: python t21_traj.py <worker index> <n workers>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402
import t21_levels as LV  # noqa: E402  (mesh cache in W2)

sys.path.insert(0, os.path.join(C.EXP5, "task0", "magician_comparison", "scripts"))
sys.path.insert(0, os.path.join(C.EXP5, "task0", "magician_comparison", "adapters"))
import t12_sim as T12  # noqa: E402
import grid_env as GE  # noqa: E402

import numpy as np  # noqa: E402

S = T12.S
N_GT = 100000


def gt_points(scan, card, v, f):
    cen = v[f].mean(1)
    box = np.asarray(card["box_m"])
    inb = np.all((cen >= box[0]) & (cen <= box[1]), axis=1)
    tri = v[f[inb]]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    rng = np.random.default_rng(C.seed_of(f"exp5-T2|gtpts|{card['key']}"))
    fi = rng.choice(len(tri), N_GT, p=area / area.sum())
    r1, r2 = rng.random(N_GT), rng.random(N_GT)
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * tri[fi, 0] + (s * (1 - r2))[:, None] * tri[fi, 1] + (s * r2)[:, None] * tri[fi, 2]
    return S.zup_m_to_rt(p)


def near(tree, obs, eps):
    from scipy.spatial import cKDTree
    if len(obs) == 0:
        return np.zeros(tree.n, bool)
    d, _ = cKDTree(obs).query(tree.data, k=1, distance_upper_bound=eps, workers=8)
    return np.isfinite(d)


def run(job, dev):
    import torch
    from scipy.spatial import cKDTree
    od = os.path.join(C.W2, "mp3d", "traj", job["key"], job["traj"])
    if os.path.exists(os.path.join(od, "traj.json")):
        return None
    t0 = time.time()
    C.assert_mesh_allowed(job["scan"], job["purpose"])
    card = C.load_card(job["key"])
    v, f = LV.mesh_zup(job["scan"])
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    blk = T12.Blocker(v, f)
    g = GE.grid_def(card)
    P = np.asarray(card["pano_rt"])[card["pano_idx"].index(job["start_pano"])]
    pose, snap = T12.start_pose(g, S.rt_to_zup_m(P), blk)
    pose = (pose[0], pose[1], pose[2], 2, job["start_azim"])
    tree = eps = covered = None
    if job["policy"] == "ogr":
        tree = cKDTree(gt_points(job["scan"], card, v, f))
        eps = S.EPS_MODEL * card["k_I"]
        covered = np.zeros(tree.n, bool)
    os.makedirs(os.path.join(od, "svc"), exist_ok=True)
    hist, rows = [], []
    for t in range(C.MOVES_INDOOR + 1):
        hist.append(pose)
        e, az = GE.angles(g, pose)
        R, T = sen.RT(GE.position(g, pose), e, az)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(job["key"], job["start_pano"], t, "sample"))
        if covered is not None:
            covered |= near(tree, pts.cpu().numpy(), eps)
        legal = T12.legal_actions(g, pose, blk)
        rec = T12.service_record(job["key"], job["start_pano"], t, fr, pts, flat, pose, hist, legal, g, sen)
        np.savez(os.path.join(od, "svc", f"svc_t{t}.npz"), **rec)
        if t == C.MOVES_INDOOR:
            break
        seen = set(hist)
        cand = [j for j, q in enumerate(legal) if q not in seen] or list(range(len(legal)))
        rng = np.random.default_rng(C.seed_of(f"exp5-T2|{job['key']}|{job['traj']}|{t}"))
        gains = None
        if job["policy"] == "ogr" and cand:
            gains = []
            seed = S.derived_seed(job["key"], job["start_pano"], t + 1, "sample")
            for j in cand:
                q = legal[j]
                e, az = GE.angles(g, q)
                Rq, Tq = sen.RT(GE.position(g, q), e, az)
                pq = sen.sample(sen.render(Rq, Tq), seed)[0]
                gains.append(int((near(tree, pq.cpu().numpy(), eps) & ~covered).sum()))
            best = [j for j, x in zip(cand, gains) if x == max(gains)]
            pick = best[int(rng.integers(len(best)))]
        else:
            pick = cand[int(rng.integers(len(cand)))] if cand else None
        rows.append({"t": t, "n_legal": len(legal), "pick": pick, "max_gain": max(gains) if gains else None,
                     "gt_cov": float(covered.mean()) if covered is not None else None})
        pose = legal[pick] if pick is not None else pose
        torch.cuda.empty_cache() if t % 50 == 0 else None
    out = {**job, "moves": C.MOVES_INDOOR, "grid_n": list(g["n"]), "start_pose": list(hist[0]), "start_snap_m": snap,
           "executed": [list(p) for p in hist], "steps": rows, "seconds": time.time() - t0}
    json.dump(out, open(os.path.join(od, "traj.json.tmp"), "w"))
    os.replace(os.path.join(od, "traj.json.tmp"), os.path.join(od, "traj.json"))
    return out


def main():
    import torch
    wi, wn = int(sys.argv[1]), int(sys.argv[2])
    dev = torch.device("cuda")
    jobs = json.load(open(os.path.join(C.W2, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    for j in jobs[wi::wn]:
        o = run(j, dev)
        if o:
            print(j["key"], j["traj"], "positions", len({tuple(p[:3]) for p in o["executed"]}),
                  "gt_cov", o["steps"][-1]["gt_cov"], "%.0f s" % o["seconds"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
