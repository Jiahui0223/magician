"""EXP Density A/B -- dense-coverage sensitivity check, Macarons++ (magician env, 1 GPU; user request 2026-11-03).

Question: does the coverage metric (MAGICIAN protocol: a GT point is covered only when one of the simulator's 10 % sampled
points lies within eps) penalise D100 scoring? Each existing G3 episode (12 configs x 30) is replayed with the frozen
M-protocol EpisodeSim exactly as task2/scripts/t25_out_cov.py (same executed keys, seeded observations, GT points, eps,
visibility normalisation) and, per step t = 0..100, three monotone coverage curves are computed on the SAME frames:
  cov_official   MAGICIAN official (capped cells); must equal the stored episode curve bitwise (else STOP)
  cov_mono       the Density A/B primary metric recomputed (all 10 % samples kept); must equal the stored metrics file
  cov_dense      the same GT points / eps / normalisation, but EVERY valid pixel (mask & depth < sensor range) of every
                 observed frame is back-projected (camera.compute_partial_point_cloud with gathering_factor = 1.0, i.e.
                 the official back-projection without sub-sampling)
The planner trajectories are not changed (no re-planning): this measures the same trajectories with a dense yardstick.
Output work_dirs/exp_density_ab/dense_cov/<cfg>/V2M_<scene>_s<k>_r0.json
Usage: python da_dense_cov.py --cfg CFG --scene S --start K
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import v2_common as LC  # noqa: E402

sys.path.insert(1, LC.EXP3B)
import m_t6_sim as T6S  # noqa: E402

np, torch = T6S.np, T6S.torch
W3 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
NEW = os.path.join(REPO, "work_dirs", "exp_density_ab", "g3")
OUT = os.path.join(REPO, "work_dirs", "exp_density_ab", "dense_cov")
HIST = {"B2_S10_S10": "B0", "B2_D100_S10": "B2", "B2_D100_D100": "B1", "E2S_S10_S10": "E2S", "E2D_D100_S10": "E2D"}


def roots(cfg):
    if cfg in HIST:
        g = HIST[cfg]
        return os.path.join(W3, "closed_loop", "outdoor", g, "V2M"), os.path.join(W3, "metrics", "outdoor", g, "V2M")
    return os.path.join(NEW, "closed_loop", "outdoor", cfg, "V2M"), os.path.join(NEW, "metrics", "outdoor", cfg, "V2M")


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    a = ap.parse_args()
    tag = f"V2M_{a.scene}_s{a.start}_r0"
    out = os.path.join(OUT, a.cfg, f"{tag}.json")
    if os.path.exists(out):
        return
    t0 = time.time()
    root, mroot = roots(a.cfg)
    ep = json.load(open(os.path.join(root, "episodes", f"ep_{tag}.json")))
    ref = json.load(open(os.path.join(mroot, f"{tag}.json")))
    keys = ep["executed_keys"]
    fr = os.path.join(OUT, "tmp", f"{a.cfg}_{tag}")
    shutil.rmtree(fr, ignore_errors=True)
    sim = T6S.EpisodeSim(a.scene, a.start, fr)
    gt, eps, vis = sim.gt_scene, sim.eps, sim.vis
    cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
    G = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
    tree = cKDTree(G)
    r = eps * (1 - 1e-12)
    cov_s, cov_d = np.zeros(len(G), bool), np.zeros(len(G), bool)
    off, mono, dense, n_s, n_d = [], [], [], [], []
    for t in range(len(keys)):
        assert sim.key() == keys[t], (t, sim.key(), keys[t])
        ob = sim.observe(t)
        off.append(ob["cov"])
        ps = ob["part_pc"].double().cpu().numpy()
        if len(ps):
            for idx in tree.query_ball_point(ps, r=r, workers=8):
                cov_s[idx] = True
        frame = ob["frame"]
        depth, mask, error_mask, R, T = T6S.MP.apply_perfect_depth_simple(frame, sim.dev)
        m = (mask * error_mask).bool()
        with torch.no_grad():
            pd, _ = sim.camera.compute_partial_point_cloud(depth=depth, mask=m, images=frame["rgb"], fov_cameras=ob["fov_camera"],
                                                           gathering_factor=1.0, fov_range=sim.params.sensor_range)
        pdn = pd.double().cpu().numpy()
        if len(pdn):
            unc = np.flatnonzero(~cov_d)
            if len(unc):
                d, _ = cKDTree(pdn).query(G[unc], k=1, distance_upper_bound=r, workers=8)
                cov_d[unc[np.isfinite(d) & (d <= r)]] = True
        cov_d |= cov_s                                   # the 10 % samples are a subset of the dense points
        mono.append(float(cov_s.sum()) / len(G) / vis)
        dense.append(float(cov_d.sum()) / len(G) / vis)
        n_s.append(int(len(ps)))
        n_d.append(int(len(pdn)))
        if t + 1 < len(keys):
            sim.move(keys[t + 1])
    shutil.rmtree(fr, ignore_errors=True)
    if off != ep["coverage"]:
        raise SystemExit(f"STOP: official coverage replay differs {a.cfg} {tag}")
    mono_eq = bool(np.allclose(mono, ref["cov_mono"], rtol=0, atol=1e-12))
    if not mono_eq:
        raise SystemExit(f"STOP: recomputed 10 % monotone coverage differs from the metrics file {a.cfg} {tag}")
    moves = ep["moves"]
    cm, cd = np.asarray(mono), np.asarray(dense)
    res = {"cfg": a.cfg, "tag": tag, "scene": a.scene, "start": a.start, "moves": moves, "n_gt": int(len(G)), "eps": float(eps), "vis": float(vis),
           "official_equal_stored": True, "mono_equal_metrics": mono_eq,
           "cov_official": off, "cov_mono": mono, "cov_dense": dense,
           "auc_mono": float(np.trapz(cm, dx=1.0) / moves), "final_mono": float(cm[-1]),
           "auc_dense": float(np.trapz(cd, dx=1.0) / moves), "final_dense": float(cd[-1]),
           "n_points_sample_mean": float(np.mean(n_s)), "n_points_dense_mean": float(np.mean(n_d)),
           "dense_ge_mono_all_steps": bool(np.all(cd >= cm - 1e-15)), "seconds": time.time() - t0}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out + ".tmp", "w"), indent=1)
    os.replace(out + ".tmp", out)
    print(a.cfg, tag, "mono AUC %.4f final %.4f | dense AUC %.4f final %.4f | pts %.0f / %.0f | %.0f s" % (
        res["auc_mono"], res["final_mono"], res["auc_dense"], res["final_dense"], res["n_points_sample_mean"], res["n_points_dense_mean"], res["seconds"]), flush=True)


if __name__ == "__main__":
    main()
