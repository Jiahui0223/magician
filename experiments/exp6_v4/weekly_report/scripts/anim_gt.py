"""GT surface points of an episode and the step at which each point becomes covered (evaluation side only; never used by
the planner). Outdoor (magician env, 1 GPU): the frozen task2/scripts/t25_out_cov.py replay (EpisodeSim, same GT
points, eps, visibility normalisation); the monotone coverage curve is asserted equal to the stored metrics file.
Indoor (any env, CPU): the cached level GT points / visibility (task0 preprocessing/coverage/<level>.npz) and the
episode's own 10 % sample points; the C_obs curve is asserted equal to episode.json.
Output: pts [N, 3] float32 (runtime frame), first [N] int16 (first covered step, -1 = never), counted [N] bool
(points in the coverage denominator), curve.
Usage: python anim_gt.py --domain outdoor --scene S --start K --out F.npz | --domain indoor --key SCAN_Lx --start K --out F.npz"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
EXP5 = os.path.join(R, "experiments", "exp5_future_view")
W4 = os.path.join(R, "work_dirs", "exp6_v4", "eval")


def main():
    from scipy.spatial import cKDTree
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["outdoor", "indoor"], required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--key", default=None)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.domain == "outdoor":
        sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
        import t2_common as C  # noqa: F401
        sys.path.insert(0, os.path.join(EXP5, "task1", "scripts"))
        import v2_common as LC
        sys.path.insert(1, LC.EXP3B)
        import m_t6_sim as T6S
        torch = T6S.torch
        s, k = a.scene, a.start
        tag = f"V2M_{s}_s{k}_r0"
        ep = json.load(open(os.path.join(W4, "closed_loop", "outdoor", "V4", "V2M", "episodes", f"ep_{tag}.json")))
        met = json.load(open(os.path.join(W4, "metrics", "outdoor", "V4", "V2M", f"{tag}.json")))
        keys = ep["executed_keys"]
        fr = os.path.join(os.path.dirname(a.out), f"_frames_{tag}")
        shutil.rmtree(fr, ignore_errors=True)
        sim = T6S.EpisodeSim(s, k, fr)
        gt, eps, vis = sim.gt_scene, sim.eps, sim.vis
        cells = [c for c in gt.cells if len(gt.cells[c].cell_pts) > 0]
        G = torch.cat([gt.cells[c].cell_pts for c in cells]).double().cpu().numpy()
        tree = cKDTree(G)
        first = np.full(len(G), -1, np.int16)
        mono = []
        for t in range(len(keys)):
            assert sim.key() == keys[t], (t, sim.key(), keys[t])
            pts = sim.observe(t)["part_pc"].double().cpu().numpy()
            if len(pts):
                for idx in tree.query_ball_point(pts, r=eps * (1 - 1e-12), workers=16):
                    idx = np.asarray(idx, int)
                    first[idx[first[idx] < 0]] = t
            mono.append(float((first >= 0).sum()) / len(G) / vis)
            if t + 1 < len(keys):
                sim.move(keys[t + 1])
        shutil.rmtree(fr, ignore_errors=True)
        assert mono == met["cov_mono"], "monotone coverage replay differs from the stored metrics"
        np.savez_compressed(a.out, pts=G.astype(np.float32), first=first, counted=np.ones(len(G), bool), curve=np.asarray(mono), eps=eps, vis=vis)
    else:
        key, k = a.key, a.start
        root = os.path.join(W4, "closed_loop", "indoor", "V4", "V2M", f"{key}_S{k}")
        ep = json.load(open(os.path.join(root, "episode.json")))
        z = np.load(os.path.join(R, "work_dirs", "exp5_future_view", "task0", "preprocessing", "coverage", f"{key}.npz"))
        P, vis, eps = z["pts"].astype(np.float64), z["visible"], float(z["eps_rt"])
        tree = cKDTree(P)
        first = np.full(len(P), -1, np.int16)
        curve = []
        for t in range(len(ep["coverage"])):
            obs = np.asarray(np.load(os.path.join(root, "svc", f"svc_t{t}.npz"))["sample_points"], np.float64)
            if len(obs):
                d, _ = cKDTree(obs).query(P, k=1, distance_upper_bound=eps, workers=-1)
                new = np.isfinite(d) & (first < 0)
                first[new] = t
            curve.append(float(((first >= 0) & vis).sum() / max(1, vis.sum())))
        assert np.allclose(curve, ep["coverage"], rtol=0, atol=1e-12), "C_obs replay differs from episode.json"
        np.savez_compressed(a.out, pts=P.astype(np.float32), first=first, counted=vis, curve=np.asarray(curve), eps=eps)
    print("saved", a.out)


if __name__ == "__main__":
    main()
