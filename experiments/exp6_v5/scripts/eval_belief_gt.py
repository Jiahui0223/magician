"""EXP6 V5 -- evaluation-side multi-surface GT of the dumped belief rays (magician env, 1 GPU). NEVER used for training,
checkpoint choice or planning: it only scores the V5 surface belief after the fixed-state evaluation.

Input  work_dirs/exp6_v5/eval_fixed/<set>_<tag>/<stratum>/belief_<episode>.npz (eval_g1_g2.py: runtime camera R, T,
       k_M, ray indices of the planner's 64 x 114 ray grid, model-frame range [0, min(tcap, grid-box exit)])
Mesh   Macarons++ Dev-3: the scene OBJ x 10 (runtime units), geometry role checked with
       t0_common.assert_geometry_allowed(scene, "dev_eval"); MP3D VALSEL: mp3d_sim.mesh_zup(scan) -> runtime frame
       (zup_m_to_rt, 1 m = 7 runtime units). Same label camera / rasteriser as the training labels
       (build_multisurface_labels.Peeler: faces_per_pixel 16, both faces, z clip 0.5, jitter 0 = the planner rays).
Output gt_<episode>.npz next to the input: surf [n, 8] model units (sorted, inf = none; duplicates within 0.5 VS
       merged), n_surf, t_lo (near clip), t_hi (min(tcap, box exit, t_16 if all 16 layers used, half-way to a 9th surface)).
Usage: python eval_belief_gt.py --dir DIR"""
import argparse
import glob
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_multisurface_labels as B  # noqa: E402

L11, MC, RY, C0, np, torch = B.L11, B.MC, B.RY, B.C0, B.np, B.torch
K, K_STORE, TOL = 16, 8, 0.5 * B.VS


def mesh_of(scene, dev):
    from pytorch3d.structures import Meshes
    if "_L" in scene:                                     # MP3D level "<scan>_L<level>"
        sys.path.insert(0, os.path.join(B.R, "experiments", "exp5_future_view", "task0", "scripts"))
        import mp3d_sim as S
        scan = scene.rsplit("_L", 1)[0]
        v, f = S.mesh_zup(scan)
        vr = S.zup_m_to_rt(v)
        return Meshes(verts=[torch.as_tensor(vr, dtype=torch.float32, device=dev)], faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)]), f"mp3d:{scan}"
    import exp3b_paths as P
    from pytorch3d.io import load_obj
    C0.assert_geometry_allowed(scene, "dev_eval")
    objs = sorted(glob.glob(os.path.join(P.scene_dir(scene), "*.obj")))
    assert len(objs) == 1, objs
    verts, faces, _ = load_obj(objs[0], load_textures=False, device=dev)
    return Meshes(verts=[verts * 10.0], faces=[faces.verts_idx]), objs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda")
    files = sorted(glob.glob(os.path.join(a.dir, "belief_*.npz")))
    cache = {}
    for fp in files:
        po = fp.replace("belief_", "gt_")
        if os.path.exists(po):
            continue
        z = np.load(fp)
        scene = str(z["scene"][0])
        if scene not in cache:
            cache.clear()
            mesh, src = mesh_of(scene, dev)
            cache[scene] = (B.Peeler(mesh, dev, K, batch=4), src)
        peel, src = cache[scene]
        n = len(z["sid"])
        Rs, Ts = list(z["R"]), list(z["T"])
        surf = np.full((n, z["idx"].shape[1], K_STORE), np.inf, np.float32)
        lo_all = np.zeros((n, z["idx"].shape[1]), np.float32)
        hi_all = np.zeros_like(lo_all)
        nsurf = np.zeros(lo_all.shape, np.int16)
        sat = np.zeros(lo_all.shape, bool)
        for s in range(0, n, 16):
            sl = slice(s, min(n, s + 16))
            kM = z["k_M"][sl]
            assert np.allclose(kM, kM[0])
            t = peel.layers(Rs[sl], Ts[sl], [(0.0, 0.0)] * len(Rs[sl]), float(kM[0]))      # [b, 7296, K]
            for b in range(t.shape[0]):
                i = s + b
                idx = z["idx"][i].astype(np.int64)
                tt = t[b][idx]
                u, v = RY.grid_uv((0.0, 0.0))
                _, dz, _ = MC.pixel_rays(Rs[i], Ts[i], u[idx], v[idx])
                tm, cnt = B.merge_dups(tt, TOL)
                st = np.isfinite(tt[:, K - 1])
                hi = np.minimum(z["tcap"][i].astype(np.float64), z["tbox"][i].astype(np.float64))
                hi = np.where(st, np.minimum(hi, tt[:, K - 1]), hi)
                lo = MC.NEAR_CLIP / dz / float(kM[0])
                inr = (tm >= lo[:, None]) & (tm <= hi[:, None])
                ts = np.where(inr, tm, np.inf)
                ts.sort(1)
                over = inr.sum(1) > K_STORE
                hi = np.where(over, 0.5 * (ts[:, K_STORE - 1] + np.where(over, ts[:, min(K_STORE, K - 1)], 0.0)), hi)
                ts = np.where(ts <= hi[:, None], ts, np.inf)[:, :K_STORE]
                surf[i], lo_all[i], hi_all[i], nsurf[i], sat[i] = ts, lo, hi, np.isfinite(ts).sum(1), st
        np.savez_compressed(po, surf=surf, n_surf=nsurf, t_lo=lo_all, t_hi=hi_all, saturated=sat, mesh=src,
                            mesh_sha256=hashlib.sha256(open(src, "rb").read()).hexdigest() if os.path.exists(src) else src)
        print(os.path.basename(po), n, "views, >=2 surfaces %.3f" % float((nsurf >= 2).mean()), flush=True)


if __name__ == "__main__":
    main()
