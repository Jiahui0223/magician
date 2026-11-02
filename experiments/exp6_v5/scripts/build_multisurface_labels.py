"""EXP6 V5 -- multi-surface training labels from the Macarons++ Train-6 meshes (magician env, 1 GPU; label side only,
never imported by any model / planner code).

Every training view of the V4 data (E2 D100 outdoor shards: rows (t, a, R, T, jitter) of Train-6 trajectories) is
rasterised with the T1.1 label camera (futureview.rays.label_intrinsics: the 64 x 114 pixel centres are exactly the
planner's rays, jitter included) and the frozen sensor settings (both faces, z clip 0.5, no blur), but with
faces_per_pixel = K: the rasteriser returns, per ray, the K nearest triangle intersections sorted by depth (depth
peeling). t_k = zbuf_k / dz / k_M (model units, the frame of the V4 rays).

Per ray (all 7,296 rays of a view, for the statistics) and for a stored subset of rays per view:
  surfaces   sorted intersection depths after merging numerical duplicates (|t_i - t_j| < tol_dup: shared edges,
             coincident faces); depths before the near clip are not returned by the rasteriser
  valid      [t_lo, t_hi]: t_lo = near clip, t_hi = min(sensor range tcap, exit of the scene grid box, t_K if all K
             layers were used -> beyond the K-th layer the mesh is not resolved = UNSUPERVISED, never negative)
Checks: layer 1 == the stored T1.1 first hit t* (same camera, same rasteriser) for every ray.
Modes
  test    synthetic two-wall test (walls at 3 and 9 runtime units, a duplicated face, a wall before the near clip) and
          an exact multi-hit comparison with trimesh (multiple_hits=True) on sampled real rays of one Train-6 scene
  probe   full statistics of one trajectory with a large K (saturation of K, surfaces per ray, gaps t2 - t1)
  run     label cache <out>/<scene>/<traj>.npz for the given trajectories (resumable)
Only Train-6 geometry can be loaded (t0_common.assert_geometry_allowed(scene, "train") inside t11_labels.load_mesh)."""
import argparse
import hashlib
import json
import os
import sys
import time

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task1", "scripts"))
import t11_labels as L11  # noqa: E402  (load_mesh, label_cameras, frame_of; sets the exp3b / task0 paths)

MC, RY, C0 = L11.MC, L11.RY, L11.C0
np, torch = L11.np, L11.torch
SHARDS = os.path.join(R, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2", "out_views_d100")
VS = 0.011
SCHEMA = "v5-multisurface-labels-v1"


def rasterizer_k(K, max_faces_per_bin, bin_size=None):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    rs = RasterizationSettings(image_size=(RY.GH, RY.GW), blur_radius=0.0, faces_per_pixel=K, z_clip_value=MC.NEAR_CLIP,
                               cull_backfaces=False, perspective_correct=True, max_faces_per_bin=max_faces_per_bin, bin_size=bin_size)
    return MeshRasterizer(raster_settings=rs)


def merge_dups(t, tol):
    """t [N, K] sorted ascending (inf = none) -> merged [N, K] (duplicates within tol removed), count [N]."""
    out = np.full_like(t, np.inf)
    cnt = np.zeros(len(t), np.int32)
    keep = np.isfinite(t)
    with np.errstate(invalid="ignore"):
        keep[:, 1:] &= ~(np.isfinite(t[:, :-1]) & (t[:, 1:] - t[:, :-1] < tol))
    for i in np.flatnonzero(keep.any(1)):
        v = t[i, keep[i]]
        out[i, :len(v)] = v
        cnt[i] = len(v)
    return out, cnt


def ray_box_exit(o, d, ext):
    """Exit distance of rays o + t d (model frame) from the box [0, ext]^3 (inf if the ray never leaves forward)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (0.0 - o[None]) / d
        t2 = (ext[None] - o[None]) / d
    tmax = np.where(d > 0, t2, np.where(d < 0, t1, np.inf))
    return tmax.min(1)


class Peeler:
    def __init__(self, mesh, dev, K, batch=8):
        self.mesh, self.dev, self.K, self.batch = mesh, dev, K, batch
        self.ras = rasterizer_k(K, int(mesh.num_faces_per_mesh()[0]))

    def layers(self, Rs, Ts, jits, k_M):
        """-> t [B, 7296, K] model units (inf = no layer), tcap / o / d not included."""
        out = []
        for s in range(0, len(Rs), self.batch):
            R_, T_, J_ = Rs[s:s + self.batch], Ts[s:s + self.batch], jits[s:s + self.batch]
            fr = self.ras(self.mesh.extend(len(R_)), cameras=L11.label_cameras(R_, T_, J_, self.dev))
            z = fr.zbuf.reshape(len(R_), -1, self.K).double().cpu().numpy()
            hit = fr.pix_to_face.reshape(len(R_), -1, self.K).cpu().numpy() >= 0
            for b in range(len(R_)):
                u, v = RY.grid_uv(J_[b])
                _, dz, _ = MC.pixel_rays(R_[b], T_[b], u, v)
                out.append(np.where(hit[b] & (z[b] > 0), z[b] / dz[:, None] / k_M, np.inf))
        return np.stack(out)


def view_geometry(fr, Rs, Ts, Js, k_M):
    o, d, tcap, dz = RY.model_rays_torch(torch.as_tensor(np.asarray(Rs), dtype=torch.float64), torch.as_tensor(np.asarray(Ts), dtype=torch.float64),
                                         torch.as_tensor(np.asarray(Js), dtype=torch.float64), torch.as_tensor(np.asarray(fr.origin_z), dtype=torch.float64), k_M)
    return o.numpy(), d.numpy(), tcap.numpy(), dz.numpy()


def label_views(peel, fr, z, rows, n_sel, tol_dup, seed_key, k_store=8):
    """Labels of the shard rows -> dict of arrays (subset per view) + per-view statistics."""
    Rs, Ts, Js = z["R"][rows], z["T"][rows], z["jitter"][rows]
    k_M = float(z["k_M"])
    ext = np.asarray(z["dims"], np.float64) * VS
    t = peel.layers(list(Rs), list(Ts), list(Js), k_M)                                  # [B, R, K]
    o, d, tcap, dz = view_geometry(fr, Rs, Ts, Js, k_M)
    near = MC.NEAR_CLIP / dz / k_M
    B, Rn, K = t.shape
    out = {k: [] for k in ("ray_idx", "surf", "n_surf", "t_lo", "t_hi")}
    stats = {"hist": [], "first_eq": [], "first_maxrel": [], "saturated": [], "gap21": [], "n_ge2_valid": []}
    rng = np.random.default_rng(int(hashlib.sha256(seed_key.encode()).hexdigest()[:15], 16))
    for b in range(B):
        tb, cnt = merge_dups(t[b], tol_dup)
        sat = np.isfinite(t[b][:, K - 1])                                                 # all K layers used -> unresolved beyond
        hi = np.minimum(tcap[b], ray_box_exit(o[b], d[b], ext))
        hi = np.where(sat, np.minimum(hi, t[b][:, K - 1]), hi)
        lo = near[b]
        inr = (tb >= lo[:, None]) & (tb <= hi[:, None])
        n_in = inr.sum(1)
        # store at most k_store surfaces: if more lie in range, end the supervised range half-way to the next surface
        tt_all = np.where(inr, tb, np.inf)
        tt_all.sort(1)
        over = n_in > k_store
        if over.any():
            hi = np.where(over, 0.5 * (tt_all[:, k_store - 1] + np.where(over, tt_all[:, min(k_store, K - 1)], 0.0)), hi)
            inr = (tb >= lo[:, None]) & (tb <= hi[:, None])
            n_in = inr.sum(1)
        ts_ref = z["tstar"][rows[b]].astype(np.float64)
        f_ok = np.isfinite(ts_ref) == np.isfinite(t[b][:, 0])
        both = np.isfinite(ts_ref) & np.isfinite(t[b][:, 0])
        rel = np.abs(t[b][both, 0] - ts_ref[both]) / np.maximum(ts_ref[both], 1e-9)
        stats["first_eq"].append(float(f_ok.mean()))
        stats["first_maxrel"].append(float(rel.max()) if len(rel) else 0.0)
        stats["hist"].append(np.bincount(np.minimum(n_in, 6), minlength=7))
        stats["saturated"].append(float(sat.mean()))
        m2 = n_in >= 2
        stats["n_ge2_valid"].append(int(m2.sum()))
        if m2.any():
            tt = np.where(inr, tb, np.inf)
            tt.sort(1)
            stats["gap21"].append((tt[m2, 1] - tt[m2, 0]).astype(np.float32))
        A = np.flatnonzero(m2)
        Bp = np.flatnonzero(~m2)
        na = min(len(A), n_sel // 2)
        sel = np.concatenate([rng.choice(A, na, replace=False) if na else np.zeros(0, int),
                              rng.choice(Bp, min(len(Bp), n_sel - na), replace=False)])
        sel = np.sort(sel)
        pad = n_sel - len(sel)
        surf = np.where(inr[sel], tb[sel], np.inf)
        surf.sort(1)
        surf = surf[:, :k_store]
        out["ray_idx"].append(np.concatenate([sel, np.full(pad, -1)]).astype(np.int16))
        out["surf"].append(np.concatenate([surf, np.full((pad, k_store), np.inf)]).astype(np.float16))
        out["n_surf"].append(np.concatenate([n_in[sel], np.zeros(pad, int)]).astype(np.uint8))
        out["t_lo"].append(np.concatenate([lo[sel], np.zeros(pad)]).astype(np.float16))
        out["t_hi"].append(np.concatenate([hi[sel], np.zeros(pad)]).astype(np.float16))
    res = {k: np.stack(v) for k, v in out.items()}
    return res, stats


def mesh_sha(obj):
    return C0.sha256_file(obj)


def mode_test(a):
    """Synthetic two-wall test + exact trimesh multi-hit comparison on real rays."""
    from pytorch3d.structures import Meshes
    dev = torch.device("cuda")
    res = {}
    # (1) synthetic: identity camera, walls at view z = 3 and 9 covering the frustum, the 9-wall duplicated (coincident
    # faces), a wall at z = 0.3 (before the near clip 0.5, must not appear); k_M = 1 -> model t = runtime ray length
    def quad(zv, s):
        return np.array([[-s, -s, zv], [s, -s, zv], [s, s, zv], [-s, s, zv]], np.float64), np.array([[0, 1, 2], [0, 2, 3]])
    V, F = [], []
    for zv, s in ((3.0, 5.0), (9.0, 15.0), (9.0, 15.0), (0.3, 1.0)):
        v, f = quad(zv, s)
        F.append(f + sum(len(x) for x in V)); V.append(v)
    mesh = Meshes(verts=[torch.as_tensor(np.concatenate(V), dtype=torch.float32, device=dev)], faces=[torch.as_tensor(np.concatenate(F), device=dev)])
    peel = Peeler(mesh, dev, 8, 1)
    for jit in ((0.0, 0.0), (1.3, -0.7)):
        t = peel.layers([np.eye(3)], [np.zeros(3)], [jit], 1.0)[0]
        u, v = RY.grid_uv(jit)
        _, dz, _ = MC.pixel_rays(np.eye(3), np.zeros(3), u, v)
        tb, cnt = merge_dups(t, 0.5 * VS)
        e1 = np.abs(tb[:, 0] - 3.0 / dz).max()
        e2 = np.abs(tb[:, 1] - 9.0 / dz).max()
        res[f"synthetic_jit{jit}"] = {"rays": int(len(tb)), "count_after_merge": np.bincount(cnt).tolist(), "raw_layers_mean": float(np.isfinite(t).sum(1).mean()),
                                      "max_err_wall_A": float(e1), "max_err_wall_B": float(e2), "near_clip_wall_absent": bool((tb[np.isfinite(tb)] > 1.0).all()),
                                      "pass": bool((cnt == 2).all() and e1 < 1e-3 and e2 < 1e-3)}
    # (2) exact multi-hit agreement with trimesh on real rays of one Train-6 trajectory
    import trimesh
    mesh_r, obj = L11.load_mesh(a.scene, dev)
    vtx = mesh_r.verts_packed().cpu().numpy().astype(np.float64)
    fcs = mesh_r.faces_packed().cpu().numpy()
    tm = trimesh.Trimesh(vertices=vtx, faces=fcs, process=False)
    traj = a.trajs.split(",")[0]
    z = np.load(os.path.join(SHARDS, a.scene, f"{traj}.npz"))
    fr = L11.frame_of(a.scene)
    k_M = float(z["k_M"])
    peel = Peeler(mesh_r, dev, a.K, 4)
    rng = np.random.default_rng(0)
    rows = rng.choice(len(z["t"]), 4, replace=False)
    agree, errs, n_cmp, n_hits = 0, [], 0, []
    for r in rows:
        Rm, Tm, J = z["R"][r], z["T"][r], z["jitter"][r]
        t = peel.layers([Rm], [Tm], [J], k_M)[0]
        tb, cnt = merge_dups(t, 0.5 * VS)
        u, v = RY.grid_uv(J)
        d_y, dz, c = MC.pixel_rays(Rm, Tm, u, v)
        idx = rng.choice(len(dz), a.n_rays // len(rows), replace=False)
        locs, ri, _ = tm.ray.intersects_location(np.repeat(c[None], len(idx), 0), d_y[idx], multiple_hits=True)
        for q, i in enumerate(idx):
            L = np.sort(np.linalg.norm(locs[ri == q] - c[None], axis=1)) / k_M                      # model units along the ray
            L = L[L * k_M * dz[i] > MC.NEAR_CLIP]                                                   # same near clip (view z)
            Lm, _ = merge_dups(np.concatenate([L, np.full(max(0, 64 - len(L)), np.inf)])[None, :64], 0.5 * VS)
            Lm = Lm[0][np.isfinite(Lm[0])]
            Kc = min(a.K, len(Lm)) if not np.isfinite(t[i, a.K - 1]) else min(cnt[i], len(Lm))
            ours = tb[i][np.isfinite(tb[i])][:Kc]
            ref = Lm[:Kc]
            n_cmp += 1
            n_hits.append(len(Lm))
            if len(ours) == len(ref) and (len(ref) == 0 or np.allclose(ours, ref, rtol=1e-4, atol=1e-4)):
                agree += 1
            elif len(ours) == len(ref):
                errs.append(float(np.max(np.abs(ours - ref))))
    res["trimesh"] = {"scene": a.scene, "traj": traj, "rays": n_cmp, "agree_frac": agree / max(n_cmp, 1), "n_disagree": n_cmp - agree,
                      "disagree_max_abs": errs[:20], "hits_per_ray_hist": np.bincount(np.minimum(n_hits, 8)).tolist()}
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, f"test_{a.scene}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


def mode_run(a, probe=False):
    dev = torch.device("cuda")
    mesh, obj = L11.load_mesh(a.scene, dev)
    sha = mesh_sha(obj)
    fr = L11.frame_of(a.scene)
    peel = Peeler(mesh, dev, a.K, a.batch)
    trajs = a.trajs.split(",") if a.trajs else sorted(f[:-4] for f in os.listdir(os.path.join(SHARDS, a.scene)) if f.endswith(".npz"))
    for traj in trajs:
        p = os.path.join(a.out, a.scene, f"{traj}.npz")
        if os.path.exists(p) and not probe:
            continue
        tb = time.time()
        z = np.load(os.path.join(SHARDS, a.scene, f"{traj}.npz"))
        rows = np.arange(len(z["t"]))
        acc = {k: [] for k in ("ray_idx", "surf", "n_surf", "t_lo", "t_hi")}
        st = {"hist": [], "first_eq": [], "first_maxrel": [], "saturated": [], "gap21": [], "n_ge2_valid": []}
        for s in range(0, len(rows), 64):
            r_ = rows[s:s + 64]
            res, stats = label_views(peel, fr, z, r_, a.n_sel, a.tol_dup * VS, f"exp6-V5|labels|{a.scene}|{traj}|{s}", a.k_store)
            for k in acc:
                acc[k].append(res[k])
            for k in st:
                st[k] += stats[k]
        out = {k: np.concatenate(v) for k, v in acc.items()}
        gap = np.concatenate(st["gap21"]) if st["gap21"] else np.zeros(0, np.float32)
        meta = {"schema": SCHEMA, "scene": a.scene, "traj": traj, "mesh_obj": obj, "mesh_sha256": sha, "K": a.K, "k_store": a.k_store, "tol_dup_vox": a.tol_dup, "n_sel": a.n_sel,
                "k_M": float(z["k_M"]), "heldout": bool(z["heldout"]), "rows": int(len(rows)), "rows_sha256": hashlib.sha256(np.stack([z["t"], z["a"]]).tobytes()).hexdigest(),
                "shard": os.path.join(SHARDS, a.scene, f"{traj}.npz"), "time": time.strftime("%F %T"), "seconds": time.time() - tb}
        summ = {"hist_all_rays": np.sum(st["hist"], 0).tolist(), "first_eq_min": float(np.min(st["first_eq"])), "first_maxrel_max": float(np.max(st["first_maxrel"])),
                "saturated_mean": float(np.mean(st["saturated"])), "gap21_vox_quantiles": (np.quantile(gap / VS, [0.05, 0.25, 0.5, 0.75, 0.95]).tolist() if len(gap) else None)}
        meta["summary"] = summ
        if probe:
            print(json.dumps(meta, indent=1))
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f".tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, meta=json.dumps(meta), t=z["t"], a=z["a"], gap21=gap.astype(np.float16), **out)
        os.replace(tmp, p)
        print(f"{a.scene}/{traj}: {len(rows)} views, rays>=2 surfaces {summ['hist_all_rays'][2:]} / total {sum(summ['hist_all_rays'])}, "
              f"first==t* min {summ['first_eq_min']:.4f} maxrel {summ['first_maxrel_max']:.1e}, sat {summ['saturated_mean']:.4f}, {time.time() - tb:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "probe", "run"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--trajs", default=None)
    ap.add_argument("--K", type=int, default=16, help="peeled layers")
    ap.add_argument("--k-store", type=int, default=8, help="stored surfaces per ray")
    ap.add_argument("--n-sel", type=int, default=1024)
    ap.add_argument("--n-rays", type=int, default=2000)
    ap.add_argument("--tol-dup", type=float, default=0.5, help="duplicate-merge tolerance in voxels")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(R, "work_dirs", "exp6_v5", "labels", "v1"))
    a = ap.parse_args()
    C0.assert_geometry_allowed(a.scene, "train")
    if a.mode == "test":
        mode_test(a)
    else:
        mode_run(a, probe=a.mode == "probe")


if __name__ == "__main__":
    main()
