"""EXP5 Task 1.1 D-LABEL (magician env, 1 GPU; Train-6 GT meshes only; label side, never imported by the planner).

First GT intersection t* of every candidate view's 114 x 64 rays (protocol §2.4), computed with the PyTorch3D
rasteriser of the frozen sensor (MeshRasterizer, faces_per_pixel 1, both faces, z clip 0.5) and a PerspectiveCameras
whose 64 x 114 pixel centres are exactly the MAGICIAN-convention rays of the (shifted) scoring grid
(futureview.rays.label_intrinsics). zbuf is the view-space z of the first face along the ray -> t*_runtime = z / dz,
t*_model = t*_runtime / k_M. (Deviation DEV-1: the magician env has no embree; the rasteriser computes the same exact
first ray-triangle intersection along the same rays.)

Modes
  test   : T-L0 pixel-centre placement (a tiny triangle on a target ray must cover exactly that pixel), T-L6 agreement
           with trimesh's exact ray-triangle intersector on sampled rays, timing; one scene
  check  : T-L3 (FoV camera of the sensor at 256 x 456 vs the recorded zbuf of the executed next frame) and T-L4
           (MAGICIAN-convention integer-pixel rays vs the recorded zbuf), 50 frames per scene
  run    : t* of all candidate views of the given trajectories -> <out>/<scene>/<traj>.npz
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
T1 = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(T1, "..", "..", ".."))
EXP3B = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer")
DATA = os.path.join(REPO, "data", "macaronspp_belief", "v1")
sys.path.insert(0, EXP3B)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp4_multiworld_belief", "task0", "scripts"))  # = data/v1/code
sys.path.insert(0, os.path.join(REPO, "seen2scene", "futureview"))
import m_common as MC  # noqa: E402
import t0_common as C0  # noqa: E402
import rays as RY  # noqa: E402  (futureview/rays.py, numpy only; imported by file so the magician env needs no package)

import numpy as np  # noqa: E402
import torch  # noqa: E402

warnings.filterwarnings("error", message=".*Bin size was too small.*")
CFG = json.load(open(os.path.join(T1, "configs", "data_v1.json")))
FV = json.load(open(os.path.join(T1, "frozen_values.json")))


def load_mesh(scene, dev):
    import exp3b_paths as P
    from pytorch3d.io import load_obj
    from pytorch3d.structures import Meshes
    C0.assert_geometry_allowed(scene, "train")
    objs = sorted(glob.glob(os.path.join(P.scene_dir(scene), "*.obj")))
    assert len(objs) == 1, objs
    verts, faces, _ = load_obj(objs[0], load_textures=False, device=dev)
    return Meshes(verts=[verts * 10.0], faces=[faces.verts_idx]), objs[0]


def rasterizer(image_size, max_faces_per_bin, bin_size=None):
    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
    rs = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, z_clip_value=MC.NEAR_CLIP,
                               cull_backfaces=False, perspective_correct=True, max_faces_per_bin=max_faces_per_bin,
                               bin_size=bin_size)
    return MeshRasterizer(raster_settings=rs)


def label_cameras(Rs, Ts, jits, dev):
    from pytorch3d.renderer import PerspectiveCameras
    f = np.asarray([RY.label_intrinsics(j) for j in jits], dtype=np.float32)
    return PerspectiveCameras(focal_length=torch.as_tensor(f[:, :2], device=dev),
                              principal_point=torch.as_tensor(f[:, 2:], device=dev),
                              R=torch.as_tensor(np.asarray(Rs), dtype=torch.float32, device=dev),
                              T=torch.as_tensor(np.asarray(Ts), dtype=torch.float32, device=dev), in_ndc=True, device=dev)


class Renderer:
    def __init__(self, mesh, dev, max_faces_per_bin=0, bin_size=None, batch=8):
        """max_faces_per_bin 0 = the number of faces of the mesh: a bin can never overflow (coarse-to-fine output is
        then complete; PyTorch3D only prints, does not raise, on overflow)."""
        self.mesh, self.dev, self.batch = mesh, dev, batch
        self.n_faces = int(mesh.num_faces_per_mesh()[0])
        self.ras = rasterizer((RY.GH, RY.GW), max_faces_per_bin or self.n_faces, bin_size)

    def tstar(self, Rs, Ts, jits, k_M):
        """t* [B, 7296] float64 model units (inf = no hit) for views (R, T, jitter)."""
        out = []
        for s in range(0, len(Rs), self.batch):
            R_, T_, J_ = Rs[s:s + self.batch], Ts[s:s + self.batch], jits[s:s + self.batch]
            cams = label_cameras(R_, T_, J_, self.dev)
            fr = self.ras(self.mesh.extend(len(R_)), cameras=cams)
            z = fr.zbuf[..., 0].reshape(len(R_), -1).double().cpu().numpy()
            hit = fr.pix_to_face[..., 0].reshape(len(R_), -1).cpu().numpy() >= 0
            for b in range(len(R_)):
                u, v = RY.grid_uv(J_[b])
                _, dz, _ = MC.pixel_rays(R_[b], T_[b], u, v)
                out.append(np.where(hit[b] & (z[b] > 0), z[b] / dz / k_M, np.inf))
        return np.stack(out)


def frame_of(scene):
    em2 = json.load(open(os.path.join(REPO, "work_dirs", "exp3b_macaronspp_transfer", "task1_protocol", "em2_windows.json")))
    import exp3b_paths as P
    settings = json.load(open(os.path.join(P.scene_dir(scene), "settings.json")))
    return MC.SceneFrame(scene, settings, em2["scenes"][scene] if "scenes" in em2 else em2[scene])


def records(scene, traj):
    d = os.path.join(DATA, "trajectories", scene, traj, "records")
    fs = {int(os.path.basename(f).split("_t")[-1][:-4]): f for f in glob.glob(os.path.join(d, "svc_*.npz"))}
    assert sorted(fs) == list(range(101)), (scene, traj, len(fs))
    return fs


def is_heldout(scene, traj):
    return traj in FV["split"]["scenes"][scene]["heldout_trajs"]


# ---------------------------------------------------------------------------------------------------- test (T-L0, T-L6)
def mode_test(a):
    from pytorch3d.structures import Meshes
    dev = torch.device("cuda")
    res = {"scene": a.scene}
    # T-L0: one tiny triangle facing the camera on the ray of pixel (i, j) -> pix_to_face >= 0 exactly at (i, j)
    rng = np.random.default_rng(0)
    R = np.eye(3)
    T = np.zeros(3)
    ok, n = 0, 0
    for jit in [(0.0, 0.0), (0.5, 0.5), (-2.0, -2.0), (1.9999, -1.3)] + [tuple(rng.uniform(-2, 2, 2)) for _ in range(4)]:
        u, v = RY.grid_uv(jit)
        d_y, dz, c = MC.pixel_rays(R, T, u, v)
        for (i, j) in [(0, 0), (0, 113), (63, 0), (63, 113), (31, 57), (10, 90)] + [tuple(rng.integers([64, 114])) for _ in range(6)]:
            r = i * RY.GW + j
            p = c + d_y[r] * 7.0
            z = p @ R + T                      # view coords (X right? PyTorch3D: +X left); triangle in the plane z = const
            eps = 0.002 * z[2]                 # << pixel footprint (block 4 px ~ 0.016 z)
            tri = np.stack([z + [eps, 0, 0], z + [-eps, eps, 0], z + [-eps, -eps, 0]])
            tri_w = (tri - T) @ R.T
            mesh = Meshes(verts=[torch.as_tensor(tri_w, dtype=torch.float32, device=dev)],
                          faces=[torch.as_tensor([[0, 1, 2]], device=dev)])
            fr = rasterizer((RY.GH, RY.GW), 1000, 0)(mesh, cameras=label_cameras([R], [T], [jit], dev))
            cov = torch.nonzero(fr.pix_to_face[0, ..., 0] >= 0).cpu().numpy().tolist()
            n += 1
            ok += cov == [[i, j]]
            if cov != [[i, j]]:
                print("T-L0 mismatch", jit, (i, j), cov)
    res["TL0"] = {"n": n, "ok": ok, "pass": ok == n}
    print("T-L0", res["TL0"])
    # T-L6 + timing on real candidate views of one trajectory
    mesh, obj = load_mesh(a.scene, dev)
    fr_ = frame_of(a.scene)
    traj = sorted(os.listdir(os.path.join(DATA, "trajectories", a.scene)))[0]
    fs = records(a.scene, traj)
    Rs, Ts, Js = [], [], []
    for t in range(0, 100, 10):
        z = np.load(fs[t])
        for ai in range(len(z["legal_R"])):
            Rs.append(z["legal_R"][ai]); Ts.append(z["legal_T"][ai]); Js.append(RY.jitter_of(a.scene, traj, t, ai))
    ren = Renderer(mesh, dev, a.max_faces_per_bin, a.bin_size, a.batch)
    torch.cuda.synchronize(); t0 = time.time()
    ts = ren.tstar(Rs, Ts, Js, fr_.k_M)
    torch.cuda.synchronize(); dt = time.time() - t0
    res["timing"] = {"views": len(Rs), "s": dt, "views_per_s": len(Rs) / dt}
    print("timing", res["timing"], "hit frac", float(np.isfinite(ts).mean()))
    import trimesh
    m = trimesh.load(obj, force="mesh", process=False)
    m.vertices *= 10.0
    inter = trimesh.ray.ray_triangle.RayMeshIntersector(m)
    agree, dts = [], []
    sel = rng.choice(len(Rs), 8, replace=False)
    for b in sel:
        u, v = RY.grid_uv(Js[b])
        d_y, dz, c = MC.pixel_rays(Rs[b], Ts[b], u, v)
        rr = rng.choice(len(u), 250, replace=False)
        loc, ir, _ = inter.intersects_location(np.repeat(c[None], len(rr), 0), d_y[rr], multiple_hits=True)
        tt = np.full(len(rr), np.inf)
        if len(ir):
            dist = np.linalg.norm(loc - c[None], axis=1)
            np.minimum.at(tt, ir, dist)
        tt = tt / fr_.k_M
        tr = ts[b][rr]
        fin = np.isfinite(tt) & np.isfinite(tr)
        agree.append(float((np.isfinite(tt) == np.isfinite(tr)).mean()))
        dts.extend((np.abs(tt[fin] - tr[fin]) / tt[fin]).tolist())
    dts = np.asarray(dts)
    res["TL6"] = {"rays": 8 * 250, "hit_agree": float(np.mean(agree)), "rel_err_median": float(np.median(dts)),
                  "rel_err_p99": float(np.quantile(dts, 0.99)), "rel_err_max": float(dts.max())}
    print("T-L6", res["TL6"])
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- check (T-L3, T-L4)
def silhouette(z, mask, rel=0.05):
    zz = np.where(mask, z, np.inf)
    sil = np.zeros_like(mask)
    H_, W_ = z.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == dj == 0:
                continue
            sh = np.full_like(zz, np.inf)
            sm = np.zeros_like(mask)
            ys, ye = max(0, di), H_ + min(0, di)
            xs, xe = max(0, dj), W_ + min(0, dj)
            sh[ys - di:ye - di, xs - dj:xe - dj] = zz[ys:ye, xs:xe]
            sm[ys - di:ye - di, xs - dj:xe - dj] = mask[ys:ye, xs:xe]
            sil |= (sm != mask)
            both = np.isfinite(sh) & np.isfinite(zz)
            sil |= both & (np.abs(sh - zz) > rel * np.minimum(sh, zz))
    return sil


def mode_check(a):
    from pytorch3d.renderer import FoVPerspectiveCameras
    dev = torch.device("cuda")
    mesh, _ = load_mesh(a.scene, dev)
    rng = np.random.default_rng(int(RY.sha_seed(f"exp5-V2|TL3|{a.scene}")))
    trajs = sorted(os.listdir(os.path.join(DATA, "trajectories", a.scene)))
    picks = [(trajs[i], int(t)) for i, t in zip(rng.integers(0, len(trajs), a.frames), rng.integers(0, 100, a.frames))]
    full = rasterizer((MC.H, MC.W), a.max_faces_per_bin or int(mesh.num_faces_per_mesh()[0]), a.bin_size)
    ren = Renderer(mesh, dev, a.max_faces_per_bin, a.bin_size, 1)
    fr_ = frame_of(a.scene)
    rows = []
    for traj, t in picks:
        fs = records(a.scene, traj)
        z0, z1 = np.load(fs[t]), np.load(fs[t + 1])
        R, T = z1["R"], z1["T"]
        pose_ok = None
        hits = [i for i, x in enumerate(z0["legal_idx"]) if list(x) == list(z1["pose_idx"])]
        if hits:
            pose_ok = bool(np.array_equal(z0["legal_R"][hits[0]], R) and np.array_equal(z0["legal_T"][hits[0]], T))
        zr, mr = z1["zbuf"].astype(np.float64), z1["mask"]
        cams = FoVPerspectiveCameras(R=torch.as_tensor(R[None], dtype=torch.float32, device=dev),
                                     T=torch.as_tensor(T[None], dtype=torch.float32, device=dev), device=dev)
        f = full(mesh, cameras=cams)
        zf = f.zbuf[0, ..., 0].double().cpu().numpy()
        mf = (f.pix_to_face[0, ..., 0] >= 0).cpu().numpy() & (zf > 0)
        sil = silhouette(zr, mr, CFG["checks"]["silhouette_rel_jump"])
        ns = ~sil
        both = ns & mr & mf
        row = {"traj": traj, "t": t, "pose_consistent": pose_ok,
               "TL3_hit_agree": float((mf[ns] == mr[ns]).mean()),
               "TL3_relz_median": float(np.median(np.abs(zf[both] - zr[both]) / zr[both])) if both.any() else None,
               "TL3_bitwise_frac": float((zf[both] == zr[both]).mean()) if both.any() else None}
        # T-L4: MAGICIAN-convention rays at integer pixels (4 j + 2, 4 i + 2)
        tl = ren.tstar([R], [T], [(0.5, 0.5)], fr_.k_M)[0]
        u, v = RY.grid_uv((0.5, 0.5))
        ui, vi = u.astype(int), v.astype(int)
        _, dz, _ = MC.pixel_rays(R, T, u, v)
        zl = tl * fr_.k_M * dz
        ml = np.isfinite(zl)
        zr4, mr4, ns4 = zr[vi, ui], mr[vi, ui], ns[vi, ui]
        b4 = ns4 & ml & mr4
        row.update({"TL4_hit_agree": float((ml[ns4] == mr4[ns4]).mean()),
                    "TL4_relz_median": float(np.median(np.abs(zl[b4] - zr4[b4]) / zr4[b4])) if b4.any() else None,
                    "TL4_relz_p95": float(np.quantile(np.abs(zl[b4] - zr4[b4]) / zr4[b4], 0.95)) if b4.any() else None})
        rows.append(row)
    agg = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in
           ("TL3_hit_agree", "TL3_relz_median", "TL3_bitwise_frac", "TL4_hit_agree", "TL4_relz_median", "TL4_relz_p95")}
    agg["TL3_relz_median_of_frames"] = float(np.median([r["TL3_relz_median"] for r in rows if r["TL3_relz_median"] is not None]))
    agg["TL4_relz_median_of_frames"] = float(np.median([r["TL4_relz_median"] for r in rows if r["TL4_relz_median"] is not None]))
    agg["pose_consistent_all"] = all(r["pose_consistent"] in (True, None) for r in rows)
    ck = CFG["checks"]
    agg["TL3_pass"] = (min(r["TL3_hit_agree"] for r in rows) >= ck["TL3_hit_agree_min"]
                       and agg["TL3_relz_median_of_frames"] <= ck["TL3_rel_z_median_max"])
    agg["TL4_pass"] = (agg["TL4_hit_agree"] >= ck["TL4_hit_agree_min"] and agg["TL4_relz_median_of_frames"] <= ck["TL4_rel_z_median_max"])
    print(a.scene, json.dumps(agg))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"scene": a.scene, "frames": rows, "summary": agg}, open(a.out, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------- run
def mode_run(a):
    dev = torch.device("cuda")
    mesh, obj = load_mesh(a.scene, dev)
    fr_ = frame_of(a.scene)
    ren = Renderer(mesh, dev, a.max_faces_per_bin, a.bin_size, a.batch)
    trajs = a.trajs.split(",") if a.trajs else sorted(os.listdir(os.path.join(DATA, "trajectories", a.scene)))
    for traj in trajs:
        p = os.path.join(a.out, a.scene, f"{traj}.npz")
        if os.path.exists(p):
            continue
        tb = time.time()
        fs = records(a.scene, traj)
        hold = is_heldout(a.scene, traj)
        T_, A_, Rs, Ts, Js = [], [], [], [], []
        for t in range(100):
            z = np.load(fs[t])
            for ai in range(len(z["legal_R"])):
                T_.append(t); A_.append(ai); Rs.append(z["legal_R"][ai]); Ts.append(z["legal_T"][ai])
                Js.append((0.0, 0.0) if hold else RY.jitter_of(a.scene, traj, t, ai))
        ts = ren.tstar(Rs, Ts, Js, fr_.k_M).astype(np.float32)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f".tmp{os.getpid()}.npz"
        np.savez(tmp, tstar=ts, t=np.asarray(T_, np.int16), a=np.asarray(A_, np.int8), R=np.asarray(Rs), T=np.asarray(Ts),
                 jitter=np.asarray(Js, np.float64), heldout=np.asarray(hold), k_M=np.asarray(fr_.k_M),
                 obj_sha256=np.asarray(C0.sha256_file(obj)) if a.obj_sha else np.asarray(""))
        os.replace(tmp, p)
        print(f"{a.scene}/{traj}: {len(Rs)} views, hit {np.isfinite(ts).mean():.3f}, {time.time() - tb:.1f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["test", "check", "run"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--trajs", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--frames", type=int, default=CFG["checks"]["TL3_frames_per_scene"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-faces-per-bin", type=int, default=0)
    ap.add_argument("--bin-size", type=int, default=None)
    ap.add_argument("--obj-sha", action="store_true")
    a = ap.parse_args()
    {"test": mode_test, "check": mode_check, "run": mode_run}[a.mode](a)


if __name__ == "__main__":
    main()
