"""EXP5 Task 0 -- Matterport3D indoor simulator side (magician env; GT is used here only: rendering, collision /
line-of-sight, coverage). Mesh-rendered view protocol (M) of TASK0_RESTART_PLAN.md §2.4.

Conventions (T0.3, conventions.json): MP3D world = z-up metres. Runtime frame = MAGICIAN-style y-up with
1 m = S_RT = 7 runtime units, so the frozen sensor range 70 = 10 m (the MP3D sensor maximum) and every frozen
component (sensor, P1 fusion, V2, DistP1) runs unchanged; model units = runtime / k_I, k_I = level height / 2.816.

Sensor = the frozen MAGICIAN sensor: PyTorch3D FoVPerspectiveCameras(R, T, zfar 750; fov 60, znear 1), 256 x 456,
faces_per_pixel 1, blur 0; zbuf -1 = no hit; mask = zbuf > -1; depth = clamp(zbuf, 0.5, 750); 10 % sample points =
MAGICIAN compute_partial_point_cloud (NDC tabs, depth < 70, CPU randperm under the derived "sample" seed); R, T from
MAGICIAN get_camera_RT(X, [elev, azim]). max_faces_per_bin = number of faces (no bin overflow).
Scene = one level: floor = median panorama height - 1.5 m, ceiling = p99 of the level's mesh vertex heights (cap
floor + 4.5 m); xy box = panorama bbox +- 2 m clipped to the .house level box.
Poses = the level's panoramas x elev {-30, 0, 30} x azim 8 x 45 deg; legal actions = rotate +-1 azim / +-1 elev bin,
or move to a panorama-graph neighbour (<= 3 m, three parallel line-of-sight rays at camera height clear of the mesh).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
MP = "/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans"
W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402

S_RT = 7.0
TRIPOD_M = 1.5
ELEVS = (-30.0, 0.0, 30.0)
N_AZIM = 8
EDGE_MAX_M = 3.0
LOS_OFFSET_M = 0.15
XY_MARGIN_M = 2.0
CEIL_CAP_M = 4.5
N_WIN = 3
H_IMG, W_IMG, ZFAR, GATHER, RANGE = 256, 456, 750.0, 0.05 * 2, 70.0


# ------------------------------------------------------------------------------------------------ coordinates
def zup_m_to_rt(p):
    p = np.asarray(p, np.float64)
    return S_RT * np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def rt_to_zup_m(p):
    p = np.asarray(p, np.float64) / S_RT
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


# ------------------------------------------------------------------------------------------------ house file
def parse_house(scan):
    levels, regions, panos = {}, {}, []
    for line in open(os.path.join(MP, scan, "house_segmentations", f"{scan}.house")):
        f = line.split()
        if not f:
            continue
        if f[0] == "L":
            levels[int(f[1])] = {"box": [float(x) for x in f[7:13]]}
        elif f[0] == "R":
            regions[int(f[1])] = {"level": int(f[2]), "label": f[5], "box": [float(x) for x in f[9:15]]}
        elif f[0] == "P" and len(f) >= 8 and not f[1].isdigit():
            panos.append({"name": f[1], "idx": int(f[2]), "region": int(f[3]), "pos": [float(x) for x in f[5:8]]})
    for p in panos:
        p["level"] = regions[p["region"]]["level"] if p["region"] in regions else -1
    return {"levels": levels, "regions": regions, "panos": sorted(panos, key=lambda p: p["idx"])}


# ------------------------------------------------------------------------------------------------ mesh
def mesh_zup(scan):
    """Textured matterport_mesh geometry (z-up metres), cached as npz in W (derived data, not the dataset)."""
    cp = os.path.join(W, "preprocessing", "mesh_cache", f"{scan}.npz")
    if scan.startswith("OUT_"):                 # T0.5: Macarons++ scene, z-up OBJ units (runtime = OBJ x 10)
        return macaronspp_zup(scan[4:])
    if os.path.exists(cp):
        z = np.load(cp)
        return z["v"], z["f"]
    import trimesh
    obj = glob.glob(os.path.join(MP, scan, "matterport_mesh", "*", "*.obj"))
    assert len(obj) == 1, obj
    m = trimesh.load(obj[0], process=False, force="mesh", skip_materials=True)
    v, f = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.savez(cp, v=v, f=f, obj_sha256=hashlib.sha256(open(obj[0], "rb").read()).hexdigest())
    return v, f


class Level:
    """One level of an MP3D house as a Task 0 indoor scene."""

    def __init__(self, scan, level, center_pano=None, n_win=N_WIN):
        """center_pano (panorama idx of this level): restrict the scene to the n_win x n_win window region around it."""
        self.scan, self.level = scan, int(level)
        self.key = f"{scan}_L{level}" if center_pano is None else f"{scan}_L{level}_P{int(center_pano)}"
        h = parse_house(scan)
        self.panos = [p for p in h["panos"] if p["level"] == self.level]
        assert self.panos, (scan, level)
        P = np.asarray([p["pos"] for p in self.panos])
        lb = np.asarray(h["levels"][self.level]["box"])
        lo_xy = np.maximum(P[:, :2].min(0) - XY_MARGIN_M, lb[:2])
        hi_xy = np.minimum(P[:, :2].max(0) + XY_MARGIN_M, lb[3:5])
        # floor / ceiling: vertical rays from every panorama centre (first mesh hit below / above), medians
        import trimesh
        v, f = mesh_zup(scan)
        keep = np.all((v[f][..., :2] >= lo_xy - 1.0) & (v[f][..., :2] <= hi_xy + 1.0), axis=(1, 2))
        self._ri = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh.Trimesh(v, f[keep], process=False))
        down, up = [], []
        for p in P:
            for dz, acc in ((-1.0, down), (1.0, up)):
                loc, _, _ = self._ri.intersects_location([p], [[0.0, 0.0, dz]], multiple_hits=False)
                if len(loc):
                    acc.append(float(loc[0][2]))
        self.floor = float(np.median(down)) if down else float(np.median(P[:, 2]) - TRIPOD_M)
        ceil = float(np.median(up)) if up else self.floor + 3.0
        self.ceiling = min(ceil, self.floor + CEIL_CAP_M)
        self.vertical_rays = {"down_hits": len(down), "up_hits": len(up), "n": len(P),
                              "tripod_m_median": float(np.median(P[:, 2]) - self.floor)}
        self.height_m = self.ceiling - self.floor
        if center_pano is not None:              # region: n_win windows (overlap 0.2) per horizontal axis, clipped
            c0 = np.asarray(next(p["pos"] for p in self.panos if p["idx"] == int(center_pano)))[:2]
            half = 0.5 * (1 + (n_win - 1) * 0.8) * self.height_m * 0.99
            lo_xy, hi_xy = np.maximum(c0 - half, lo_xy), np.minimum(c0 + half, hi_xy)
            self.panos = [p for p in self.panos if lo_xy[0] <= p["pos"][0] <= hi_xy[0] and lo_xy[1] <= p["pos"][1] <= hi_xy[1]]
            P = np.asarray([p["pos"] for p in self.panos])
        self.box_m = np.array([[lo_xy[0], lo_xy[1], self.floor], [hi_xy[0], hi_xy[1], self.ceiling]])
        # runtime (y-up) box -> SceneFrame through a settings dict in the official x 10 convention
        c = zup_m_to_rt(self.box_m)
        rt_min, rt_max = c.min(0), c.max(0)
        settings = {"scene": {"x_min": (rt_min / MC.SCALE).tolist(), "x_max": (rt_max / MC.SCALE).tolist()}}
        self.k = (rt_max[1] - rt_min[1]) / (MC.PATCH * MC.VS)
        self.frame = MC.SceneFrame(self.key, settings, {"k_M": self.k})
        self.z0 = float(np.sum(rt_max - rt_min) / 3.0 / 2.0)          # MAGICIAN scene_scale / 2 (runtime units)
        self.pano_rt = zup_m_to_rt(P[:, :3])                             # camera centres (runtime)
        self.graph = None

    def describe(self):
        return {"key": self.key, "n_panos": len(self.panos), "floor_m": self.floor, "ceiling_m": self.ceiling, "vertical_rays": self.vertical_rays, "m_per_model_unit": self.height_m / (MC.PATCH * MC.VS),
                "height_m": self.height_m, "box_m": self.box_m.tolist(), "k_I": self.k, "z0_rt": self.z0,
                "extent_model": self.frame.extent_model.tolist(), "dims_scene": self.frame.dims_scene}

    # ---------------------------------------------------------------- panorama graph (line of sight, GT side)
    def build_graph(self):
        ri = self._ri
        P = np.asarray([p["pos"] for p in self.panos])
        n = len(P)
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                d = P[j] - P[i]
                L = float(np.linalg.norm(d))
                if L > EDGE_MAX_M or L < 1e-6:
                    continue
                u = d / L
                side = np.cross(u, [0.0, 0.0, 1.0])
                side = side / max(1e-9, np.linalg.norm(side))
                ok = True
                for off in (0.0, LOS_OFFSET_M, -LOS_OFFSET_M):
                    o = P[i] + off * side
                    loc, _, _ = ri.intersects_location([o], [u], multiple_hits=False)
                    if len(loc) and np.linalg.norm(loc[0] - o) < L:
                        ok = False
                        break
                if ok:
                    edges.append((i, j, L))
        self.graph = {i: sorted({b for a, b, _ in edges if a == i} | {a for a, b, _ in edges if b == i}) for i in range(n)}
        self.edges = edges
        return self.graph


# ------------------------------------------------------------------------------------------------ poses / actions
def pose_key(p):
    return (int(p[0]), 0, 0, int(p[3]), int(p[4]))


def legal_actions(level, pose):
    """Pose = (pano, 0, 0, elev_idx, azim_idx). Rotations first, then moves; sorted, duplicates removed."""
    pn, _, _, ei, ai = pose
    out = {(pn, 0, 0, ei, (ai + 1) % N_AZIM), (pn, 0, 0, ei, (ai - 1) % N_AZIM)}
    for de in (-1, 1):
        if 0 <= ei + de < len(ELEVS):
            out.add((pn, 0, 0, ei + de, ai))
    for q in level.graph[pn]:
        out.add((q, 0, 0, ei, ai))
    return sorted(out)


class Sensor:
    """The frozen MAGICIAN sensor on an arbitrary runtime-frame mesh."""

    def __init__(self, verts_rt, faces, device):
        import torch
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        from pytorch3d.structures import Meshes
        sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69")
        from macarons.utility.macarons_utils import get_camera_RT
        self.torch, self.dev, self.get_camera_RT = torch, device, get_camera_RT
        self.mesh = Meshes(verts=[torch.as_tensor(verts_rt, dtype=torch.float32, device=device)],
                           faces=[torch.as_tensor(faces, dtype=torch.int64, device=device)])
        rs = RasterizationSettings(image_size=(H_IMG, W_IMG), blur_radius=0.0, faces_per_pixel=1,
                                   max_faces_per_bin=max(200000, int(len(faces))))
        self.rast = MeshRasterizer(raster_settings=rs)
        x_tab = torch.Tensor([[i for j in range(W_IMG)] for i in range(H_IMG)]).to(device)
        y_tab = torch.Tensor([[j for j in range(W_IMG)] for i in range(H_IMG)]).to(device)
        mn = min(W_IMG, H_IMG)
        self.ndc_x = W_IMG / mn - (y_tab / (mn - 1)) * 2
        self.ndc_y = H_IMG / mn - (x_tab / (mn - 1)) * 2

    def RT(self, X_rt, elev, azim):
        t = self.torch
        X = t.tensor(np.asarray(X_rt, np.float32).reshape(1, 3), device=self.dev)
        V = t.tensor([[float(elev), float(azim)]], dtype=t.float32, device=self.dev)
        return self.get_camera_RT(X, V)

    def cam(self, R, T):
        from pytorch3d.renderer import FoVPerspectiveCameras
        return FoVPerspectiveCameras(R=R, T=T, zfar=ZFAR, device=self.dev)

    def render(self, R, T):
        t = self.torch
        fov = self.cam(R, T)
        with t.no_grad():
            frag = self.rast(self.mesh, cameras=fov)
        zbuf = frag.zbuf                                         # (1, H, W, 1), -1 = no hit
        return {"zbuf": zbuf, "mask": zbuf > -1, "fov": fov, "R": R, "T": T}

    def unproject(self, depth, fov):
        t = self.torch
        ndc = t.cat((self.ndc_x.view(1, -1, 1), self.ndc_y.view(1, -1, 1), depth.view(1, -1, 1)), -1)
        return fov.unproject_points(ndc.view(1, H_IMG * W_IMG, 3), scaled_depth_input=False)

    def sample(self, fr, seed):
        """MAGICIAN compute_partial_point_cloud (10 % of the pixels with depth < 70) + the sim's sample_flat."""
        t = self.torch
        depth = t.clamp(fr["zbuf"], min=0.5, max=750.0)
        m = fr["mask"].bool()
        pm = (m.view(1, -1) * (depth < RANGE).view(1, -1))
        allp = self.unproject(depth, fr["fov"])
        wp = allp[pm]
        n = int(len(wp) * GATHER)
        t.manual_seed(seed)
        idx = t.randperm(len(wp))[:n]
        pts = wp[idx]
        flat = t.nonzero(pm.view(-1))[:, 0].cpu()
        t.manual_seed(seed)
        sample_flat = flat[t.randperm(len(flat))[:int(len(flat) * GATHER)]]
        return pts, sample_flat, depth, allp, pm


def service_record(scene_key, start, t, fr, pts, sample_flat, pose, hist, legal, level, sensor):
    """Whitelisted service record (frozen format, m_common.validate_service_record)."""
    legal_R, legal_T, legal_C = [], [], []
    for a in legal:
        R, T = sensor.RT(level.pano_rt[a[0]], ELEVS[a[3]], 360.0 * a[4] / N_AZIM)
        legal_R.append(R[0].cpu().numpy())
        legal_T.append(T[0].cpu().numpy())
        legal_C.append(level.pano_rt[a[0]])
    rec = {"scene_key": np.asarray(f"{scene_key}/{start}"), "start": np.asarray(start), "t": np.asarray(t),
           "zbuf": fr["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32),
           "mask": fr["mask"][0, ..., 0].cpu().numpy().astype(bool),
           "R": fr["R"][0].cpu().numpy().astype(np.float64), "T": fr["T"][0].cpu().numpy().astype(np.float64),
           "cam_center": fr["fov"].get_camera_center()[0].cpu().numpy().astype(np.float64),
           "pose_idx": np.asarray(pose, np.int64), "sample_pixel_flat": sample_flat.numpy().astype(np.int64),
           "sample_points": pts.cpu().numpy().astype(np.float32), "executed_pose_idx_history": np.asarray(hist, np.int64),
           "legal_idx": np.asarray(legal, dtype=np.int64).reshape(-1, 5),
           "legal_R": np.asarray(legal_R, np.float64).reshape(-1, 3, 3),
           "legal_T": np.asarray(legal_T, np.float64).reshape(-1, 3),
           "legal_center": np.asarray(legal_C, np.float64).reshape(-1, 3)}
    return MC.validate_service_record(rec)


def derived_seed(scene_key, start, step, purpose):
    h = hashlib.sha256(f"exp5-T0|{scene_key}|{int(start)}|{int(step)}|{purpose}".encode()).hexdigest()
    return int(h[:15], 16)


# ------------------------------------------------------------------------------------------------ coverage (GT side)
EPS_MODEL = 0.08          # coverage tolerance in model units (outdoor Dev-3: 1 runtime unit = 0.059-0.087 model units)
N_GT_POINTS = 300000


def all_poses(level):
    return [(i, 0, 0, e, a) for i in range(len(level.panos)) for e in range(len(ELEVS)) for a in range(N_AZIM)]


class Coverage:
    """C_obs = |GT surface points within eps of an observed (10 % sample) point| / |GT points visible from any pose|.
    GT points: area-weighted samples on the textured mesh faces whose centroid is inside the scene box (seeded)."""

    def __init__(self, level, sensor, eps_model=EPS_MODEL, n_points=N_GT_POINTS, cache=True):
        from scipy.spatial import cKDTree
        self.level, self.eps_rt = level, eps_model * level.k
        cp = os.path.join(W, "preprocessing", "coverage", f"{level.key}.npz")
        if cache and os.path.exists(cp):
            z = np.load(cp)
            self.pts, self.visible = z["pts"], z["visible"]
        else:
            v, f = mesh_zup(level.scan)
            cen = v[f].mean(1)
            inb = np.all((cen >= level.box_m[0]) & (cen <= level.box_m[1]), axis=1)
            tri = v[f[inb]]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            rng = np.random.default_rng(derived_seed(level.key, -1, -1, "gt") % (2 ** 63))
            fi = rng.choice(len(tri), n_points, p=area / area.sum())
            r1, r2 = rng.random(n_points), rng.random(n_points)
            s = np.sqrt(r1)
            p = (1 - s)[:, None] * tri[fi, 0] + (s * (1 - r2))[:, None] * tri[fi, 1] + (s * r2)[:, None] * tri[fi, 2]
            self.pts = zup_m_to_rt(p)
            self.visible = np.zeros(n_points, bool)
            tree = cKDTree(self.pts)
            for pose in all_poses(level):
                R, T = sensor.RT(level.pano_rt[pose[0]], ELEVS[pose[3]], 360.0 * pose[4] / N_AZIM)
                fr = sensor.render(R, T)
                depth = sensor.torch.clamp(fr["zbuf"], min=0.5, max=750.0)
                pm = (fr["mask"].view(-1) & (depth.view(-1) < RANGE))
                obs = sensor.unproject(depth, fr["fov"])[0][pm].cpu().numpy()
                self.visible |= self._near(tree, obs)
            if cache:
                os.makedirs(os.path.dirname(cp), exist_ok=True)
                np.savez(cp, pts=self.pts, visible=self.visible, eps_rt=self.eps_rt)
        from scipy.spatial import cKDTree as _T
        self.tree = _T(self.pts)
        self.covered = np.zeros(len(self.pts), bool)

    def _near(self, tree, obs):
        from scipy.spatial import cKDTree
        if len(obs) == 0:
            return np.zeros(len(self.pts), bool)
        d, _ = cKDTree(obs).query(self.pts, k=1, distance_upper_bound=self.eps_rt, workers=-1)
        return np.isfinite(d)

    def update(self, sample_points):
        self.covered |= self._near(self.tree, np.asarray(sample_points, np.float64))
        return self.value()

    def value(self):
        return float((self.covered & self.visible).sum() / max(1, self.visible.sum()))


def write_scene_card(level):
    """Planner-side scene metadata (the indoor analogue of settings.json: box, k, panorama centres, graph);
    no GT geometry. W/preprocessing/scenes/<level key>.json"""
    card = {**level.describe(), "scan": level.scan, "level": level.level,
            "pano_idx": [p["idx"] for p in level.panos], "pano_rt": level.pano_rt.tolist(),
            "graph": {str(k): v for k, v in level.graph.items()}, "elevs": list(ELEVS), "n_azim": N_AZIM,
            "s_rt": S_RT, "n_win": N_WIN}
    p = os.path.join(W, "preprocessing", "scenes", f"{level.key}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(card, open(p, "w"), indent=1, default=float)
    return p


MACARONS = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"


def macaronspp_zup(scene):
    """Macarons++ mesh geometry in z-up OBJ units ((x, y, z)_y-up -> (x, -z, y)), cached in W like MP3D meshes."""
    cp = os.path.join(W, "preprocessing", "mesh_cache", f"OUT_{scene}.npz")
    if os.path.exists(cp):
        z = np.load(cp)
        return z["v"], z["f"]
    import trimesh
    obj = [f for f in os.listdir(os.path.join(MACARONS, scene)) if f.endswith(".obj")]
    assert len(obj) == 1, obj
    m = trimesh.load(os.path.join(MACARONS, scene, obj[0]), process=False, force="mesh", skip_materials=True)
    v = np.asarray(m.vertices, np.float64)
    v = np.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)
    f = np.asarray(m.faces, np.int64)
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    np.savez(cp, v=v, f=f)
    return v, f
