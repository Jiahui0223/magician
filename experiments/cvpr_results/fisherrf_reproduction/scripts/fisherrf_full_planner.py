"""FisherRF reproduction, Stage F1b -- FisherRF-full planner child (fisherrf env; planner side, no GT).

Same online map / scorer / inputs as fisherrf_planner.py (official GaussianSLAM track_rgbd every step, official pose_eval,
D1-D13 incl. D12 = docstring init, D13 = no-valid-depth skip, RGB side files, open() guard), plus the OFFICIAL FisherRF
active-mapping planning loop (FisherRF-active-mapping 8b196cd: tester.py NavTester.test_navigation / plan_best_path /
action_planning, planning/astar.py AstarPlanner) adapted from the 2D ground robot to the 3D Macarons++ pose grid:

  official (AM/...)                                         here
  ------------------------------------------------------    ---------------------------------------------------------------
  update_occ_map (astar.py:114-211): 2D 3-channel counts    OccMap.update: the same 3-channel accumulator (unknown prior 1,
    unknown / occupied x100 / free (lines to the camera),     occupied 100 h, free 1 on every ray voxel, per-voxel per-frame
    per-cell normalised, argmax label, height band            normalisation, argmax with ties -> unknown) on a 3D voxel grid,
                                                              no height band; voxel = explore.cell_size (0.05 FR = 0.35 rt)
  build_connected_freespace + build_frontiers               Frontier3D: free minus Gaussian-dense voxels (>25), 3x3x3 open,
    (astar.py:212-410): free - Gaussian cells, open,          largest 26-connected free component; frontier = dilate(free) -
    largest component, frontier = boundary & unknown,         free & unknown; 26-connected clusters > 10 voxels; "combined"
    clusters > 10, "combined" count / (mean dist + 20)        count / (mean voxel distance + 20); centering = cluster mean
  generate_candidate (astar.py:704-746): K = 256 on a       256 samples on the 3D shell r in [min_range, sample_range x
    ring r in [0.5, 1.0 x expansion] m around the centre,     expansion] (FR units) around the centre, snapped to the nearest
    yaw facing the centre; eroded free-space filter           settings-grid position, elevation / azimuth bins facing the
    (astar.py:584-594); no frontier -> random free poses      centre; same eroded-free filter rule; no frontier -> random
                                                              free grid poses (astar.py:470-526 analogue)
  pose_eval + top-20 (astar.py:615-662)                     official slam.pose_eval on the unique snapped poses, top 20
  action_planning (tester.py:895-1030): A* on the 2D map,   BFS shortest path over the settings-grid positions (official
    forward 0.05 m / turn 5 deg actions, <= 30 actions        get_neighboring_poses moves: +-1 along one axis), obstacles =
                                                              dilated occupied voxels + Gaussian-dense voxels (>50), unknown =
                                                              free; then elevation / azimuth bin turns to the goal view;
                                                              <= planning_queue_size actions
  plan_best_path (tester.py:763-893): <= 21 paths, H every  identical formula, official compute_Hessian / compute_H_train,
    acc_H_train_every, log point EIG with H_reg_lambda,       official config values (H_reg_lambda 1e-6, acc 5, weights 1.0 /
    + path_end_weight x final EIG (zip of the sorted EIGs)    0.2 / 30), including the official zip(EIGs) pairing
  main loop (tester.py:284-405): replan when the queue is   replan when the queue is empty or its next pose is not in the
    empty; no path -> 1 turn + expansion += 1; stuck ->       simulator's legal list; no path -> 1 azimuth turn + expansion +=
    mark obstacle, clear queue (tester.py:497-518)            1; illegal unvisited neighbours (GT segment test) -> blocked edges
Planner inputs: service record + RGB side file of the current step, the scene's settings.json (camera grid + scene box,
read once before the guard is installed; the same file every MAGICIAN planner reads). Nothing else.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fisherrf_planner as FP  # noqa: E402  (constants, Guard, seed_of; its main() is not run)
import numpy as np  # noqa: E402

DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"


class PoseGrid:
    """The official Macarons++ camera grid of settings.json (macarons_utils.Camera.__init__ pose formula, runtime units)."""

    def __init__(self, cam, np):
        self.np = np
        self.x_min, self.x_max = 10.0 * np.asarray(cam["x_min"], np.float64), 10.0 * np.asarray(cam["x_max"], np.float64)
        self.n = np.array([int(cam["pose_l"]), int(cam["pose_w"]), int(cam["pose_h"])])
        self.n_theta, self.n_azim = int(cam["pose_n_theta"]), int(cam["pose_n_azim"])
        self.step = (self.x_max - self.x_min) / self.n
        self.elev_bins = np.array([-90.0 + 180.0 * (1 + i) / (self.n_theta + 1) for i in range(self.n_theta)])
        self.azim_bins = np.array([360.0 * i / self.n_azim for i in range(self.n_azim)])

    def position(self, idx):
        return self.x_min + (0.5 + self.np.asarray(idx[:3], np.float64)) * self.step

    def RT(self, idx):
        """MAGICIAN get_camera_RT (look_at_view_transform, up = +y) in numpy: row-vector R, T of PyTorch3D."""
        np = self.np
        X = self.position(idx)
        e, a = math.radians(self.elev_bins[idx[3]]), math.radians(self.azim_bins[idx[4]])
        z = np.array([math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)])
        x = np.cross([0.0, 1.0, 0.0], z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        y = y / np.linalg.norm(y)
        R = np.stack([x, y, z], 0).T
        return R, -(R.T @ X)

    def neighbours(self, idx):
        """Official get_neighboring_poses (macarons_utils.py:3922-): +-1 along ONE axis or ONE angle, clamp / wrap, unique,
        the pose itself removed (it is not an action)."""
        out = set()
        lim = list(self.n) + [self.n_theta]
        for ax in range(5):
            for d in (-1, 1):
                q = list(idx)
                q[ax] += d
                for k in range(4):
                    q[k] = min(max(q[k], 0), lim[k] - 1)
                q[4] %= self.n_azim
                out.add(tuple(int(v) for v in q))
        out.discard(tuple(int(v) for v in idx))
        return sorted(out)

    def snap(self, P):
        np = self.np
        return np.clip(np.round((np.asarray(P) - self.x_min) / self.step - 0.5), 0, self.n - 1).astype(np.int64)

    def facing(self, p3, target):
        """Elevation / azimuth bins of the grid closest to the view direction from grid position p3 to target."""
        np = self.np
        d = np.asarray(target, np.float64) - self.position(p3)
        e = math.degrees(math.atan2(d[1], math.hypot(d[0], d[2])))
        a = math.degrees(math.atan2(d[0], d[2])) % 360.0
        it = int(np.argmin(np.abs(self.elev_bins - e)))
        da = np.abs((self.azim_bins - a + 180.0) % 360.0 - 180.0)
        return it, int(np.argmin(da))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--svc-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--resume-t", type=int, default=0)
    ap.add_argument("--scale", type=float, default=7.0)
    ap.add_argument("--map-iters", type=int, default=60)
    ap.add_argument("--init-pose", choices=["tester", "docstring"], default="docstring")
    ap.add_argument("--verify-every", type=int, default=0)        # accepted for CLI compatibility; unused here
    a = ap.parse_args()
    if a.resume_t:
        raise SystemExit("FisherRF-full: resume is not supported (episodes are re-run from t = 0, risk R1)")

    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    assert os.environ.get("SLURM_JOB_QOS") == "kd-high"
    sys.path.insert(0, FP.AM)
    sys.path.insert(0, FP.STUBS)
    import numpy as np
    import torch
    import torch.nn.functional as Fnn
    import open3d
    from scipy import ndimage
    assert os.path.abspath(open3d.__file__).startswith(FP.STUBS + "/")
    import importlib.util
    spec = importlib.util.spec_from_file_location("m_common", FP.M_COMMON)
    MC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MC)
    from configs.base_config import get_cfg_defaults
    import models.SLAM.gaussian as G

    settings = json.load(open(os.path.join(DATA, a.scene, "settings.json")))   # before the guard (camera grid, scene box)
    grid = PoseGrid(settings["camera"], np)

    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    s_unit = float(a.scale)
    tag = f"{a.scene}_s{a.start}"
    seed = FP.seed_of(f"fisherrf|{a.scene}|{a.start}")
    np.random.seed(seed % 2 ** 32)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(FP.AM_YAML)
    over = {"SLAM.Dataset.Calibration.fx": FP.FX, "SLAM.Dataset.Calibration.fy": FP.FX, "SLAM.Dataset.Calibration.cx": FP.CX,
            "SLAM.Dataset.Calibration.cy": FP.CY, "SLAM.Dataset.Calibration.width": FP.W_IMG,
            "SLAM.Dataset.Calibration.height": FP.H_IMG, "map_every": 1, "keyframe_every": 1,
            "mapping.num_iters": int(a.map_iters), "workdir": os.path.abspath(a.workdir), "run_name": tag,
            "use_wandb": False, "report_iter_progress": False}
    for k, v in over.items():
        node = cfg
        *path, leaf = k.split(".")
        for p in path:
            node = node[p]
        assert leaf in node, k
        node[leaf] = v
    slam = G.GaussianSLAM(cfg)
    assert not slam.sm.on_cluster
    # official planning constants (eccv yaml + base_config)
    OFF = {"sample_view_num": int(cfg.explore.sample_view_num), "sample_range": float(cfg.explore.sample_range),
           "min_range": float(cfg.explore.min_range), "cell_size": float(cfg.explore.cell_size),
           "centering": bool(cfg.explore.centering), "frontier_select_method": cfg.explore.frontier_select_method,
           "planning_queue_size": int(cfg.policy.planning_queue_size), "H_reg_lambda": float(cfg.H_reg_lambda),
           "acc_H_train_every": int(cfg.acc_H_train_every), "path_point_weight": float(cfg.path_point_weight),
           "path_pose_weight": float(cfg.path_pose_weight), "path_end_weight": float(cfg.path_end_weight),
           "pcd_far_distance": float(cfg.policy.pcd_far_distance), "topk": 20, "max_paths": 21,
           "frontier_min_area": 10, "free_gauss_thresh": 25, "obst_gauss_thresh": 50, "erode_kernel": 10}
    assert OFF["frontier_select_method"] == "combined" and OFF["centering"]

    TIM = {}

    def timed(name, fn):
        def w(*x, **y):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            r = fn(*x, **y)
            torch.cuda.synchronize()
            TIM[name] = TIM.get(name, 0.0) + time.perf_counter() - t0
            return r
        return w
    slam.compute_H_train = timed("h_train_s", slam.compute_H_train)
    slam.visualize_frame = timed("map_png_s", slam.visualize_frame)

    def to_w2c(R, T):
        R = np.asarray(R, np.float64).reshape(3, 3)
        T = np.asarray(T, np.float64).reshape(3)
        Fm = np.diag(FP.F_FLIP)
        w = np.eye(4)
        w[:3, :3] = Fm @ R.T
        w[:3, 3] = Fm @ T / s_unit
        return w

    def c2w_of(idx):
        R, T = grid.RT(idx)
        return torch.from_numpy(np.linalg.inv(to_w2c(R, T))).float().to(dev)

    # ------------------------------------------------------------------------------------------- planner occupancy (3D)
    sc = settings["scene"]
    lo = np.minimum(10.0 * np.asarray(sc["x_min"], np.float64), grid.x_min) - 1e-3
    hi = np.maximum(10.0 * np.asarray(sc["x_max"], np.float64), grid.x_max) + 1e-3
    VOX = OFF["cell_size"] * s_unit                              # 0.05 FR = 0.35 runtime
    DIMS = tuple(int(math.ceil((h - l) / VOX)) for l, h in zip(lo, hi))
    LO_T = torch.tensor(lo, dtype=torch.float32, device=dev)
    DIMS_T = torch.tensor(DIMS, device=dev)
    acc = torch.zeros((3,) + DIMS, device=dev)
    acc[0] = 1.0
    H_IMG, W_IMG = FP.H_IMG, FP.W_IMG
    vv, uu = torch.meshgrid(torch.arange(H_IMG, device=dev).float(), torch.arange(W_IMG, device=dev).float(), indexing="ij")
    RAY = torch.stack([(uu - FP.CX) / FP.FX, (vv - FP.CY) / FP.FX, torch.ones_like(uu)], -1)      # OpenCV camera rays

    def vox_index(P):
        q = torch.floor((P - LO_T) / VOX).long()
        inb = ((q >= 0) & (q < DIMS_T)).all(1)
        return q[inb], inb

    def flat(q):
        return (q[:, 0] * DIMS[1] + q[:, 1]) * DIMS[2] + q[:, 2]

    @torch.no_grad()
    def occ_update(depth_fr, c2w):
        """astar.update_occ_map in 3D: rays of all valid pixels (the 2D map's projection densifies the official 4x
        downsampled rays; in 3D every pixel ray is needed for the same density), free voxels along [0, 0.95] depth up to
        pcd_far_distance, occupied endpoint if depth < pcd_far_distance."""
        far = OFF["pcd_far_distance"]
        d = depth_fr
        m = d > 0
        if not bool(m.any()):
            return {"n_rays": 0}
        rays = RAY[m]
        dv = d[m]
        Rm, tv = c2w[:3, :3], c2w[:3, 3]
        hit = dv < far
        Pw = ((rays[hit] * dv[hit, None]) @ Rm.T + tv) * s_unit
        qh, _ = vox_index(Pw)
        hit_flat = flat(qh)
        n_tot = DIMS[0] * DIMS[1] * DIMS[2]
        hits = torch.bincount(hit_flat, minlength=n_tot).float() if len(hit_flat) else torch.zeros(n_tot, device=dev)
        line = torch.zeros(n_tot, dtype=torch.bool, device=dev)
        dmax = float(torch.clamp(dv, max=far).max()) * 0.95 * s_unit
        nk = int(min(max(math.ceil(dmax / (0.5 * VOX)) + 1, 11), 600))
        fr = torch.linspace(1e-3, 0.95, nk, device=dev)
        dirs_w = rays @ Rm.T
        for c in range(0, nk, 16):
            z = fr[c:c + 16, None] * dv[None, :]                            # (k, N)
            keep = z < far
            P = (dirs_w[None] * z[..., None])[keep] + tv
            qf, _ = vox_index(P * s_unit)
            line[flat(qf)] = True
        line |= hits > 0                                                      # the cv2.line includes its occupied end cell
        h = hits.view(DIMS)
        lv = line.view(DIMS)
        occ = 100.0 * h
        free = lv.float()
        tot = occ + free
        nz = tot > 0
        acc[1][nz] += occ[nz] / tot[nz]
        acc[2][nz] += free[nz] / tot[nz]
        cam = torch.floor((tv * s_unit - LO_T) / VOX).long()
        sl = [slice(max(int(cam[i]) - 1, 0), max(min(int(cam[i]) + 2, DIMS[i]), 0)) for i in range(3)]
        acc[2][sl[0], sl[1], sl[2]] = 1e3                                     # astar.py:125 (robot cell free)
        return {"n_rays": int(m.sum()), "n_hit_vox": int((hits > 0).sum()), "n_line_vox": int(line.sum()), "n_samples": nk}

    def gauss_counts():
        P = slam.params["means3D"].detach() * s_unit
        q, _ = vox_index(P)
        return torch.bincount(flat(q), minlength=DIMS[0] * DIMS[1] * DIMS[2]).view(DIMS) if len(q) else torch.zeros(DIMS, device=dev, dtype=torch.long)

    def dilate(x, k=3):
        return Fnn.max_pool3d(x[None, None].float(), k, 1, k // 2)[0, 0] > 0

    def erode(x, k=3):
        return ~dilate(~x, k)

    @torch.no_grad()
    def frontier3d(cam_rt, gc):
        lab = acc.argmax(0)
        free = lab == 2
        unknown = lab == 0
        if int(free.sum()) > 18:
            free = free & ~(gc > OFF["free_gauss_thresh"])
        free = dilate(erode(free))                                            # cv2.MORPH_OPEN 3x3 (astar.py:244-245)
        fl, nl = ndimage.label(free.cpu().numpy(), structure=np.ones((3, 3, 3)))
        info = {"n_free": int(free.sum()), "n_unknown": int(unknown.sum()), "n_occ": int((lab == 1).sum()), "n_free_cc": int(nl)}
        if nl == 0:
            return None, info, torch.zeros_like(free)
        cnt = np.bincount(fl.ravel())
        cnt[0] = 0
        free_cc = torch.from_numpy(fl == int(cnt.argmax())).to(dev)
        fr_mask = dilate(free_cc) & ~free_cc & unknown
        info["n_frontier_vox"] = int(fr_mask.sum())
        if info["n_frontier_vox"] == 0:
            return None, info, free_cc
        flab, nf = ndimage.label(fr_mask.cpu().numpy(), structure=np.ones((3, 3, 3)))
        idx = np.argwhere(flab > 0)
        labs = flab[idx[:, 0], idx[:, 1], idx[:, 2]]
        cnt = np.bincount(labs, minlength=nf + 1)
        cam_v = (np.asarray(cam_rt) - lo) / VOX
        dist = np.linalg.norm(idx + 0.5 - cam_v, axis=1)
        mean_d = np.bincount(labs, weights=dist, minlength=nf + 1) / np.maximum(cnt, 1)
        ok = (cnt > OFF["frontier_min_area"]) & (cnt >= 4)
        ok[0] = False
        info.update({"n_clusters": int(nf), "n_clusters_gt10": int(ok.sum())})
        if not ok.any():
            return None, info, free_cc
        score = np.where(ok, cnt / (mean_d + 20.0), 0.0)                    # "combined" (astar.py:366-389)
        best = int(score.argmax())
        if score[best] <= 0:
            return None, info, free_cc
        sel = idx[labs == best]
        center = lo + (sel.mean(0) + 0.5) * VOX
        info.update({"cluster_size": int(cnt[best]), "cluster_mean_dist_rt": float(mean_d[best] * VOX), "center_rt": center.tolist()})
        return center, info, free_cc

    def candidates(center, free_cc, expansion, rng):
        """generate_candidate + eroded-free filter (astar.py:556-594) on the settings grid; None centre -> random free poses."""
        K = OFF["sample_view_num"]
        er = erode(free_cc, OFF["erode_kernel"] + 1) if bool(free_cc.any()) else free_cc
        er_np = er.cpu().numpy()

        def in_er(p3):
            q = np.floor((grid.position(p3) - lo) / VOX).astype(int)
            return bool(np.all(q >= 0) and np.all(q < DIMS) and er_np[q[0], q[1], q[2]])
        # astar.py:584-594 filters candidates to eroded free space when it is non-trivial (> 40 cells) and the expansion
        # loop grows the ring until a candidate survives. On the settings grid the filter can only be met at grid
        # positions, so it is active iff at least one grid position lies in eroded free space (then the loop terminates).
        n_er_pos = sum(1 for i in range(grid.n[0]) for j in range(grid.n[1]) for k in range(grid.n[2]) if in_er((i, j, k)))
        use_filter = int(er.sum()) > 40 and n_er_pos > 0

        def ok(p3):
            return in_er(p3) if use_filter else True
        info = {"filter_eroded_free": use_filter, "n_grid_pos_in_eroded_free": n_er_pos, "n_sampled": 0, "attempts": 0}
        if center is None:                                                     # no frontier: random free poses
            fc = free_cc.cpu().numpy() if not use_filter else er_np
            allp = np.array([(i, j, k) for i in range(grid.n[0]) for j in range(grid.n[1]) for k in range(grid.n[2])])
            good = [p for p in allp if (lambda q: np.all(q >= 0) and np.all(q < DIMS) and fc[q[0], q[1], q[2]])(np.floor((grid.position(p) - lo) / VOX).astype(int))]
            if not good:
                return [], dict(info, mode="random_free", n_free_positions=0)
            pick = rng.choice(len(good), max(len(good) // 4, 1))
            out = sorted({(int(good[i][0]), int(good[i][1]), int(good[i][2]), int(rng.integers(grid.n_theta)), int(rng.integers(grid.n_azim))) for i in pick})
            return out, dict(info, mode="random_free", n_free_positions=len(good))
        out = set()
        exp = float(expansion)
        while not out and info["attempts"] < 30:
            info["attempts"] += 1
            r = (OFF["min_range"] + rng.random(K) * (OFF["sample_range"] * exp - OFF["min_range"])) * s_unit
            v = rng.normal(size=(K, 3))
            v /= np.linalg.norm(v, axis=1, keepdims=True)
            P = center[None] + r[:, None] * v
            info["n_sampled"] += K
            for p in P:
                p3 = tuple(int(x) for x in grid.snap(p))
                if ok(p3):
                    out.add(p3 + grid.facing(p3, center))
            exp *= 1.5
        return sorted(out), dict(info, mode="frontier", final_expansion=exp / 1.5)

    FRONTIER_MODE = os.environ.get("FRF_FRONTIER_MODE", "official")
    assert FRONTIER_MODE in ("official", "magician"), FRONTIER_MODE
    ALL_POSES = np.array([(i, j, k, e, z) for i in range(grid.n[0]) for j in range(grid.n[1]) for k in range(grid.n[2])
                          for e in range(grid.n_theta) for z in range(grid.n_azim)], dtype=np.int64)

    @torch.no_grad()
    def candidates_magician(rng):
        """F1c: MAGICIAN's 3D frontier selection (MAGICIAN e4e08fe macarons_utils.py:3751-3786
        sample_valid_poses_in_space_for_frontier, the design of frontier_planning.py:467-480): sample poses uniformly from
        the whole pose space (with replacement), keep those whose position is not occupied and whose field of view (within
        the sensor range) contains unknown cells, sorted by that count. Planner-side: the 'unknown' cells are the planner's
        own unknown voxels (a seeded subsample of <= 50k centres) instead of MAGICIAN's proxy points; no GT mesh test."""
        K = OFF["sample_view_num"]
        lab = acc.argmax(0)
        unk = torch.nonzero(lab == 0)
        if len(unk) > 50000:
            unk = unk[torch.from_numpy(rng.choice(len(unk), 50000, replace=False)).to(dev)]
        U = (unk.float() + 0.5) * VOX + LO_T                                  # runtime
        occ = (lab == 1).cpu().numpy()
        pick = ALL_POSES[rng.integers(len(ALL_POSES), size=K)]
        out = []
        for p in {tuple(int(x) for x in q) for q in pick}:
            q = np.floor((grid.position(p) - lo) / VOX).astype(int)
            if np.all(q >= 0) and np.all(q < DIMS) and occ[q[0], q[1], q[2]]:
                continue
            R, Tt = grid.RT(p)
            w2c = torch.from_numpy(to_w2c(R, Tt)).float().to(dev)
            Pc = (U / s_unit) @ w2c[:3, :3].T + w2c[:3, 3]
            z = Pc[:, 2]
            u = FP.FX * Pc[:, 0] / z.clamp(min=1e-6) + FP.CX
            v = FP.FX * Pc[:, 1] / z.clamp(min=1e-6) + FP.CY
            inf = (z > 0.5 / s_unit) & (z < FP.SENSOR_RANGE / s_unit) & (u >= 0) & (u < FP.W_IMG) & (v >= 0) & (v < FP.H_IMG)
            c = int(inf.sum())
            if c > 0:
                out.append((c, p))
        out.sort(key=lambda x: -x[0])
        return [p for _, p in out], {"mode": "magician_pose_space", "n_sampled": K, "n_unknown_used": int(len(U)),
                                     "n_kept": len(out), "max_unknown_in_fov": out[0][0] if out else 0}

    def paths_to(cur, goals, blocked, gc, visited, visited5):
        """BFS on the 5D pose graph (official moves), obstacles = dilated occupied + Gaussian-dense voxels; unknown = free."""
        lab = acc.argmax(0)
        obst = dilate((lab == 1) | (gc > OFF["obst_gauss_thresh"])).cpu().numpy()
        cur3 = tuple(cur[:3])
        # astar.py:125 / :290 -- the robot's own cells are free (3x3 around the robot, start cell cleared after the
        # dilation). 3D / grid analogue: the 3x3x3 cells of every executed camera position, cleared after the 3x3x3
        # dilation (5x5x5). Needed because the official get_pointcloud max-pool quirk puts zero-scale Gaussians at the
        # camera centre whenever a 4x4 block has an invalid top-left depth (sky), which would wall the camera in.
        for p3 in visited | {cur3}:
            q0 = np.floor((grid.position(p3) - lo) / VOX).astype(int)
            sl = tuple(slice(max(q0[i] - 2, 0), max(min(q0[i] + 3, DIMS[i]), 0)) for i in range(3))
            obst[sl] = False
        seg_cache = {}

        def vox_of(P):
            q = np.floor((P - lo) / VOX).astype(int)
            return q[np.all((q >= 0) & (q < np.array(DIMS)), axis=1)]

        def seg_blocked(u, w):
            key = (min(u, w), max(u, w))
            if key in blocked:
                return True
            if key not in seg_cache:
                A, B = grid.position(u), grid.position(w)
                n = int(math.ceil(np.linalg.norm(B - A) / (0.5 * VOX))) + 1
                q = vox_of(A[None] + np.linspace(0, 1, n)[:, None] * (B - A)[None])
                seg_cache[key] = bool(obst[q[:, 0], q[:, 1], q[:, 2]].any()) if len(q) else False
            return seg_cache[key]
        # v3: shortest paths on the 5D pose graph = the environment's action space (official get_neighboring_poses:
        # +-1 along one position axis OR one angle bin). Translations need a free segment (planner obstacles, learned
        # blocked edges); rotations are always legal. The environment never offers an executed pose while an unexecuted
        # neighbour exists (get_valid_neighbors), so executed poses are not traversable unless all neighbours of the
        # current pose are executed (then the environment offers them).
        cur5 = tuple(int(x) for x in cur)
        allow_visited = all(q in visited5 for q in grid.neighbours(cur5))
        prev = {cur5: None}
        frontier = [cur5]
        while frontier:
            nxt = []
            for u in frontier:
                for w in grid.neighbours(u):
                    if w in prev or (w in visited5 and not allow_visited):
                        continue
                    if w[:3] != u[:3] and seg_blocked(u[:3], w[:3]):
                        continue
                    prev[w] = u
                    nxt.append(w)
            frontier = nxt
        out = []
        for g in goals:
            g = tuple(int(x) for x in g)
            g3 = g[:3]
            qg = np.floor((grid.position(g3) - lo) / VOX).astype(int)
            if (np.all(qg >= 0) and np.all(qg < DIMS) and obst[qg[0], qg[1], qg[2]] and g3 != cur3) or g not in prev:
                out.append(None)                                               # astar.planning: goal occupied / unreachable
                continue
            acts = []
            x = g
            while x != cur5:
                acts.append(x)
                x = prev[x]
            acts.reverse()
            out.append(acts[:OFF["planning_queue_size"]])
        return out, {"n_reachable_poses": len(prev), "allow_visited": allow_visited, "n_seg_checked": len(seg_cache), "n_obst_vox": int(obst.sum())}

    def plan_best_path(cur, expansion, t, blocked, visited, visited5):
        """global_planning (frontier -> 256 candidates -> pose_eval -> top 20) + action_planning + plan_best_path."""
        T = {}
        rng = np.random.default_rng(FP.seed_of(f"fisherrf-full|{a.scene}|{a.start}|{t}|{expansion}"))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gc = gauss_counts()
        cam_rt = grid.position(cur)
        if FRONTIER_MODE == "official":
            center, finfo, free_cc = frontier3d(cam_rt, gc)
        else:                                                                  # F1c: MAGICIAN frontier selection
            finfo = {"mode": "magician"}
        torch.cuda.synchronize()
        T["frontier_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if FRONTIER_MODE == "official":
            cands, cinfo = candidates(center, free_cc, expansion, rng)
        else:
            cands, cinfo = candidates_magician(rng)
        torch.cuda.synchronize()
        T["candidates_s"] = time.perf_counter() - t0
        info = {"frontier": finfo, "cand": dict(cinfo, n_unique=len(cands))}
        if not cands:
            return None, info, T
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        TIM.pop("h_train_s", None)
        scores, _ = slam.pose_eval(torch.stack([c2w_of(c) for c in cands]))  # official EIG, lambda 0.1
        torch.cuda.synchronize()
        T["pose_eval_s"] = time.perf_counter() - t0
        T["pose_eval_h_train_s"] = TIM.pop("h_train_s", 0.0)
        sc_np = scores.numpy().astype(np.float64)
        order = np.argsort(-sc_np, kind="stable")[:OFF["topk"]]
        goals = [cands[i] for i in order]
        EIGs = sc_np[order]
        info["top_eig"] = [float(x) for x in EIGs]
        t0 = time.perf_counter()
        acts_all, pinfo = paths_to(cur, goals, blocked, gc, visited, visited5)
        T["paths_s"] = time.perf_counter() - t0
        valid, path_actions = [], []
        for g, acts in zip(goals, acts_all):                                   # tester.py:1026-1029 (dedupe identical)
            if acts and acts not in path_actions:                              # empty: official divides by len 0
                path_actions.append(acts)
                valid.append(g)
        info["paths"] = dict(pinfo, n_goals=len(goals), n_valid=len(valid), lengths=[len(x) for x in path_actions])
        if not path_actions:
            return None, info, T
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        H_train = slam.compute_H_train()
        best, best_eig, totals, n_hess = None, -1.0, [], 0
        for i, (goal, acts, final_EIG) in enumerate(zip(valid, path_actions, EIGs)):   # official zip with sorted EIGs
            if i > OFF["max_paths"] - 1:
                break
            Hp = H_train.clone()
            total, n = 0.0, 0
            for pose in acts:
                R, Tt = grid.RT(pose)
                cur_H, pose_H = slam.compute_Hessian(torch.from_numpy(to_w2c(R, Tt)).float().to(dev), return_points=True,
                                                     random_gaussian_params=None, return_pose=True)
                n_hess += 1
                point_EIG = torch.log(torch.sum(cur_H * torch.reciprocal(Hp + OFF["H_reg_lambda"])))
                pose_EIG = torch.log(torch.linalg.det(pose_H))
                n += 1
                total += OFF["path_pose_weight"] * pose_EIG.item()
                if (n + 1) % OFF["acc_H_train_every"] == 0:
                    total += OFF["path_point_weight"] * point_EIG.item()
                    Hp = Hp + cur_H
            total = total / n + OFF["path_end_weight"] * float(final_EIG)
            totals.append(float(total))
            if total > best_eig:
                best_eig, best = total, (goal, acts)
        torch.cuda.synchronize()
        T["path_eig_s"] = time.perf_counter() - t0
        T["path_h_train_s"] = TIM.pop("h_train_s", 0.0)
        info["path_eig"] = {"totals": totals, "n_hessians": n_hess, "best": best_eig}
        if best is None:
            return None, info, T
        info["best_goal"] = list(best[0])
        info["best_len"] = len(best[1])
        return best[1], info, T

    # ------------------------------------------------------------------------------------------------------ loop
    logf = open(a.log, "a")
    svc_dir = os.path.abspath(a.svc_dir)
    ST = {"queue": [], "expansion": 1, "blocked": set(), "n_replans": 0, "n_fallback": 0}
    rt_checked = {"max_err": 0.0}

    def map_stats():
        p = slam.params
        return {"n_gauss": int(p["means3D"].shape[0]), "n_keyframes": len(slam.keyframe_list)}

    @torch.no_grad()      # official calling context (tester.py:227)
    def step(t, svc_path):
        TIM.clear()
        torch.cuda.reset_peak_memory_stats(dev)
        torch.cuda.synchronize()
        t_all = time.perf_counter()
        assert os.path.dirname(os.path.abspath(svc_path)) == svc_dir, svc_path
        rec = MC.load_service_record(svc_path)
        if int(rec["t"]) != t or rec["scene_key"] != f"{a.scene}/{a.start}":
            raise RuntimeError(f"record mismatch at t={t}")
        z = np.load(svc_path[:-len(".npz")] + "_rgb.npz", allow_pickle=False)
        rgb = z["rgb"]
        if int(z["t"]) != t or not np.array_equal(z["R"], rec["R"]) or not np.array_equal(z["T"], rec["T"]):
            raise RuntimeError(f"RGB side file does not match the record at t={t}")
        zb, mk = rec["zbuf"].astype(np.float64), rec["mask"].astype(bool)
        valid = mk & (zb > 0) & (zb < FP.SENSOR_RANGE)
        depth = np.where(valid, zb / s_unit, 0.0).astype(np.float32)
        w2c = to_w2c(rec["R"], rec["T"])
        if float(np.abs(np.linalg.inv(w2c)[:3, 3] * s_unit - rec["cam_center"]).max()) > 1e-4:
            raise RuntimeError("pose self-check failed")
        cur = tuple(int(x) for x in rec["pose_idx"])
        legal = [tuple(int(x) for x in r) for r in rec["legal_idx"]]
        for j, L in enumerate(legal):                                          # numpy grid R, T == simulator R, T
            R, Tt = grid.RT(L)
            rt_checked["max_err"] = max(rt_checked["max_err"], float(np.abs(R - rec["legal_R"][j]).max()), float(np.abs(Tt - rec["legal_T"][j]).max()))
        if rt_checked["max_err"] > 1e-3:
            raise RuntimeError(f"grid R/T differ from the simulator's: {rt_checked['max_err']}")
        color_t = torch.from_numpy(rgb).to(dev)
        depth_t = torch.from_numpy(depth).to(dev).reshape(1, FP.H_IMG, FP.W_IMG)
        w2c_t = torch.from_numpy(w2c).float().to(dev)
        has_depth = bool(valid.any())
        row = {"t": t, "pose": list(cur), "n_legal": len(legal), "skipped_no_valid_depth": not has_depth}
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if has_depth and not slam.initialize:
            slam.init(color_t, depth_t, w2c_t if a.init_pose == "tester" else torch.linalg.inv(w2c_t))   # D10 / D12
        if has_depth:
            slam.track_rgbd(color_t, depth_t, w2c_t)
        torch.cuda.synchronize()
        T = {"map_update_s": time.perf_counter() - t0, "map_png_s": TIM.get("map_png_s", 0.0)}
        t0 = time.perf_counter()
        row["occ"] = occ_update(depth_t[0], torch.linalg.inv(w2c_t)) if has_depth else {"n_rays": 0}
        torch.cuda.synchronize()
        T["occ_update_s"] = time.perf_counter() - t0
        # simulator legality as observation (tester.py:497-518 analogue): an unvisited official neighbour missing from the
        # legal list failed the GT segment test (get_valid_neighbors keeps all unvisited ones when any exists)
        hist = {tuple(int(x) for x in r) for r in rec["executed_pose_idx_history"]}
        nb = grid.neighbours(cur)
        unv = [q for q in nb if q not in hist]
        ref = unv if unv else nb
        newly = 0
        for q in ref:
            if q not in legal and q[:3] != cur[:3]:
                key = (min(cur[:3], q[:3]), max(cur[:3], q[:3]))
                if key not in ST["blocked"]:
                    ST["blocked"].add(key)
                    newly += 1
        row["blocked_edges_new"], row["blocked_edges_total"] = newly, len(ST["blocked"])
        pick, fallback, replans = None, None, []
        legal_set = {q: j for j, q in enumerate(legal)}
        if not legal:
            fallback = "hover_empty_legal"
            ST["queue"] = []
        elif not slam.initialize:                                              # D13
            pick = int(np.random.default_rng(FP.seed_of(f"fisherrf|nomap|{a.scene}|{a.start}|{t}")).integers(len(legal)))
            fallback = "no_map_seeded_random"
            ST["queue"] = []
        else:
            if ST["queue"] and ST["queue"][0] in legal_set:
                pick = legal_set[ST["queue"].pop(0)]
            else:
                reason = "queue_empty" if not ST["queue"] else "next_not_legal"
                ST["queue"] = []
                ST["n_replans"] += 1
                best, info, PT = plan_best_path(cur, ST["expansion"], t, ST["blocked"], {q[:3] for q in hist}, hist)
                replans.append({"reason": reason, "expansion": ST["expansion"], "info": info, "timing": PT})
                for k2, v2 in PT.items():
                    T[k2] = T.get(k2, 0.0) + v2
                if best is None:                                               # tester.py:367-371: turn, expansion += 1
                    ST["expansion"] += 1
                    if ST["expansion"] > 10:                                   # official: NoFrontierError ends the episode
                        row["no_frontier_error"] = True
                        ST["expansion"] = 1
                    fallback = "no_best_path_turn"
                else:
                    ST["expansion"] = 1
                    ST["queue"] = list(best)
                    if ST["queue"][0] in legal_set:
                        pick = legal_set[ST["queue"].pop(0)]
                    else:
                        fallback = "first_move_not_legal_turn"
                        ST["queue"] = []
            if pick is None:                                                   # one azimuth turn (official action 2)
                ST["n_fallback"] += 1
                turn = cur[:4] + ((cur[4] + 1) % grid.n_azim,)
                turn2 = cur[:4] + ((cur[4] - 1) % grid.n_azim,)
                if turn in legal_set:
                    pick = legal_set[turn]
                elif turn2 in legal_set:
                    pick = legal_set[turn2]
                else:
                    pick = int(np.random.default_rng(FP.seed_of(f"fisherrf-full|turn|{a.scene}|{a.start}|{t}")).integers(len(legal)))
                    fallback = (fallback or "") + "+seeded_random"
        torch.cuda.synchronize()
        T["total_s"] = time.perf_counter() - t_all
        row.update({"pick": pick, "pick_pose": list(legal[pick]) if pick is not None else None, "fallback": fallback,
                    "replans": replans, "queue_len": len(ST["queue"]), "map": map_stats() if slam.initialize else {"n_gauss": 0},
                    "timing": T, "peak_alloc_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "rt_max_err": rt_checked["max_err"]})
        logf.write(json.dumps(row, default=float) + "\n")
        logf.flush()
        return row

    g = FP.Guard(svc_dir).install()
    hello = {"ready": True, "resume_t": 0,
             "info": {"method": "FisherRF-full (official FisherRF-active-mapping core + planning loop on the settings grid)",
                      "am_commit": FP.AM_COMMIT, "yaml": FP.AM_YAML, "overrides": over, "official_planning": OFF, "frontier_mode": FRONTIER_MODE,
                      "scale_s": s_unit, "init_pose": a.init_pose, "seed": seed, "voxel_rt": VOX, "voxel_dims": DIMS,
                      "voxel_lo": lo.tolist(), "grid_n": grid.n.tolist(), "n_theta": grid.n_theta, "n_azim": grid.n_azim,
                      "open3d": "stub " + open3d.__file__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(dev)}}
    proto.write(json.dumps(hello, default=float) + "\n")
    proto.flush()
    tot = 0.0
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("cmd") == "end" or q.get("bye"):
            break
        t = int(q["t"])
        try:
            r = step(t, q["svc"])
        except Exception as e:
            import traceback
            traceback.print_exc()
            proto.write(json.dumps({"t": t, "error": f"{type(e).__name__}: {e}"}) + "\n")
            proto.flush()
            continue
        tot += r["timing"]["total_s"]
        proto.write(json.dumps({"t": t, "pick": r["pick"], "fallback": r["fallback"], "decision_s": r["timing"]["total_s"],
                                "timing": dict(r["timing"], decision_s=r["timing"]["total_s"]), "scores": [],
                                "diag": {"replanned": bool(r["replans"]), "queue_len": r["queue_len"], "map": r["map"]},
                                "guard_denied": list(g.denied)}, default=float) + "\n")
        proto.flush()
    g.uninstall()
    proto.write(json.dumps({"bye": True, "decision_s_total": tot, "guard_denied": list(g.denied), "n_replans": ST["n_replans"],
                            "n_fallback": ST["n_fallback"], "peak_mem_gb_process": torch.cuda.max_memory_allocated(dev) / 2 ** 30}) + "\n")
    proto.flush()


if __name__ == "__main__":
    main()
