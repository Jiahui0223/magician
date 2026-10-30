"""EXP5 Task 0 / T0.9 -- metrics of the official MAGICIAN runs (released-code MP3D protocol; protocol v1.1 §9 +
DEVIATIONS: the primary cloud comes from the frames the official tester saved). Magician env, one GPU, evaluation side.
  released-code metrics : coverage curve (151 values, / visibility_ratio = 1); official evaluation_lmdb.py AUC
                          = trapz(coverage[:100], x = 0..99) (raw) and its max; plus final coverage at 150 and
                          trapz(coverage[:101]) / 100 (the Task 0 convention)
  metric scale          : per scene uniform scale s (OBJ per metre) and translation from the bounding boxes of the
                          official mesh and our MP3D matterport_mesh, axes (x, y, z)_official <-> (x, z, +-y)_ours;
                          valid when the median NN distance of the transformed official vertices to our vertices < 1 cm
  Comp@5cm / distance   : GT = official mesh vertices inside the settings scene box (runtime units x 10); observed
                          cloud = all valid pixels (z < sensor_range) of the official frames 0..T via the official
                          Camera.compute_partial_point_cloud (gathering_factor 1); secondary = LMDB `points` (full_pc);
                          threshold 0.05 m = 0.5 s runtime units; T = 10, 20, 50, 100, 150;
                          distance exact up to 1 m, farther vertices via 1 cm voxel centres of the union (DEVIATIONS)
Output W/magician_comparison/official_reproduction/metrics/<run>.json
Usage: python t09_metrics.py --run formal_<scene>_rep<k>
"""
import argparse
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import magician_core as CORE  # noqa: E402

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

MP = CORE.MP
MC = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/magician_comparison"
RAW = "/bsuhome/jiahuizhang/scratch/data/mp3d_magician/mp3d_magician"
MPD = "/bsuhome/jiahuizhang/scratch/data/matterport3d/v1/scans"
CFG = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/exp5_future_view/task0/magician_comparison/configs/t09_official"


def obj_vertices(p):
    v, used, faces = [], None, []
    for line in open(p):
        if line.startswith("v "):
            v.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            faces.extend(int(t.split("/")[0]) - 1 for t in line.split()[1:4])
    v = np.asarray(v)
    used = np.zeros(len(v), bool)
    used[np.asarray(faces)] = True
    return v, used


def dist_to_cloud(gt, frames_pts, bound, vox):
    """Exact NN distance up to `bound`; vertices farther than `bound` from every point get the distance to the
    `vox`-voxel centres of the union (error <= 0.87 vox). frames_pts: list of (N_i, 3) arrays."""
    dmin = np.full(len(gt), np.inf)
    for p in frames_pts:
        if len(p):
            d, _ = cKDTree(p).query(gt, k=1, workers=8, distance_upper_bound=bound)
            dmin = np.minimum(dmin, d)
    far = ~np.isfinite(dmin)
    if far.any():
        keys = np.unique(np.concatenate([np.floor(p / vox).astype(np.int64) for p in frames_pts if len(p)]), axis=0)
        d2, _ = cKDTree((keys + 0.5) * vox).query(gt[far], k=1, workers=8)
        dmin[far] = d2
    return dmin, far


def metric_scale(scene, v_off):
    """(s, sign, median NN cm) or (None, ...) when the bounding-box transform does not validate."""
    v_our, _ = obj_vertices(glob.glob(os.path.join(MPD, scene, "matterport_mesh", "*", "*.obj"))[0])
    tree = cKDTree(v_our)
    best = None
    for sign in (1.0, -1.0):
        o = np.stack([v_our[:, 0], v_our[:, 2], sign * v_our[:, 1]], 1)       # ours (x, z, +-y) in the official axes
        s = float(np.median(np.ptp(v_off, 0) / np.ptp(o, 0)))
        q = (v_off - v_off.min(0)) / s + o.min(0)                               # official -> ours-in-official-axes, metres
        back = np.stack([q[:, 0], sign * q[:, 2], q[:, 1]], 1)                  # -> our z-up metres
        sel = np.random.default_rng(0).choice(len(back), min(20000, len(back)), replace=False)
        d, _ = tree.query(back[sel], k=1, workers=8)
        med = float(np.median(d) * 100)
        if best is None or med < best[2]:
            best = (s, sign, med)
    return (best if best[2] < 1.0 else (None, best[1], best[2]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    rd = os.path.join(MC, "official_reproduction", a.run)
    rec = json.load(open(os.path.join(rd, "run.json")))
    scene = rec["scene"]
    env = lmdb.open(os.path.join(rd, "lmdb"), readonly=True, lock=False)
    with env.begin() as txn:
        d = pickle.loads(txn.get(f"{scene}/0".encode()))
    cov = [float(x) for x in d["coverage"]]
    out = {"run": a.run, "scene": scene, "rep": rec["rep"], "n_coverage": len(cov), "coverage": cov,
           "official_auc_raw_first100": float(np.trapz(np.asarray(cov[:100]), np.arange(len(cov[:100])))),
           "official_max_first100": float(max(cov[:100])), "cov100": cov[100] if len(cov) > 100 else None,
           "cov150": cov[150] if len(cov) > 150 else None, "auc100_task0": float(np.trapz(np.asarray(cov[:101]), dx=1.0) / 100),
           "auc150_norm": float(np.trapz(np.asarray(cov[:151]), dx=1.0) / 150), "wall_s": rec["wall_s"], "peak_gpu_mem_gb": rec["peak_gpu_mem_gb"],
           "n_poses_executed": int(len(d["X_cam_history"]))}
    dev = torch.device("cuda:0")
    params = MP.load_params(os.path.join(CFG, "macarons_mp3d_formal.json"))
    st = json.load(open(os.path.join(RAW, scene, "settings.json")))
    settings = MP.Settings(st, dev, params.scene_scale_factor)
    v_off, used = obj_vertices(os.path.join(RAW, scene, f"{scene}.obj"))
    s, sign, med = metric_scale(scene, v_off)
    out["metric_scale"] = {"obj_per_m": s, "axis_sign": sign, "validation_median_nn_cm": med, "valid": s is not None}
    if s is not None:
        lo, hi = settings.scene.x_min.cpu().numpy() / params.scene_scale_factor, settings.scene.x_max.cpu().numpy() / params.scene_scale_factor
        gt = v_off[used & np.all((v_off >= lo) & (v_off <= hi), 1)] * params.scene_scale_factor       # runtime units
        thr = 0.05 * s * params.scene_scale_factor
        fov0 = MP.FoVPerspectiveCameras(device=dev)
        renderer = MP.get_rgb_renderer(image_height=params.image_height, image_width=params.image_width,
                                       ambient_light_intensity=params.ambient_light_intensity, cameras=fov0, device=dev, max_faces_per_bin=200000)
        sc = settings.camera
        cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=1, pose_w=1, pose_h=1, pose_n_elev=1, pose_n_azim=1,
                        n_interpolation_steps=1, zfar=params.zfar, renderer=renderer, device=dev, save_dir_path=None)
        fd = glob.glob(os.path.join(rd, "data", scene, "test_memory_0", "training", "0", "frames"))[0]
        bound, vox = 1.0 * s * params.scene_scale_factor, 0.01 * s * params.scene_scale_factor     # 1 m, 1 cm
        dmin, curve, keys = np.full(len(gt), np.inf), {}, []
        for i in range(151):
            fr = torch.load(os.path.join(fd, f"{i}.pt"), map_location=dev)
            depth, mask, err, R, T = MP.apply_perfect_depth_simple(fr, dev)
            fov = cam.get_fov_camera_from_RT(R_cam=R, T_cam=T)
            torch.manual_seed(0)
            pc, _ = cam.compute_partial_point_cloud(depth=depth, mask=(mask * err).bool(), images=fr["rgb"], fov_cameras=fov,
                                                    gathering_factor=1.0, fov_range=params.sensor_range)
            p = pc.cpu().numpy().astype(np.float64)
            if len(p):
                dd, _ = cKDTree(p).query(gt, k=1, workers=8, distance_upper_bound=bound)
                dmin = np.minimum(dmin, dd)
                keys.append(np.unique(np.floor(p / vox).astype(np.int64), axis=0))
            if i in (10, 20, 50, 100, 150):
                dist = dmin.copy()
                far = ~np.isfinite(dist)
                if far.any():
                    kk = np.unique(np.concatenate(keys), axis=0)
                    dist[far], _ = cKDTree((kk + 0.5) * vox).query(gt[far], k=1, workers=8)
                curve[i] = {"comp5": float((dmin <= thr).mean()), "dist_mean_cm": float(dist.mean() / (s * params.scene_scale_factor) * 100),
                            "dist_median_cm": float(np.median(dist) / (s * params.scene_scale_factor) * 100), "frac_far_1m": float(far.mean())}
        lp = np.asarray(d["points"], np.float64)
        dl, _ = dist_to_cloud(gt, [lp], bound, vox)
        allc = []
        for i in range(0, 151, 15):
            fr = torch.load(os.path.join(fd, f"{i}.pt"), map_location=dev)
            depth, mask, err, R, T = MP.apply_perfect_depth_simple(fr, dev)
            torch.manual_seed(0)
            pc, _ = cam.compute_partial_point_cloud(depth=depth, mask=(mask * err).bool(), images=fr["rgb"],
                                                    fov_cameras=cam.get_fov_camera_from_RT(R_cam=R, T_cam=T), gathering_factor=1.0,
                                                    fov_range=params.sensor_range)
            allc.append(pc.cpu().numpy())
        chk, _ = cKDTree(np.concatenate(allc)).query(lp[np.random.default_rng(0).choice(len(lp), min(20000, len(lp)), replace=False)], k=1, workers=8)
        out.update({"n_gt_vertices": int(len(gt)), "threshold_rt": thr, "comp_frames": curve,
                    "comp5_lmdb_points": float((dl <= thr).mean()), "dist_mean_cm_lmdb_points": float(dl.mean() / (s * params.scene_scale_factor) * 100),
                    "distance_rule": "exact up to 1 m; farther vertices: 1 cm voxel centres of the union (error <= 0.87 cm)",
                    "lmdb_points_on_frame_cloud_median_rt": float(np.median(chk)),
                    "lmdb_points_check_note": "frames sampled every 15 of 151, so points of other frames can sit farther away"})
    od = os.path.join(MC, "official_reproduction", "metrics")
    os.makedirs(od, exist_ok=True)
    json.dump(out, open(os.path.join(od, f"{a.run}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "coverage"}))


if __name__ == "__main__":
    main()
