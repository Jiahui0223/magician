"""EXP5 Task 0 / T0.9 -- the LMDB-points check exactly as protocol v1.1 §9 states it: NN distance of the stored full_pc
points (20,000 sampled, seed 0) to the frame clouds of ALL 151 official frames (the first implementation in
t09_metrics.py compared with every 15th frame only, see DEVIATIONS). Bounded queries (0.05 runtime units); pass when
the median < 1e-3 OBJ units = 0.01 runtime units. Magician env. Output official_reproduction/metrics/lmdb_check.json
"""
import glob
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t09_metrics as T  # noqa: E402

import lmdb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

MP = T.MP


def main():
    dev = torch.device("cuda:0")
    params = MP.load_params(os.path.join(T.CFG, "macarons_mp3d_formal.json"))
    res = {}
    for p in sorted(glob.glob(os.path.join(T.MC, "official_reproduction", "metrics", "formal_*.json"))):
        r = json.load(open(p))
        if not r["metric_scale"]["valid"]:
            continue
        run, scene = r["run"], r["scene"]
        rd = os.path.join(T.MC, "official_reproduction", run)
        with lmdb.open(os.path.join(rd, "lmdb"), readonly=True, lock=False).begin() as txn:
            lp = np.asarray(pickle.loads(txn.get(f"{scene}/0".encode()))["points"], np.float64)
        q = lp[np.random.default_rng(0).choice(len(lp), 20000, replace=False)]
        st = json.load(open(os.path.join(T.RAW, scene, "settings.json")))
        settings = MP.Settings(st, dev, params.scene_scale_factor)
        sc = settings.camera
        renderer = MP.get_rgb_renderer(image_height=params.image_height, image_width=params.image_width,
                                       ambient_light_intensity=params.ambient_light_intensity, cameras=MP.FoVPerspectiveCameras(device=dev),
                                       device=dev, max_faces_per_bin=200000)
        cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=1, pose_w=1, pose_h=1, pose_n_elev=1, pose_n_azim=1, n_interpolation_steps=1,
                        zfar=params.zfar, renderer=renderer, device=dev, save_dir_path=None)
        fd = glob.glob(os.path.join(rd, "data", scene, "test_memory_0", "training", "0", "frames"))[0]
        dmin = np.full(len(q), np.inf)
        for i in range(151):
            fr = torch.load(os.path.join(fd, f"{i}.pt"), map_location=dev)
            depth, mask, err, R, Tt = MP.apply_perfect_depth_simple(fr, dev)
            torch.manual_seed(0)
            pc, _ = cam.compute_partial_point_cloud(depth=depth, mask=(mask * err).bool(), images=fr["rgb"],
                                                    fov_cameras=cam.get_fov_camera_from_RT(R_cam=R, T_cam=Tt), gathering_factor=1.0,
                                                    fov_range=params.sensor_range)
            if len(pc):
                d, _ = cKDTree(pc.cpu().numpy().astype(np.float64)).query(q, k=1, workers=8, distance_upper_bound=0.05)
                dmin = np.minimum(dmin, d)
        res[run] = {"median_rt": float(np.median(dmin)), "p95_rt": float(np.quantile(dmin, 0.95)),
                    "frac_within_1e-4_rt": float((dmin <= 1e-4).mean()), "pass": bool(np.median(dmin) < 0.01)}
        print(run, res[run], flush=True)
    json.dump(res, open(os.path.join(T.MC, "official_reproduction", "metrics", "lmdb_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
