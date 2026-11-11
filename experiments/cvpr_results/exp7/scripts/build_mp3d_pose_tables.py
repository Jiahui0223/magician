"""EXP7 -- camera pose tables of the MP3D T0.12 grid for the D1 lookahead (magician env, 1 GPU; no GT access).
For every grid pose (l, w, h, i_theta, i_azim) of a level: R, T from the same macarons get_camera_RT the simulator's Sensor.RT
uses (on the same device, so the values are bitwise those of the service records), centre C = grid_env.position (runtime frame).
The grid is a pure function of the scene card (level box, floor, ceiling): experiments/exp5_future_view/task0/magician_comparison/
adapters/grid_env.py. Same format as work_dirs/exp6_v3/H/pose_tables/<scene>.npz (idx, R, T, C, grid json with the MAGICIAN
settings field names used by h_policy.neighbours). Output work_dirs/cvpr_results/exp7/pose_tables/<level key>.npz.
Usage: python build_mp3d_pose_tables.py KEY [KEY ...]   (KEY = <scan>_L<level>)"""
import itertools
import json
import os
import sys

import numpy as np
import torch

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters"))
sys.path.insert(0, "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69")
import grid_env as GE  # noqa: E402
from macarons.utility.macarons_utils import get_camera_RT  # noqa: E402

CARDS = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0", "preprocessing", "scenes")
OUT = os.path.join(REPO, "work_dirs", "cvpr_results", "exp7", "pose_tables")
dev = torch.device("cuda")
os.makedirs(OUT, exist_ok=True)
for key in sys.argv[1:]:
    card = json.load(open(os.path.join(CARDS, f"{key}.json")))
    g = GE.grid_def(card)
    dims = list(g["n"]) + [g["n_theta"], g["n_azim"]]
    idx = np.asarray(list(itertools.product(*[range(n) for n in dims])), np.int64)
    R, T, C = [], [], []
    for a in idx:
        X = GE.position(g, a)
        e, az = GE.angles(g, a)
        r, t = get_camera_RT(torch.tensor(np.asarray(X, np.float32).reshape(1, 3), device=dev), torch.tensor([[float(e), float(az)]], dtype=torch.float32, device=dev))
        R.append(r[0].cpu().numpy().astype(np.float64)); T.append(t[0].cpu().numpy().astype(np.float64)); C.append(np.asarray(X, np.float64))
    grid = {"pose_l": int(g["n"][0]), "pose_w": int(g["n"][1]), "pose_h": int(g["n"][2]), "pose_n_elev": int(g["n_theta"]), "pose_n_azim": int(g["n_azim"])}
    np.savez(os.path.join(OUT, f"{key}.npz"), idx=idx, R=np.stack(R), T=np.stack(T), C=np.stack(C), grid=np.asarray(json.dumps(grid)))
    print(key, "poses", len(idx), "grid", grid, flush=True)
