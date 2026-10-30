"""EXP5 Task 0 MAGICIAN comparison -- shared helpers (paths, episodes, frozen T0.6 records, scene cards, the frozen
sensor back-projection, GT vertices). Read-only on every T0.1-T0.7 output; writes only under W/magician_comparison.
Back-projection / rendering need PyTorch3D: run with the magician env (`env -u LD_PRELOAD .../envs/magician/bin/python`).
"""
import glob
import json
import os
import sys

import numpy as np

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
T0 = os.path.join(REPO, "experiments", "exp5_future_view", "task0")
sys.path.insert(0, os.path.join(T0, "scripts"))
import mp3d_sim as S  # noqa: E402  (frozen T0 simulator code)

W = S.W
MC = os.path.join(W, "magician_comparison")
GROUPS = ("ref", "frozen", "C", "V2K4", "V2K1")
NEW_GROUPS = ("MAG-G", "MAG-B", "V2K4-NV")          # T0.10 groups (protocol v1.2), same record / episode format
GRID_GROUPS = ("G-RAND", "G-V2K4", "G-V2K4-NV", "G-MAG-G", "G-MAG-B")    # T0.12 grid groups (protocol v1.3)
GRID_EPISODES = [("Z6MFQCViBuw", 0, 6), ("Z6MFQCViBuw", 0, 17), ("oLBMNvg9in8", 1, 36), ("oLBMNvg9in8", 1, 78),
                 ("QUCTc6BB5sX", 1, 34), ("QUCTc6BB5sX", 1, 135), ("2azQ1b91cZZ", 0, 60), ("2azQ1b91cZZ", 0, 179),
                 ("TbHJrupSAjP", 1, 112), ("TbHJrupSAjP", 1, 114)]
LABEL = {"ref": "随机参考", "frozen": "Frozen", "C": "T1-C", "V2K4": "V2K4", "V2K1": "V2K1", "MAG-G": "MAG-G", "MAG-B": "MAG-B",
         "V2K4-NV": "V2K4-NV"}


def grid_episodes():
    """[(tag, scan, level, start)] of the 10 T0.12 grid episodes."""
    return [(f"{s}_L{lv}_S{st}", s, lv, st) for s, lv, st in GRID_EPISODES]


def episodes():
    """[(tag, scan, level, start)] of the 22 frozen T0.6 episodes."""
    out = []
    for line in open(os.path.join(W, "manifests", "cl_episodes.txt")):
        if line.strip():
            scan, lv, st = line.split()
            out.append((f"{scan}_L{lv}_S{st}", scan, int(lv), int(st)))
    return out


def card(scan, level):
    return json.load(open(os.path.join(W, "preprocessing", "scenes", f"{scan}_L{level}.json")))


def svc_path(group, tag, t):
    if group == "ref":
        return os.path.join(W, "preprocessing", "traj", tag, f"svc_t{t}.npz")
    if group in NEW_GROUPS:
        return os.path.join(MC, "shared_protocol", "closed_loop", group, tag, "svc", f"svc_t{t}.npz")
    if group in GRID_GROUPS:
        return os.path.join(MC, "grid_protocol", "closed_loop", group, tag, "svc", f"svc_t{t}.npz")
    return os.path.join(W, "closed_loop", group, tag, "svc", f"svc_t{t}.npz")


def stored_curve(group, tag):
    """The stored 101-value C_obs curve, executed poses and per-step timing of a frozen T0.6 episode."""
    if group == "ref":
        tr = json.load(open(os.path.join(W, "preprocessing", "traj", tag, "traj.json")))
        return {"coverage": [r["cov"] for r in tr["rows"]], "executed": [r["pose"] for r in tr["rows"]],
                "decision_s": None, "wall_s": None}
    root = (os.path.join(MC, "shared_protocol", "closed_loop") if group in NEW_GROUPS else
            os.path.join(MC, "grid_protocol", "closed_loop") if group in GRID_GROUPS else os.path.join(W, "closed_loop"))
    e = json.load(open(os.path.join(root, group, tag, "episode.json")))
    return {"coverage": e["coverage"], "executed": e["executed"],
            "decision_s": [s["decision_s"] for s in e["steps"]], "wall_s": e["wall_s"]}


def regions(scan, crd):
    """Local panorama index -> .house region index (evaluation-side annotation)."""
    h = S.parse_house(scan)
    reg = {p["idx"]: p["region"] for p in h["panos"]}
    return [reg[i] for i in crd["pano_idx"]]


def gt_vertices(scan, crd):
    """Face-referenced vertices of the textured matterport_mesh inside the level box (z-up metres)."""
    v, f = S.mesh_zup(scan)
    lo, hi = np.asarray(crd["box_m"][0]), np.asarray(crd["box_m"][1])
    used = np.zeros(len(v), bool)
    used[f.ravel()] = True
    return v[used & np.all((v >= lo) & (v <= hi), 1)]


class BackProjector:
    """The frozen MAGICIAN sensor back-projection (mp3d_sim.Sensor.unproject, same NDC tables and camera)."""

    def __init__(self, device="cuda"):
        import torch
        self.torch = torch
        self.dev = torch.device(device)
        tri = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        self.sensor = S.Sensor(tri, np.array([[0, 1, 2]]), self.dev)      # dummy mesh: only the camera code is used

    def points(self, rec, all_pixels=True):
        """Runtime-frame points of one service record: all valid pixels (mask & z < RANGE), and the validity mask."""
        t = self.torch
        R = t.tensor(np.asarray(rec["R"], np.float32)[None], device=self.dev)
        T = t.tensor(np.asarray(rec["T"], np.float32)[None], device=self.dev)
        fov = self.sensor.cam(R, T)
        zb = t.tensor(np.asarray(rec["zbuf"], np.float32), device=self.dev).view(1, S.H_IMG, S.W_IMG, 1)
        depth = t.clamp(zb, min=0.5, max=750.0)
        m = t.tensor(np.asarray(rec["mask"]), device=self.dev).view(1, -1) & (depth.view(1, -1) < S.RANGE)
        allp = self.sensor.unproject(depth, fov)[0]
        return allp.cpu().numpy().astype(np.float64), m[0].cpu().numpy()
