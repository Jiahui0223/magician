"""EXP5 Task 0 / T0.10 -- MAGICIAN on the Task 0 station-graph action space (planner side; reads only the scene card:
level box, k, panorama centres; never the GT-derived graph, the mesh or any evaluation file). Rules frozen in
magician_comparison protocol v1.2 §11:
  Settings : scene box = card box (runtime units / 10, official settings convention); grid_w = 1 (vertical, as the
             official mp3d settings); grid_l / grid_h = max(1, round(extent_m / 1.40)) (1.40 m = median horizontal cell
             of the 5 official mp3d scenes, manifests/official_mp3d_scale.json); cell_capacity 1000, cell_resolution
             0.05, visibility_ratio 1 (official mp3d values); a 1 x 1 x 1 x 3 x 8 dummy camera grid (unused).
  pose key : (station, elev index, azim index); X = card panorama centre (runtime), V = (ELEVS[e], 45 a) degrees.
  layer 0  : the environment-legal list of the current service record (simulator: panorama-graph moves with the GT
             line-of-sight rule + in-place rotations) -- the counterpart of the official layer-0 GT segment test;
  layer >0 : rotations (elev +-1 within the 3 bins, azim +-1 mod 8) and moves keeping the orientation to every station
             within 3.0 m (card centres only, no line of sight); the core then applies the official occupancy-point
             segment test. Order: lexicographic in the key (as the official torch.unique of the grid neighbours).
  validity : official get_valid_neighbors = keep the poses NOT visited by the real trajectory; if none, all of them.
"""
import json
import os

import numpy as np
import torch

S_RT, EDGE_MAX_M, CELL_M = 7.0, 3.0, 1.40
ELEVS, N_AZIM = (-30.0, 0.0, 30.0), 8


def zup_m_to_rt(p):
    p = np.asarray(p, np.float64)
    return S_RT * np.stack([p[..., 0], p[..., 2], -p[..., 1]], -1)


def settings_dict(card):
    lo, hi = np.asarray(card["box_m"][0]), np.asarray(card["box_m"][1])
    c = zup_m_to_rt(np.stack([lo, hi]))
    x_min, x_max = c.min(0) / 10.0, c.max(0) / 10.0                        # official settings unit = runtime / 10
    ext_m = hi - lo
    grid = [max(1, int(round(ext_m[0] / CELL_M))), 1, max(1, int(round(ext_m[1] / CELL_M)))]
    return {"scene": {"x_min": x_min.tolist(), "x_max": x_max.tolist(), "grid_l": grid[0], "grid_w": grid[1], "grid_h": grid[2],
                      "cell_capacity": 1000, "cell_resolution": 0.05, "visibility_ratio": 1.0},
            "camera": {"x_min": x_min.tolist(), "x_max": x_max.tolist(), "pose_l": 1, "pose_w": 1, "pose_h": 1,
                       "pose_n_theta": 3, "pose_n_azim": N_AZIM, "start_positions": [[0, 0, 0, 1, 0]], "contrast_factor": 1.0}}


class StationNeighbours:
    def __init__(self, card, device):
        self.dev = device
        self.P = np.asarray(card["pano_rt"], np.float64)
        d = np.linalg.norm(self.P[:, None] - self.P[None], axis=-1) / S_RT
        self.near = [sorted(int(j) for j in np.flatnonzero((d[i] <= EDGE_MAX_M) & (np.arange(len(self.P)) != i))) for i in range(len(self.P))]
        self.visited = set()
        self.legal = []

    def XV(self, key):
        s, e, a = key
        X = torch.tensor(self.P[s].reshape(1, 3), dtype=torch.float32, device=self.dev)
        V = torch.tensor([[ELEVS[e], 360.0 * a / N_AZIM]], dtype=torch.float32, device=self.dev)
        return X, V

    def set_legal(self, legal_idx):
        self.legal = [(int(r[0]), int(r[3]), int(r[4])) for r in legal_idx]

    def _valid(self, keys):
        new = [k for k in keys if k not in self.visited]
        return new if new else keys

    def __call__(self, key, layer):
        if layer == 0:
            keys = list(self.legal)
        else:
            s, e, a = key
            ks = {(s, e2, a) for e2 in (e - 1, e + 1) if 0 <= e2 < len(ELEVS)}
            ks |= {(s, e, (a + 1) % N_AZIM), (s, e, (a - 1) % N_AZIM)}
            ks |= {(j, e, a) for j in self.near[s]}
            ks.discard(key)
            keys = sorted(ks)
        return [(k, *self.XV(k)) for k in self._valid(keys)]
