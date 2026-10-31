"""EXP6 V3 Task 2 / Experiment 2 -- R2 frontier map and frontier progress for the MP3D grid planner (planner side; numpy
only; reads NOTHING but the whitelisted service records and the scene card; protocol §4 R2, E2_PLANNING_AUDIT §5).

Grid: the (l, h) plane of the T0.12 camera grid (1 m cells; heights merged). Evidence per record:
  * the camera cell: visited, strong FREE evidence;
  * the 64 x 114 scoring rays whose world direction is near-horizontal (|dir_y| < sin 10 deg): sampled every half cell
    from the camera to the hit (depth < 70) or to the range; the cells passed get FREE evidence, the hit cell OCCUPIED
    evidence (hits beyond the range give none).
Cell state: FREE if free >= 2 and free > occ; OCC if occ > free (occ >= 1); else UNKNOWN. Frontier = UNKNOWN cell
4-adjacent to a FREE cell. Reachability / distances: 4-connected BFS over FREE cells (+ the current cell and the legal
candidate cells, which the environment has certified traversable). FrontierProgress(C) = dist(current -> nearest
reachable frontier) - dist(C's cell -> nearest reachable frontier) [+1 / 0 / -1 for a 1-cell move; 0 for rotations and
when no frontier is reachable; halved when C's cell was visited before].
"""
import json
import math
import os
import sys
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
EXP5 = os.path.join(os.path.dirname(E6), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task0", "magician_comparison", "adapters"))
sys.path.insert(0, os.path.join(EXP5, "task0", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(E6), "exp3b_macaronspp_transfer"))
import grid_env as GE  # noqa: E402
import m_common as MC  # noqa: E402
try:
    from seen2scene.futureview import rays as RY  # noqa: E402
except ModuleNotFoundError:                                   # magician env (evaluation side): import the numpy-only module by file
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(E6)), "seen2scene", "futureview"))
    import rays as RY  # noqa: E402

HORIZ_SIN = math.sin(math.radians(10.0))
RANGE = 70.0
FREE_MIN = 2


class FrontierMap:
    def __init__(self, card):
        self.g = GE.grid_def(card)
        self.n_l, self.n_w, self.n_h = self.g["n"]
        self.x_min, self.x_max = np.asarray(self.g["x_min"], np.float64), np.asarray(self.g["x_max"], np.float64)
        self.cell = (self.x_max - self.x_min) / np.asarray(self.g["n"], np.float64)       # runtime units per cell (l: x, w: y, h: z)
        self.free = np.zeros((self.n_l, self.n_h), np.int32)
        self.occ = np.zeros((self.n_l, self.n_h), np.int32)
        self.visited = np.zeros((self.n_l, self.n_h), bool)
        self.n_records = 0
        u, v = RY.grid_uv((0.0, 0.0))
        self.u, self.v = u, v
        self.pix = (np.round(v).astype(int).clip(0, MC.H - 1), np.round(u).astype(int).clip(0, MC.W - 1))

    @classmethod
    def from_env(cls):
        import mp3d_frame as F
        key = sys.argv[sys.argv.index("--card") + 1]
        return cls(F.load_card(key))

    # ------------------------------------------------------------------ cells
    def to_cell(self, p):
        """Runtime y-up point(s) [.., 3] -> (l, h) integer cells (clipped)."""
        p = np.asarray(p, np.float64)
        l = np.floor((p[..., 0] - self.x_min[0]) / self.cell[0]).astype(int).clip(0, self.n_l - 1)
        h = np.floor((p[..., 2] - self.x_min[2]) / self.cell[2]).astype(int).clip(0, self.n_h - 1)
        return l, h

    def inside(self, p):
        p = np.asarray(p, np.float64)
        return (p[..., 0] >= self.x_min[0]) & (p[..., 0] < self.x_max[0]) & (p[..., 2] >= self.x_min[2]) & (p[..., 2] < self.x_max[2])

    # ------------------------------------------------------------------ evidence
    def add_record(self, rec):
        c = np.asarray(rec["cam_center"], np.float64)
        l0, h0 = self.to_cell(c)
        self.visited[l0, h0] = True
        self.free[l0, h0] += 10
        d_y, dz, _ = MC.pixel_rays(np.asarray(rec["R"], np.float64), np.asarray(rec["T"], np.float64), self.u, self.v)
        horiz = np.abs(d_y[:, 1]) < HORIZ_SIN
        zb = np.asarray(rec["zbuf"], np.float64)[self.pix]
        mk = np.asarray(rec["mask"], bool)[self.pix]
        z = np.clip(zb, 0.5, 750.0)
        hit = mk & (z < RANGE) & horiz
        t_end = np.where(hit, z / dz, RANGE / dz)                      # runtime ray length to the hit / range
        step = 0.5 * min(self.cell[0], self.cell[2])
        for sel, is_hit in ((horiz & hit, True), (horiz & ~hit, False)):
            if not sel.any():
                continue
            d, te = d_y[sel], t_end[sel]
            n_max = int(np.ceil(te.max() / step)) + 1
            ts = np.arange(n_max) * step
            pts = c[None, None] + d[:, None] * ts[None, :, None]                   # [n, m, 3]
            ok = ts[None, :] < te[:, None] - (step if is_hit else 0.0)             # strictly before the hit cell
            ok &= self.inside(pts)
            if ok.any():
                l, h = self.to_cell(pts[ok])
                np.add.at(self.free, (l, h), 1)
            if is_hit:
                ph = c[None] + d * te[:, None]
                ins = self.inside(ph)
                if ins.any():
                    l, h = self.to_cell(ph[ins])
                    np.add.at(self.occ, (l, h), 1)
        self.n_records += 1

    # ------------------------------------------------------------------ state / frontier
    def states(self):
        free = (self.free >= FREE_MIN) & (self.free > self.occ)
        occ = (self.occ >= 1) & (self.occ >= self.free) & ~free
        unk = ~free & ~occ
        return free, occ, unk

    def frontier(self, free, unk):
        nb = np.zeros_like(free)
        nb[1:, :] |= free[:-1, :]; nb[:-1, :] |= free[1:, :]; nb[:, 1:] |= free[:, :-1]; nb[:, :-1] |= free[:, 1:]
        return unk & nb

    @staticmethod
    def bfs(passable, sources):
        """Multi-source 4-connected BFS over passable cells -> integer distances (-1 unreachable)."""
        n_l, n_h = passable.shape
        dist = -np.ones((n_l, n_h), np.int32)
        dq = deque()
        for (l, h) in sources:
            if 0 <= l < n_l and 0 <= h < n_h and dist[l, h] < 0:
                dist[l, h] = 0
                dq.append((l, h))
        while dq:
            l, h = dq.popleft()
            for dl, dh in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = l + dl, h + dh
                if 0 <= a < n_l and 0 <= b < n_h and dist[a, b] < 0 and passable[a, b]:
                    dist[a, b] = dist[l, h] + 1
                    dq.append((a, b))
        return dist

    def progress(self, rec, legal_R, legal_T):
        """FrontierProgress of every legal candidate of the record (legal_center = candidate camera centres)."""
        c = np.asarray(rec["cam_center"], np.float64)
        l0, h0 = (int(x) for x in self.to_cell(c))
        cands = np.asarray(rec["legal_center"], np.float64).reshape(-1, 3)
        cl, ch = self.to_cell(cands)
        free, occ, unk = self.states()
        front = self.frontier(free, unk)
        passable = free.copy()
        passable[l0, h0] = True
        passable[cl, ch] = True                                        # legal candidates are traversable by the environment rule
        # distance from every passable cell to the nearest frontier cell (frontier cells are entered as the last step)
        pf = passable | front
        dist_f = self.bfs(pf, list(zip(*np.nonzero(front))))
        d_cur = int(dist_f[l0, h0])
        reach = self.bfs(passable, [(l0, h0)])
        n_reach_front = int(sum(1 for (l, h) in zip(*np.nonzero(front)) if any(0 <= l + dl < self.n_l and 0 <= h + dh < self.n_h and reach[l + dl, h + dh] >= 0 for dl, dh in ((1, 0), (-1, 0), (0, 1), (0, -1)))))
        fp = []
        for j in range(len(cands)):
            if (cl[j], ch[j]) == (l0, h0) or d_cur < 0 or dist_f[cl[j], ch[j]] < 0:
                fp.append(0.0)
                continue
            v = float(d_cur - dist_f[cl[j], ch[j]])
            if self.visited[cl[j], ch[j]] and v > 0:
                v *= 0.5
            fp.append(v)
        info = {"n_free": int(free.sum()), "n_occ": int(occ.sum()), "n_unknown": int(unk.sum()), "n_frontier": int(front.sum()), "n_reachable_frontier": n_reach_front,
                "dist_to_frontier": d_cur, "cell": [l0, h0], "n_records": self.n_records}
        return fp, info

    def snapshot(self):
        free, occ, unk = self.states()
        return {"free": free.tolist(), "occ": occ.tolist(), "frontier": self.frontier(free, unk).tolist(), "visited": self.visited.tolist()}
