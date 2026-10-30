"""EXP5 Task 2 / T2.1b -- MP3D training data for V2-Mix (seen2scene env) and the two-domain batch builder.

MP3DViewData mirrors task1/scripts/v2_data.ViewData (same per-view arrays, same sampling rules) for the shards of
t21_build.py, whose scene frame changes at every refresh (t_r = 0, 10, ...): a state (trajectory, t) uses the frame
(origin, k_M) and the coarse grids of block t // 10; staleness Delta is drawn as in V2 (p = stale_p, 1..stale_max) but
capped at t - t_r, so the encoder input never crosses a refresh (exactly the closed-loop situation).
build_cond(model, data, batch, dev) dispatches on the data type; v2_data.build_cond is patched to it, so the Task 1
evaluation functions (t12_train.eval_fm / eval_samples) work unchanged on both domains.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import v2_frozen as VF  # noqa: E402
import v2_data as VD  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402

REFRESH = 10


class ZCoarse:
    """Coarse grids [n, ...] kept zlib-compressed in RAM (lossless; ~66x for the MP3D grids); item i -> torch uint8."""

    def __init__(self, arr):
        import zlib
        arr = np.ascontiguousarray(arr)
        self.shape, self.dtype = arr.shape[1:], arr.dtype
        self.z = [zlib.compress(a.tobytes(), 1) for a in arr]

    def __len__(self):
        return len(self.z)

    def __getitem__(self, i):
        import zlib
        return torch.from_numpy(np.frombuffer(zlib.decompress(self.z[i]), dtype=self.dtype).reshape(self.shape).copy())


class CompactViewData(VD.ViewData):
    """v2_data.ViewData with the coarse grids compressed in RAM (identical tensors are returned)."""

    def __init__(self, items, root=None, log=print):
        super().__init__(items, root=root, log=log)
        self.coarse = [ZCoarse(c.numpy()) for c in self.coarse]


def mp3d_items(purpose):
    jobs = json.load(open(os.path.join(C.W2, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    return [(j["key"], j["traj"]) for j in jobs if j["purpose"] == purpose
            and os.path.exists(os.path.join(C.W2, "mp3d", "views", j["key"], f"{j['traj']}.npz"))]


class MP3DViewData:
    domain = "indoor"

    def __init__(self, items, log=print):
        self.trajs, self.coarse, self.origin, self.k_M, self.scene_of = [], [], [], [], []
        cols = {k: [] for k in VD.VIEW_KEYS + ("t", "a", "R", "T", "jitter")}
        self.state_rows, self.files = {}, {}
        n0 = 0
        for ti, (key, tj) in enumerate(items):
            p = os.path.join(C.W2, "mp3d", "views", key, f"{tj}.npz")
            z = np.load(p)
            self.files[f"mp3d/{key}/{tj}"] = VF.sha256_file(p)
            nb = int(z["n_blocks"])
            self.coarse.append([ZCoarse(z[f"coarse_b{b}"]) for b in range(nb)])
            self.origin.append([np.asarray(z[f"origin_b{b}"], np.float64) for b in range(nb)])
            self.k_M.append(float(z["k_M"]))
            self.trajs.append((key, tj))
            self.scene_of.append(key)
            for k in cols:
                cols[k].append(z[k])
            tt = z["t"].astype(np.int64)
            for t in np.unique(tt):
                self.state_rows[(ti, int(t))] = n0 + np.flatnonzero(tt == t)
            n0 += len(tt)
            if (ti + 1) % 40 == 0:
                log(f"  loaded {ti + 1}/{len(items)} MP3D trajectories, {n0} views")
        self.v = {k: np.concatenate(v) for k, v in cols.items()}
        self.states = sorted(self.state_rows)
        self.n_views = n0

    def state_weights(self, t_le=20, factor=2.0):
        w = np.asarray([factor if t <= t_le else 1.0 for _, t in self.states])
        return w / w.sum()

    def sample_batch(self, rng, n_views=32, n_states=4, max_cands=8, stale_p=0.5, stale_max=9, weights=None):
        out, n = [], 0
        while n < n_views:
            ti, t = self.states[rng.choice(len(self.states), p=weights)]
            rows = self.state_rows[(ti, t)]
            k = min(len(rows), max_cands, n_views - n)
            rows = np.sort(rng.choice(rows, k, replace=False))
            cap = min(stale_max, t - REFRESH * (t // REFRESH))
            dlt = int(rng.integers(1, cap + 1)) if (cap > 0 and rng.random() < stale_p) else 0
            out.append((ti, t, dlt, rows))
            n += k
        return out

    fixed_states = VD.ViewData.fixed_states

    def state_tensors(self, ti, t, dlt, rows, dev):
        b = t // REFRESH
        assert t - dlt >= REFRESH * b, (t, dlt)
        v = self.v
        R = torch.as_tensor(v["R"][rows], dtype=torch.float64, device=dev)
        T = torch.as_tensor(v["T"][rows], dtype=torch.float64, device=dev)
        J = torch.as_tensor(v["jitter"][rows], dtype=torch.float64, device=dev)
        o, d, tcap, _ = RY.model_rays_torch(R, T, J, torch.as_tensor(self.origin[ti][b], device=dev), self.k_M[ti])
        o, d, tcap = o.float(), d.float(), tcap.float()
        g = {k: torch.as_tensor(v[k][rows], device=dev) for k in VD.VIEW_KEYS}
        st16 = torch.stack([g["tk"], g["dlo"], g["dhi"], g["lu"]], 1)
        ch = G.channels_from_stats(g["cls"], st16, g["nseg"], d[..., 2], o[:, 2], tcap)
        m_unk = g["cls"] == G.CLS_UNKNOWN
        bb = g["b"].float()
        lmask = m_unk & (g["b"] != 0)
        x1 = torch.stack([torch.where(g["b"] == 0, torch.full_like(bb, -1.0), bb), g["y"].float()], 1)
        cin = M.encoder_input(self.coarse[ti][b][t - dlt - REFRESH * b].to(dev))
        return {"o": o, "d": d, "tcap": tcap, "ch": ch, "m_unk": M.pad_rays(m_unk), "lmask": M.pad_rays(lmask),
                "x1": M.pad_rays(x1), "cat": g["cat"], "lu_vox": g["lu"].float() * tcap / G.VS, "cin": cin}


_build0 = VD.build_cond


def build_cond(model, data, batch, dev, autocast=True):
    if not isinstance(data, MP3DViewData):
        return _build0(model, data, batch, dev, autocast)
    conds, keep = [], {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
    for ti, t, dlt, rows in batch:
        st = data.state_tensors(ti, t, dlt, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = model.encode(st["cin"])
            conds.append(model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"]))
        for k in keep:
            keep[k].append(st[k])
    out = {k: torch.cat(v) for k, v in keep.items()}
    out["cond"] = torch.cat(conds)
    return out


VD.build_cond = build_cond


def build_mixed(model, parts, dev):
    """parts: [(data, batch), ...] -> one concatenated conditioning / target dict."""
    outs = [build_cond(model, d, b, dev) for d, b in parts if b]
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}
