"""EXP5 Task 1.2 -- in-memory training data of the T1.1 shards (W/data/views/<scene>/<traj>.npz) and the batch builder
shared by training, learning-curve evaluation and the smoke ladder (protocol §2.4-§3.5).

A batch is a list of states (trajectory, t, staleness Delta, view rows). Per state the encoder input is the coarse grid
of t - Delta; projection channels / targets always belong to t.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import model as M  # noqa: E402
from seen2scene.futureview import rays as RY  # noqa: E402

VIEW_KEYS = ("cls", "nseg", "b", "y", "tk", "dlo", "dhi", "lu", "cat")


def seed_of(key):
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


class ViewData:
    def __init__(self, items, root=None, keys=VIEW_KEYS, log=print):
        """items: [(scene, traj)]."""
        import t5_diagnostic as TD
        root = root or os.path.join(VF.WORK, "data", "views")
        self.trajs, self.coarse, self.scene_of = [], [], []
        cols = {k: [] for k in keys + ("t", "a", "R", "T", "jitter")}
        self.frames, self.state_rows, self.files = {}, {}, {}
        n0 = 0
        for ti, (s, tj) in enumerate(items):
            p = os.path.join(root, s, f"{tj}.npz")
            z = np.load(p)
            self.files[f"{s}/{tj}"] = VF.sha256_file(p)
            if s not in self.frames:
                fr = TD.frame_of(s)
                self.frames[s] = (np.asarray(fr.origin_z, np.float64), float(fr.k_M))
            assert abs(float(z["k_M"]) - self.frames[s][1]) < 1e-12
            self.trajs.append((s, tj))
            self.scene_of.append(s)
            self.coarse.append(torch.from_numpy(np.ascontiguousarray(z["coarse"])))
            for k in cols:
                cols[k].append(z[k])
            tt = z["t"].astype(np.int64)
            for t in np.unique(tt):
                self.state_rows[(ti, int(t))] = n0 + np.flatnonzero(tt == t)
            n0 += len(tt)
            if (ti + 1) % 20 == 0:
                log(f"  loaded {ti + 1}/{len(items)} trajectories, {n0} views")
        self.v = {k: np.concatenate(v) for k, v in cols.items()}
        self.v_traj = np.concatenate([np.full(len(c), i) for i, c in enumerate(cols["t"])])
        self.states = sorted(self.state_rows)
        self.n_views = n0

    # ------------------------------------------------------------------ sampling
    def state_weights(self, t_le=20, factor=2.0):
        w = np.asarray([factor if t <= t_le else 1.0 for _, t in self.states])
        return w / w.sum()

    def sample_batch(self, rng, n_views=64, n_states=8, max_cands=8, stale_p=0.5, stale_max=9, weights=None):
        out, n = [], 0
        while n < n_views:
            ti, t = self.states[rng.choice(len(self.states), p=weights)]
            rows = self.state_rows[(ti, t)]
            k = min(len(rows), max_cands, n_views - n)
            rows = np.sort(rng.choice(rows, k, replace=False))
            dlt = int(rng.integers(1, min(stale_max, t) + 1)) if (t > 0 and rng.random() < stale_p) else 0
            out.append((ti, t, dlt, rows))
            n += k
        return out

    def fixed_states(self, key, n_views, per_scene=False, max_cands=8):
        """Deterministic evaluation subset: whole states (<= max_cands views each, first rows) until n_views."""
        rng = np.random.default_rng(seed_of(key))
        if per_scene:
            scenes = sorted(set(self.scene_of))
            quota = {s: n_views // len(scenes) + (1 if i < n_views % len(scenes) else 0) for i, s in enumerate(scenes)}
        else:
            quota = {None: n_views}
        out = []
        for sc, q in quota.items():
            cand = [x for x in self.states if sc is None or self.scene_of[x[0]] == sc]
            order = rng.permutation(len(cand))
            n = 0
            for i in order:
                if n >= q:
                    break
                ti, t = cand[i]
                rows = self.state_rows[(ti, t)][:min(max_cands, q - n)]
                out.append((ti, t, 0, rows))
                n += len(rows)
        return out


# ------------------------------------------------------------------ GPU batch
def state_tensors(data, ti, t, dlt, rows, dev):
    """Rays, channels, masks and targets of one state's views (float32, padded maps)."""
    s = data.scene_of[ti]
    origin, k_M = data.frames[s]
    v = data.v
    R = torch.as_tensor(v["R"][rows], dtype=torch.float64, device=dev)
    T = torch.as_tensor(v["T"][rows], dtype=torch.float64, device=dev)
    J = torch.as_tensor(v["jitter"][rows], dtype=torch.float64, device=dev)
    o, d, tcap, _ = RY.model_rays_torch(R, T, J, torch.as_tensor(origin, device=dev), k_M)
    o, d, tcap = o.float(), d.float(), tcap.float()
    g = {k: torch.as_tensor(v[k][rows], device=dev) for k in VIEW_KEYS}
    st16 = torch.stack([g["tk"], g["dlo"], g["dhi"], g["lu"]], 1)
    ch = G.channels_from_stats(g["cls"], st16, g["nseg"], d[..., 2], o[:, 2], tcap)
    m_unk = g["cls"] == G.CLS_UNKNOWN
    b = g["b"].float()
    lmask = m_unk & (g["b"] != 0)
    x1 = torch.stack([torch.where(g["b"] == 0, torch.full_like(b, -1.0), b), g["y"].float()], 1)
    cin = M.encoder_input(data.coarse[ti][t - dlt].to(dev))
    return {"o": o, "d": d, "tcap": tcap, "ch": ch, "m_unk": M.pad_rays(m_unk), "lmask": M.pad_rays(lmask),
            "x1": M.pad_rays(x1), "cat": g["cat"], "lu_vox": g["lu"].float() * tcap / G.VS, "cin": cin}


def build_cond(model, data, batch, dev, autocast=True):
    """Encoder per state, ray context per state; returns cond [V, C, 64, 128] and the stacked targets."""
    conds, keep = [], {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
    for ti, t, dlt, rows in batch:
        st = state_tensors(data, ti, t, dlt, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = model.encode(st["cin"])
            conds.append(model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"]))
        for k in keep:
            keep[k].append(st[k])
    out = {k: torch.cat(v) for k, v in keep.items()}
    out["cond"] = torch.cat(conds)
    return out


def fm_sum(model, bt, eps, tau, autocast=True):
    """Sum of squared velocity errors over M_unknown rays with a valid label, and the number of such rays."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"])
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum(), lm.sum()


def fm_loss(model, bt, eps, tau, autocast=True):
    """Rectified-flow velocity MSE on M_unknown rays with a valid label (protocol §3.1)."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"])
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum() / (2.0 * lm.sum().clamp(min=1.0)), lm.sum()


def per_view_loss(model, bt, eps, tau, autocast=True):
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"])
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum((1, 2)), 2.0 * lm.sum((1, 2))


def load_split(train_set):
    """Trajectory lists of a training-set name (protocol §2.5)."""
    sp = VF.frozen_values()["split"]
    train, held = [], []
    scenes = list(sp["scenes"])
    if train_set == "train5_nopisa_minus_Sc":
        scenes = [s for s in scenes if s != sp["loso"]["scene"]]
    elif train_set == "train3_nopisa_minus_Sc":                 # conditional M-SCALE3 (frozen_values["scale3"])
        scenes = [s for s in scenes if s in VF.frozen_values()["scale3"]["scenes"]]
    else:
        assert train_set == "train6_minus_Sc", train_set
    for s in scenes:
        train += [(s, x) for x in sp["scenes"][s]["train_trajs"]]
    for s in sp["scenes"]:
        held += [(s, x) for x in sp["scenes"][s]["heldout_trajs"]]
    return train, held
