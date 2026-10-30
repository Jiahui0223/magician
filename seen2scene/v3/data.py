"""Training / selection data of EXP6 V3: the E2 D100 view shards (ray channels + labels per candidate view, computed by
the frozen builders on the 100 % P1) and the 4x partial-TSDF cache (VAE input per state).

Sampling rules are those of V2 / E2 (task2 t22_data.MP3DViewData / task1 v2_data.ViewData): a batch is a list of
(trajectory index, t, staleness, view rows); the encoder input is the state t - staleness (never across an MP3D
refresh block), the ray channels and labels are those of state t. build_cond() runs the scene encoder once per state,
conditions its views and returns everything the losses need (plus the VAE terms per state).
"""
import json
import os

import numpy as np
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import geometry as G
from seen2scene.futureview import model as M
from seen2scene.futureview import rays as RY
from seen2scene.v3 import partial_tsdf as PT

REPO = str(PROJECT_ROOT)
E2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2")
T2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2")
CACHE = os.path.join(REPO, "work_dirs", "exp6_v3", "shared_cache", "tsdf4")
VIEW_KEYS = ("cls", "nseg", "b", "y", "tk", "dlo", "dhi", "lu", "cat")
REFRESH = 10


def seed_of(key):
    return RY.sha_seed(key) % (2 ** 32)


def shard_path(domain, scene, traj):
    if domain == "indoor":
        return os.path.join(E2W, "w2_d100", "mp3d", "views", scene, f"{traj}.npz")
    return os.path.join(E2W, "out_views_d100", scene, f"{traj}.npz")


def cache_path(domain, scene, traj):
    return os.path.join(CACHE, "mp3d" if domain == "indoor" else "outdoor", scene, f"{traj}.npz")


def items(domain, purpose):
    """(scene, traj) of a split: indoor train / valsel (E2 trajectory jobs), outdoor train (Train-6 minus S-c) / sc."""
    out = []
    if domain == "indoor":
        for j in json.load(open(os.path.join(T2W, "mp3d", "manifests", "traj_jobs.json")))["jobs"]:
            if j["purpose"] == purpose:
                out.append((j["key"], j["traj"]))
    else:
        root = os.path.join(E2W, "out_views_d100")
        for s in sorted(os.listdir(root)):
            for f in sorted(os.listdir(os.path.join(root, s))):
                if f.endswith(".json"):
                    held = bool(np.load(os.path.join(root, s, f[:-5] + ".npz"))["heldout"])
                    if (purpose == "sc") == held:
                        out.append((s, f[:-5]))
    return sorted(out)


class TSDFCache:
    """Compressed 4x grids of one trajectory kept in RAM; state(t) decompresses one grid."""

    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.blobs = {int(k[5:]): z[k].tobytes() for k in z.files if k.startswith("blob_")}
        self.blocks = self.meta["blocks"]
        self.refresh = REFRESH if self.meta["dataset"] == "mp3d" else 10 ** 9
        self.nbytes = sum(len(b) for b in self.blobs.values())

    def block(self, t):
        return self.blocks[t // self.refresh]

    def state(self, t, device):
        blk = self.block(t)
        return PT.unpack(PT.from_blob(self.blobs[t], tuple(blk["shape4"])), device)


class V3Views:
    """Views of the trajectories of one domain + their TSDF caches (everything in RAM)."""

    def __init__(self, domain, items_, log=print):
        self.domain, self.trajs, self.caches, self.scene_of, self.k_M = domain, [], [], [], []
        cols = {k: [] for k in VIEW_KEYS + ("t", "a", "R", "T", "jitter")}
        self.state_rows, self.files = {}, {}
        n0 = 0
        for ti, (scene, tj) in enumerate(items_):
            p, pc = shard_path(domain, scene, tj), cache_path(domain, scene, tj)
            z = np.load(p)
            self.files[f"{domain}/{scene}/{tj}"] = {"shard": p, "shard_bytes": os.path.getsize(p), "cache": pc, "cache_bytes": os.path.getsize(pc)}
            self.caches.append(TSDFCache(pc))
            self.k_M.append(float(z["k_M"]))
            assert abs(self.k_M[-1] - self.caches[-1].meta["k_M"]) < 1e-9 * self.k_M[-1], (scene, tj)
            self.trajs.append((scene, tj))
            self.scene_of.append(scene)
            for k in cols:
                cols[k].append(z[k])
            tt = z["t"].astype(np.int64)
            for t in np.unique(tt):
                self.state_rows[(ti, int(t))] = n0 + np.flatnonzero(tt == t)
            n0 += len(tt)
            if (ti + 1) % 40 == 0:
                log(f"  loaded {ti + 1}/{len(items_)} {domain} trajectories, {n0} views")
        self.v = {k: np.concatenate(v) for k, v in cols.items()}
        self.states = sorted(self.state_rows)
        self.n_views = n0
        self.cache_bytes = sum(c.nbytes for c in self.caches)

    def state_weights(self, t_le=20, factor=2.0):
        w = np.asarray([factor if t <= t_le else 1.0 for _, t in self.states])
        return w / w.sum()

    def stale_cap(self, t, stale_max):
        return min(stale_max, t - REFRESH * (t // REFRESH)) if self.domain == "indoor" else min(stale_max, t)

    def sample_batch(self, rng, n_views=32, n_states=4, max_cands=8, stale_p=0.5, stale_max=9, weights=None):
        out, n = [], 0
        while n < n_views:
            ti, t = self.states[rng.choice(len(self.states), p=weights)]
            rows = self.state_rows[(ti, t)]
            k = min(len(rows), max_cands, n_views - n)
            rows = np.sort(rng.choice(rows, k, replace=False))
            cap = self.stale_cap(t, stale_max)
            dlt = int(rng.integers(1, cap + 1)) if (cap > 0 and rng.random() < stale_p) else 0
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

    def state_tensors(self, ti, t, dlt, rows, dev):
        """Rays, channels, masks, targets of the views (state t) and the VAE input grid (state t - dlt)."""
        cache = self.caches[ti]
        assert (t - dlt) // cache.refresh == t // cache.refresh, (t, dlt)
        origin = torch.as_tensor(cache.block(t)["origin"], dtype=torch.float64, device=dev)
        v = self.v
        R = torch.as_tensor(v["R"][rows], dtype=torch.float64, device=dev)
        T = torch.as_tensor(v["T"][rows], dtype=torch.float64, device=dev)
        J = torch.as_tensor(v["jitter"][rows], dtype=torch.float64, device=dev)
        o, d, tcap, _ = RY.model_rays_torch(R, T, J, origin, self.k_M[ti])
        o, d, tcap = o.float(), d.float(), tcap.float()
        g = {k: torch.as_tensor(v[k][rows], device=dev) for k in VIEW_KEYS}
        st16 = torch.stack([g["tk"], g["dlo"], g["dhi"], g["lu"]], 1)
        ch = G.channels_from_stats(g["cls"], st16, g["nseg"], d[..., 2], o[:, 2], tcap)
        m_unk = g["cls"] == G.CLS_UNKNOWN
        bb = g["b"].float()
        lmask = m_unk & (g["b"] != 0)
        x1 = torch.stack([torch.where(g["b"] == 0, torch.full_like(bb, -1.0), bb), g["y"].float()], 1)
        x4, valid4 = cache.state(t - dlt, dev)
        return {"o": o, "d": d, "tcap": tcap, "ch": ch, "m_unk": M.pad_rays(m_unk), "lmask": M.pad_rays(lmask),
                "x1": M.pad_rays(x1), "cat": g["cat"], "lu_vox": g["lu"].float() * tcap / G.VS, "x4": x4, "valid4": valid4}


def build_cond(model, data, batch, dev, autocast=True):
    """-> dict: cond [V, C, 64, 128], mem (T1) or None, m_unk / lmask / x1 / cat / lu_vox (per view), vae: per-state
    [{mu, logvar, valid8, x4, valid4}]. The scene encoder runs ONCE per state; gradients flow through everything."""
    conds, mems, states = [], [], []
    keep = {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
    for ti, t, dlt, rows in batch:
        st = data.state_tensors(ti, t, dlt, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = model.encode(st["x4"], st["valid4"])
            out = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
        if model.xa:
            cond, mem = out
            mems.append(mem)
        else:
            cond = out
        conds.append(cond)
        for k in keep:
            keep[k].append(st[k])
        states.append({"mu": h["mu"], "logvar": h["logvar"], "valid8": h["valid8"], "x4": st["x4"], "valid4": st["valid4"]})
    bt = {k: torch.cat(v) for k, v in keep.items()}
    bt["cond"] = torch.cat(conds)
    bt["mem"] = tuple(torch.cat([m[i] for m in mems]) for i in range(3)) if mems else None
    bt["vae"] = states
    return bt


def build_mixed(model, parts, dev, autocast=True):
    """parts: [(data, batch), ...] -> one batch dict (views concatenated, VAE states listed)."""
    outs = [build_cond(model, d, b, dev, autocast) for d, b in parts if b]
    bt = {k: torch.cat([o[k] for o in outs]) for k in ("cond", "m_unk", "lmask", "x1", "cat", "lu_vox")}
    bt["mem"] = tuple(torch.cat([o["mem"][i] for o in outs]) for i in range(3)) if outs[0]["mem"] is not None else None
    bt["vae"] = [s for o in outs for s in o["vae"]]
    bt["n_views"] = [o["cond"].shape[0] for o in outs]
    return bt
