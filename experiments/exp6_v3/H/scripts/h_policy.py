"""EXP6 V3 / H -- decision policies H0-H3 on top of the frozen V3-T1 planner (seen2scene env, planner process only).

The frozen planner child (task2 t24_v2_out_planner -> task1 v2_planner) calls v2_plan_core.score_state(...) once per step
and takes out["distp1"] as the action scores (m_score.choose: argmax, ties lowest index, all-zero -> seeded fallback).
install(policy) replaces score_state by a wrapper that (1) calls the frozen score_state unchanged (keep_hits=True: the
same V3 predictions, D100 input, R10 features, shared K = 4 noise), (2) recomputes the per-sample value of every legal
view under the policy and (3) writes the policy score into out["distp1"] (the original DistP1 kept as out["h0_distp1"]).
It also wraps the P1 fusion's integrate_record to collect what the agent itself observed: the S10 sample points of every
executed frame (runtime frame) and the pose history / legal list of the current service record. Nothing else changes.

Per-sample value of one predicted view (t_k [R] first-hit distance, inf = no hit):
  h0   SCO.new_set_from_depth: in-domain hit voxels (P1 voxel 0.011 model units) outside the sparse band_dil, unique
       voxels, weight max_v min(1, (z / z0)^2); sum                                     (= frozen DistP1, unchanged)
  cov  (H1-H3) in-domain hits (same domain test) mapped to the runtime frame; novel iff no observed S10 sample point
       within eps_cov (the official coverage tolerance 2 x 0.05 x 10 = 1.0); unique cells of size cell_size (= eps_cov)
       in the runtime frame; weight = max over the cell's hits of the same w(z); sum
Aggregation over the K samples g_k:
  H0 / H1  mean_k g_k (np.mean, as the frozen planner)        H2  mean_k g_k - lambda * std_k g_k (ddof 0)
  H3  two-step: for every legal c1 and every c2 in N(c1) (settings-grid neighbours of c1, MAGICIAN get_neighboring_poses
      + unvisited-first with c1 counted visited, c1 itself excluded, segments c1 -> c2 crossing the known D100 surface band
      dropped; no GT collision test), per sample k the union of the novel cells of c1 and c2 (max weight per cell) ->
      Q(c1, c2) = mean_k - lambda std_k; score(c1) = max_c2 Q (ties lowest index); no c2 -> the H2 score of c1. Only c1 is
      executed; the planner re-plans after the real observation. Predictions of c2 use the same cached V3 features and the
      same K noise maps (open-loop, frozen P1); nothing is written into the P1.
Config (FROZEN_CONFIG.yaml, policy section) is passed by env H_CONFIG; env H_POLICY in {h0, h1, h2, h3, h1deg, h3d1}:
  h1deg = H1 code path with the h0 per-sample value (regression: must reproduce H0); h3d1 = H3 code path with depth 1
  (regression: must reproduce H2)."""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

STATE = {"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set(), "policy": None, "cfg": None, "pose": None, "stats": {}}


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cfg():
    import yaml
    p = os.environ["H_CONFIG"]
    return yaml.safe_load(open(p)), sha256_file(p)


# ------------------------------------------------------------------------------------------------ observation capture
def capture(rec):
    sp = np.asarray(rec["sample_points"], np.float64).reshape(-1, 3)
    if len(sp):
        STATE["obs"].append(sp)
    STATE["rec"] = rec
    STATE["visited"] = {tuple(int(x) for x in r) for r in np.asarray(rec["executed_pose_idx_history"]).reshape(-1, 5)}


def obs_tree():
    from scipy.spatial import cKDTree
    n = sum(len(x) for x in STATE["obs"])
    if STATE["tree"] is None or STATE["n_tree"] != n:
        pts = np.concatenate(STATE["obs"]) if STATE["obs"] else np.zeros((1, 3)) + 1e12
        STATE["tree"], STATE["n_tree"] = cKDTree(pts), n
    return STATE["tree"]


# ------------------------------------------------------------------------------------------------ per-sample values
def value_h0(SCO, t, v, geo, z0):
    ids, wv = SCO.new_set_from_depth(t, v["o"], v["d"], v["dz"], geo.fr.k_M, geo.dims, geo.domain, geo.caster.band_dil, z0)
    return ids.cpu().numpy().astype(np.int64), wv.cpu().numpy()


def value_cov(t, v, geo, z0, cfg):
    """(cell ids int64 sorted unique, weights float64) of the predicted hits of one view / sample that are in the planning
    domain and farther than eps_cov from every observed S10 sample point (runtime frame)."""
    VS = 0.011
    hit = torch.isfinite(t)
    p = v["o"][None] + v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    sel = torch.nonzero(hit)[:, 0]
    if sel.numel() == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    pm = p[sel].double().cpu().numpy()
    pr = geo.fr.model_to_runtime(pm)                                         # runtime (y-up) frame, float64
    eps = float(cfg["eps_cov"])
    dist, _ = obs_tree().query(pr, k=1, distance_upper_bound=eps)
    novel = ~np.isfinite(dist)                                               # no observed point within eps
    if not novel.any():
        return np.zeros(0, np.int64), np.zeros(0)
    pr = pr[novel]
    z = (t[sel].double() * float(geo.fr.k_M) * v["dz"][sel].double()).cpu().numpy()[novel]
    w = np.minimum((z / float(z0)) ** 2, 1.0)
    c = np.floor(pr / float(cfg["cell_size"])).astype(np.int64) + (1 << 19)
    key = (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]
    ids, inv = np.unique(key, return_inverse=True)
    wv = np.zeros(len(ids))
    np.maximum.at(wv, inv, w)
    return ids, wv


def union_value(a, b):
    """Weighted size of the union of two (ids, weights) cell sets, max weight per cell; and |a ∩ b| / |b|."""
    if len(b[0]) == 0:
        return float(a[1].sum()), 0.0
    if len(a[0]) == 0:
        return float(b[1].sum()), 0.0
    ids = np.concatenate([a[0], b[0]])
    w = np.concatenate([a[1], b[1]])
    u, inv = np.unique(ids, return_inverse=True)
    wv = np.zeros(len(u))
    np.maximum.at(wv, inv, w)
    inter = len(a[0]) + len(b[0]) - len(u)
    return float(wv.sum()), inter / len(b[0])


# ------------------------------------------------------------------------------------------------ settings-grid graph (H3)
SHIFTS = np.asarray([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0],
                     [0, 0, 0, 1, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, -1]], np.int64)


def neighbours(idx, grid):
    """MAGICIAN get_neighboring_poses: 6 moves + 4 rotations, clamp x / y / z / elevation, azimuth modulo, unique (rows
    sorted lexicographically, as torch.unique(dim=0))."""
    r = np.asarray(idx, np.int64)[None] + SHIFTS
    r[:, :4] = np.maximum(r[:, :4], 0)
    for a, n in enumerate(("pose_l", "pose_w", "pose_h", "pose_n_elev")):
        r[:, a] = np.minimum(r[:, a], grid[n] - 1)
    r[:, 4] = r[:, 4] % grid["pose_n_azim"]
    return np.unique(r, axis=0)


def second_layer(c1, visited, grid):
    """Unvisited-first (get_valid_neighbors) with c1 counted as visited; c1 itself excluded."""
    nb = [tuple(int(x) for x in r) for r in neighbours(c1, grid)]
    vis = visited | {tuple(c1)}
    nb = [n for n in nb if n != tuple(c1)]
    new = [n for n in nb if n not in vis]
    return new if new else nb


def segment_blocked(geo, c_from, c_to, step_vox=0.5):
    """True if the straight segment between two camera centres (runtime) crosses a voxel of the known D100 surface band
    of the current P1 (inference-available geometry; unknown space counts as free)."""
    a = np.asarray(geo.fr.runtime_to_model(np.asarray(c_from, np.float64)[None]))[0]
    b = np.asarray(geo.fr.runtime_to_model(np.asarray(c_to, np.float64)[None]))[0]
    L = np.linalg.norm(b - a)
    if L == 0:
        return False
    n = int(np.ceil(L / (0.011 * step_vox))) + 1
    pts = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
    ijk = np.floor(pts / 0.011).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.asarray(geo.dims)[None]), 1)
    if not ok.any():
        return False
    ijk = torch.as_tensor(ijk[ok], device=geo.caster.obs_band.device)
    return bool(geo.caster.obs_band[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any())


# ------------------------------------------------------------------------------------------------ install
def install(policy):
    import v2_plan_core as PC
    import m_p1
    from seen2scene.futureview import scoring as SCO
    cfg, cfg_sha = load_cfg()
    pc = cfg["policies"][policy] if policy in cfg["policies"] else cfg["policies"][{"h1deg": "h1", "h3d1": "h3"}[policy]]
    STATE["policy"], STATE["cfg"] = policy, pc
    pol_sha = sha256_file(os.path.abspath(__file__))
    info = {"h_policy": policy, "policy_file_sha256": pol_sha, "config_sha256": cfg_sha, "config_version": cfg.get("version"), "policy_cfg": pc}
    if policy in ("h3", "h3d1"):
        scene = sys.argv[sys.argv.index("--scene") + 1]
        z = np.load(os.path.join(cfg["pose_tables"], f"{scene}.npz"))
        STATE["pose"] = {"lut": {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, "R": z["R"], "T": z["T"], "C": z["C"],
                         "grid": json.loads(str(z["grid"]))}
        info["pose_table_sha256"] = sha256_file(os.path.join(cfg["pose_tables"], f"{scene}.npz"))
    sys.stderr.write("H_POLICY " + json.dumps(info) + "\n")
    import v2_common as LC
    for g in ("V2M", "V2O"):
        if g in LC.ARMS:
            LC.ARMS[g]["h_policy"] = info
    cls = m_p1.P1Fusion
    orig_int = cls.integrate_record

    def integrate_record(self, rec, *a, **k):
        capture(rec)
        return orig_int(self, rec, *a, **k)
    cls.integrate_record = integrate_record
    if policy == "h0":
        return info
    orig_score = PC.score_state

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw):
        want_hits = kw.pop("keep_hits", False)
        tt = time.time()
        out, tm, views = orig_score(core, geo, h, legal_R, legal_T, eps, z0, *a, keep_hits=True, **kw)
        t_frozen = time.time() - tt
        if not views:
            if not want_hits:
                out.pop("hits", None)
            return out, tm, views
        K = eps.shape[0]
        lam = float(pc.get("lambda_risk", 0.0))
        sets = []
        for m, v in enumerate(views):
            tk = out["hits"][m]["t"]
            if policy == "h1deg" or pc["value"] == "h0":
                sets.append([value_h0(SCO, tk[k], v, geo, z0) for k in range(K)])
            else:
                sets.append([value_cov(tk[k], v, geo, z0, pc) for k in range(K)])
        g = np.asarray([[float(s_[1].sum()) for s_ in row] for row in sets])          # [n_views, K]
        mu = np.asarray([float(np.mean(list(r))) for r in g])
        sd = g.std(1)                                                                # ddof 0
        out["h0_distp1"] = list(out["distp1"])
        out["h_value_k"] = g.tolist()
        out["h_mean"] = mu.tolist()
        out["h_std"] = sd.tolist()
        if policy in ("h1", "h1deg"):
            score = mu
        else:
            score = mu - lam * sd
        out["h_risk"] = score.tolist()
        info_h3 = None
        if policy in ("h3", "h3d1"):
            depth = 1 if policy == "h3d1" else int(pc["depth"])
            score, info_h3 = h3_scores(core, geo, h, legal_R, legal_T, eps, z0, sets, score, lam, depth, orig_score, pc, kw)
            out["h3"] = info_h3
        out["distp1"] = [float(x) for x in score]
        tm = dict(tm, frozen_score_state_s=t_frozen, policy_s=time.time() - tt - t_frozen)
        if not want_hits:
            out.pop("hits", None)
        return out, tm, views
    PC.score_state = score_state
    return info


def h3_scores(core, geo, h, legal_R, legal_T, eps, z0, sets, s_h2, lam, depth, orig_score, pc, kw):
    rec = STATE["rec"]
    P = STATE["pose"]
    legal_idx = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    n = len(legal_idx)
    info = {"depth": depth, "n_c1": n, "n_c2_unique": 0, "blocked": 0, "per_c1": []}
    if depth == 1:
        return np.asarray(s_h2, float), info
    from seen2scene.futureview import scoring as SCO  # noqa: F401
    c2_of, need = [], []
    for i, c1 in enumerate(legal_idx):
        lst = []
        for c2 in second_layer(c1, STATE["visited"], P["grid"]):
            j = P["lut"][c2]
            if segment_blocked(geo, P["C"][P["lut"][c1]], P["C"][j]):
                info["blocked"] += 1
                continue
            lst.append(c2)
            if c2 not in legal_idx and c2 not in need:
                need.append(c2)
        c2_of.append(lst)
    info["n_c2_unique"] = len(need)
    # predictions of the second-layer views: same cached features h, same K noise maps, frozen score_state path
    pred = {c: sets[legal_idx.index(c)] for c in legal_idx}
    K = eps.shape[0]
    chunk = int(pc.get("c2_chunk", 16))
    for s0 in range(0, len(need), chunk):
        part = need[s0:s0 + chunk]
        Rs = [P["R"][P["lut"][c]] for c in part]
        Ts = [P["T"][P["lut"][c]] for c in part]
        o2, _, v2 = orig_score(core, geo, h, Rs, Ts, eps, z0, keep_hits=True)
        for q, c in enumerate(part):
            tk = o2["hits"][q]["t"]
            pred[c] = [value_cov(tk[k], v2[q], geo, z0, pc) for k in range(K)]
        del o2, v2
    score = np.asarray(s_h2, float).copy()
    for i, c1 in enumerate(legal_idx):
        best, best_c2, ov = None, None, []
        for c2 in c2_of[i]:
            vals = []
            for k in range(K):
                u, o = union_value(pred[c1][k], pred[c2][k])
                vals.append(u)
                ov.append(o)
            q = float(np.mean(vals)) - lam * float(np.std(vals))
            if best is None or q > best:
                best, best_c2 = q, c2
        if best is not None:
            score[i] = best
        info["per_c1"].append({"c1": list(c1), "n_c2": len(c2_of[i]), "best_c2": list(best_c2) if best_c2 else None, "q": best,
                               "s_h2": float(s_h2[i]), "overlap_mean": float(np.mean(ov)) if ov else None})
    return score, info
