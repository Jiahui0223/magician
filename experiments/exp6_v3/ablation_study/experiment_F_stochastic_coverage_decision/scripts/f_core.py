"""Experiment F -- per-state cache + scoring variants (seen2scene env, 1 GPU; offline; GT only to score). configs/F_config.yaml.

Reuses the recorded-planner machinery of the multiview study (mv_core: D1 + D-v2 installed exactly as the E3 planner child, hooks on
h_policy.value_cov and d_policy.lookahead; the recorded sets reproduce the planner's two-step scores bitwise). Per state the planner runs
with K = 4 (the planner) and K = 8 (same seeds + 4 more). Stored per state (results are scores, not raw predictions):
  hard     per-sample one-step V1_k (c1) and per-sample path unions V12_k (c1, c2) from the recorded sets (both runs); area variant (weight 1)
  soft     F2 variants (sigma in {0 (avg0), .25, .5, 1, 2} x weight {w, area}): V1 per c1 and union per path for the K-subsets of configs
  audit    F0.1: the value function on TRUE depth, on TRUE depth through the projection gating (perfect labels -> decode), with / without D-v2
  truth    official one-step per c1, geometric one-step per c1 and per path
  checks   recorded == planner, hard / gamma 1 / sigma 0 reproduce the planner, sigma-0 support == recorded sets, P1 checksums unchanged"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FD))))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "multiview_coverage_study", "scripts"))
import mv_core as C  # noqa: E402

np, torch, FX, HP, DP, DV2, PC, MC = C.np, C.torch, C.FX, C.HP, C.DP, C.DV2, C.PC, C.MC
from seen2scene.futureview import geometry as G  # noqa: E402

CFG = C.yaml.safe_load(open(os.path.join(FD, "configs", "F_config.yaml")))
SIGMAS = [float(s) for s in CFG["F2"]["sigma_over_tau_cov"]]          # tau_cov = 1.0 runtime
OFF = 1 << 19
VS = 0.011


def chk(*ts):
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ sources (exactly the hard-set rays)
def sources(Vc, k, geo, z0, tree):
    """D-v2-novel hits of sample k: (keys int64 [n], runtime points [n,3] float64, w [n]) -- the value_cov / dv2 operations of mv_core.novel_rays."""
    t = Vc.tm[k]
    vis = DV2.visible_mask(C.INST["grid"], geo.fr, Vc.v, t, Vc.kM)
    t = torch.where(vis, t, torch.full_like(t, float("inf")))
    hit = torch.isfinite(t)
    p = Vc.v["o"][None] + Vc.v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    pr = geo.fr.model_to_runtime(p.double().cpu().numpy())
    dd, _ = tree.query(pr, k=1, distance_upper_bound=C.EPS)
    nov = (hit.cpu().numpy() & ~np.isfinite(dd))
    prn = pr[nov]
    c = np.floor(prn / 1.0).astype(np.int64) + OFF
    key = (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]
    z = (t.double() * Vc.kM * Vc.dz).cpu().numpy()[nov]
    w = np.minimum((z / float(z0)) ** 2, 1.0)
    dev = hit.device
    return torch.as_tensor(key, device=dev), torch.as_tensor(prn, device=dev), torch.as_tensor(w, device=dev)


def decode_key(key):
    m = (1 << 20) - 1
    return torch.stack([(key >> 40) & m, (key >> 20) & m, key & m], 1) - OFF


def offsets(sigma, dev):
    r = int(np.floor(2.0 * sigma)) + 1
    g = torch.arange(-r, r + 1, device=dev)
    o = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    lo = (o.abs() - 1).clamp(min=0).double()
    return o[(lo ** 2).sum(1) <= (2.0 * sigma) ** 2]


def support(key, pr, w, sigma, area):
    """sparse s_k: (keys, vals); sigma 0 = own cell, max weight (the hard set)."""
    wv = torch.ones_like(w) if area else w
    if sigma == 0 or key.numel() == 0:
        u, inv = torch.unique(key, return_inverse=True)
        v = torch.zeros(len(u), dtype=torch.float64, device=key.device).scatter_reduce(0, inv, wv, "amax", include_self=False)
        return u, v
    O = offsets(sigma, key.device)
    cell = torch.floor(pr).long()                                         # runtime cell index (cell size 1.0)
    frac = pr - cell.double()
    keys, vals = [], []
    step = max(1, 4_000_000 // len(O))
    for s in range(0, len(key), step):
        cc, ff, ww = cell[s:s + step], frac[s:s + step], wv[s:s + step]
        oo = O[None].expand(len(cc), -1, -1)                             # [n, m, 3]
        oa = oo.double()
        d = torch.where(oa > 0, oa - ff[:, None], torch.where(oa < 0, ff[:, None] - (oa + 1), torch.zeros_like(oa)))
        d2 = (d.clamp(min=0) ** 2).sum(-1)
        ok = d2 <= (2.0 * sigma) ** 2
        val = ww[:, None] * torch.exp(-d2 / (2 * sigma * sigma))
        tgt = (cc[:, None] + oo)[ok] + OFF
        keys.append((tgt[:, 0] << 40) | (tgt[:, 1] << 20) | tgt[:, 2])
        vals.append(val[ok])
    k_, v_ = torch.cat(keys), torch.cat(vals)
    u, inv = torch.unique(k_, return_inverse=True)
    return u, torch.zeros(len(u), dtype=torch.float64, device=u.device).scatter_reduce(0, inv, v_, "amax", include_self=False)


def gate(keys, own, Vc, geo, ctx):
    """spread cells: centre in domain, not EMPTY in the D100 P1, not near S10 (dilated 0.5 occupancy), visible from the camera (single ray, margin);
    own cells (contain a source hit) always pass."""
    if keys.numel() == 0:
        return torch.zeros(0, dtype=torch.bool, device=keys.device)
    ctr = decode_key(keys).double() + 0.5
    m = torch.as_tensor(geo.fr.runtime_to_model(ctr.cpu().numpy()), device=keys.device)
    ijk = torch.floor(m / VS).long()
    dims = torch.as_tensor(geo.dims, device=keys.device)
    ing = ((ijk >= 0) & (ijk < dims)).all(1)
    q = torch.minimum(ijk.clamp(min=0), dims - 1)
    okd = ing & geo.domain[q[:, 0], q[:, 1], q[:, 2]] & (geo.struct[q[:, 0], q[:, 1], q[:, 2]] != MC.EMPTY)
    g = C.INST["grid"]
    qs = torch.floor((ctr - g.lo_t) / DV2.CFG["cell"]).long()
    oks = (qs >= 0).all(1) & (qs < g.n_t).all(1)
    qc = torch.minimum(qs.clamp(min=0), g.n_t - 1)
    near_s10 = oks & ctx["s10dil"][qc[:, 0], qc[:, 1], qc[:, 2]]
    okd = okd & ~near_s10
    vis = torch.ones_like(okd)
    sel = torch.nonzero(okd & ~own)[:, 0]
    if sel.numel():
        dvec = ctr[sel] - Vc.c[None]
        L = dvec.norm(dim=1)
        fp = DV2.CFG["footprint"]
        DV2.CFG["footprint"] = 0.0
        tob = g.t_obs(Vc.c.cpu().numpy(), (dvec / L[:, None]).cpu().numpy(), L.cpu().numpy())
        DV2.CFG["footprint"] = fp
        vis[sel] = L <= tob + DV2.CFG["margin"]
    return own | (okd & vis)


def agg(maps, n):
    """qbar over a subset of samples: mean of sparse maps."""
    if not maps:
        return (torch.zeros(0, dtype=torch.int64), torch.zeros(0, dtype=torch.float64))
    k = torch.cat([m[0] for m in maps])
    v = torch.cat([m[1] for m in maps]) / n
    u, inv = torch.unique(k, return_inverse=True)
    return u, torch.zeros(len(u), dtype=torch.float64, device=u.device).scatter_add(0, inv, v)


def union(a, b, rule):
    if a[0].numel() == 0:
        return float(b[1].sum())
    if b[0].numel() == 0:
        return float(a[1].sum())
    k = torch.cat([a[0], b[0]])
    v = torch.cat([a[1], b[1]])
    u, inv = torch.unique(k, return_inverse=True)
    if rule == "max":
        return float(torch.zeros(len(u), dtype=torch.float64, device=u.device).scatter_reduce(0, inv, v, "amax", include_self=False).sum())
    if rule == "clip_sum":
        return float(torch.zeros(len(u), dtype=torch.float64, device=u.device).scatter_add(0, inv, v).clamp(max=1.0).sum())
    p = torch.ones(len(u), dtype=torch.float64, device=u.device).scatter_reduce(0, inv, 1.0 - v.clamp(max=1.0), "prod", include_self=True)
    return float((1.0 - p).sum())


# ------------------------------------------------------------------------------------------------ one state
def analyze_state(S):
    dev = C.INST["dev"]
    core, rec, z0, kM, fr = S["core"], S["rec"], S["z0"], S["kM"], S["fr"]
    tim = {}
    t0 = time.time()
    geo = PC.Geometry.from_fusion(fr, S["dims"], S["dom"], S["fu"], dev)
    tree = HP.obs_tree()
    fu = S["fu"]
    c_before = chk(fu.sum, fu.cnt, fu.empty, fu.sparse.sum, fu.sparse.cnt, fu.sparse.empty, C.INST["grid"].occ)
    run4 = C.run_planner(core, geo, S["h"], rec, S["eps8"][:4], z0)
    run8 = C.run_planner(core, geo, S["h"], rec, S["eps8"], z0)
    c_after = chk(fu.sum, fu.cnt, fu.empty, fu.sparse.sum, fu.sparse.cnt, fu.sparse.empty, C.INST["grid"].occ)
    tim["planner_K4_s"], tim["planner_K8_s"] = run4["seconds"], run8["seconds"]
    keys1 = run4["keys1"]
    ch = {c1: C.children((c1,), geo) for c1 in keys1}
    paths = [(a, b) for a in keys1 for b in ch[a]]
    allk = set(keys1) | {b for _, b in paths}
    pl4 = [float(x) for x in run4["out"]["distp1"]]
    pl8 = [float(x) for x in run8["out"]["distp1"]]
    rec4 = C.two_step_scores(keys1, ch, run4["SETS"], range(4))
    reg = {"recorded_vs_planner_K4": max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(rec4, pl4)),
           "recorded_vs_planner_K8": max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(C.two_step_scores(keys1, ch, run8["SETS"], range(8)), pl8)),
           "p1_unchanged_by_scoring": c_before == c_after}
    # ---------------- hard per-sample values (both runs)
    def hard_vals(SETS, K, area=False):
        def val(s):
            return float(len(s[0])) if area else float(s[1].sum())
        V1 = {c1: [val(SETS[c1][k]) for k in range(K)] for c1 in keys1}
        V12 = {}
        for a, b in paths:
            if area:
                V12[(a, b)] = [float(len(np.union1d(SETS[a][k][0], SETS[b][k][0]))) for k in range(K)]
            else:
                V12[(a, b)] = [float(DP.union_sets(SETS[a][k], SETS[b][k])[1].sum()) for k in range(K)]
        return V1, V12
    H4, H8 = hard_vals(run4["SETS"], 4), hard_vals(run8["SETS"], 8)
    H4a, H8a = hard_vals(run4["SETS"], 4, True), hard_vals(run8["SETS"], 8, True)
    # gamma = 1 / hard switch reproduce the planner (the F1 formula of configs, applied to the cached per-sample values)
    def f1_scores(V1, V12, ks, gamma):
        out = []
        for c1 in keys1:
            v1 = float(np.mean([V1[c1][k] for k in ks]))
            if not ch[c1]:
                out.append(v1)
                continue
            if gamma == 1.0:
                out.append(max(float(np.mean([V12[(c1, b)][k] for k in ks])) for b in ch[c1]))
            else:
                out.append(v1 + gamma * max(max(0.0, float(np.mean([V12[(c1, b)][k] for k in ks])) - v1) for b in ch[c1]))
        return out
    g1 = f1_scores(*H4, range(4), 1.0)
    reg["gamma1_vs_planner"] = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(g1, pl4))
    reg["gamma1_bitwise"] = g1 == pl4
    tim["hard_s"] = time.time() - t0
    # ---------------- view caches + truth
    t1 = time.time()
    VC = {}
    for key in allk:
        for run, tag in ((run4, 4), (run8, 8)):
            c = C.ViewC(key, run["V"][key], run["HT"][key], fr, kM, dev)
            VC[(key, tag)] = c
        c = VC[(key, 4)]
        tt = torch.as_tensor(FX.gt_first_hit(S["isect"], c.c.cpu().numpy(), c.d.cpu().numpy(), float(c.tcap.max())), device=dev)
        c.ttrue = torch.where(tt <= c.tcap, tt, torch.full_like(tt, float("inf")))
    lo, hi = S["box"]
    norm = 100.0 / (len(S["gt"]) * S["vis"])
    NC = {}
    for key in allk:
        c = VC[(key, 4)]
        fin = torch.isfinite(c.ttrue)
        p = (c.c[None] + c.d * torch.where(fin, c.ttrue, torch.zeros_like(c.ttrue))[:, None]).cpu().numpy()
        okp = fin.cpu().numpy() & ((c.ttrue * c.dz).cpu().numpy() < 70.0) & np.all((p >= lo) & (p <= hi), 1)
        if okp.sum():
            dd, _ = C.cKDTree(p[okp]).query(S["gt"], k=1, distance_upper_bound=C.EPS, workers=8)
            NC[key] = np.isfinite(dd) & ~S["covered0"]
        else:
            NC[key] = np.zeros(len(S["gt"]), bool)
    geo1 = [float(NC[c1].sum()) * norm for c1 in keys1]
    geo_path = [float((NC[a] | NC[b]).sum()) * norm for a, b in paths]
    # ---------------- F0.1 GT-depth audit (first-layer views)
    audit = []
    for c1 in keys1:
        c = VC[(c1, 4)]
        v = c.v
        tm_true = (c.ttrue / kM).to(v["d"].dtype)
        lab = G.encode_labels(v["o"], v["d"], v["pr"], tm_true, v["dz"].to(v["d"].dtype), kM, geo.struct, geo.domain)
        tg, _ = G.decode(lab["b"][None].to(v["d"].dtype), lab["y"][None].to(v["d"].dtype), v["o"], v["d"], v["pr"], geo.struct, geo.domain)
        tg = tg[0]
        row = {"true_vc0": float(C.VC0(tm_true, v, geo, z0, C.VCFG)[1].sum()), "gated_vc0": float(C.VC0(tg, v, geo, z0, C.VCFG)[1].sum()),
               "gated_dv2": float(C.INST["vc_dv2"](tg, v, geo, z0, C.VCFG)[1].sum()), "true_dv2": float(C.INST["vc_dv2"](tm_true, v, geo, z0, C.VCFG)[1].sum())}
        cat = lab["cat"]
        for nm, cid in G.CAT.items():
            m = cat == cid
            if m.any():
                row[f"true_vc0_cat_{nm}"] = float(C.VC0(torch.where(m, tm_true, torch.full_like(tm_true, float("inf"))), v, geo, z0, C.VCFG)[1].sum())
                row[f"rays_cat_{nm}"] = int(m.sum())
        audit.append(row)
    tim["truth_audit_s"] = time.time() - t1
    # ---------------- F2 soft supports
    t2 = time.time()
    g = C.INST["grid"]
    s10 = torch.zeros(tuple(g.n), dtype=torch.bool, device=dev)
    obs = np.concatenate(HP.STATE["obs"]) if HP.STATE["obs"] else np.zeros((0, 3))
    if len(obs):
        q = torch.floor((torch.as_tensor(obs, device=dev) - g.lo_t) / DV2.CFG["cell"]).long()
        ok = (q >= 0).all(1) & (q < g.n_t).all(1)
        q = q[ok]
        s10[q[:, 0], q[:, 1], q[:, 2]] = True
    ctx = {"s10dil": torch.nn.functional.max_pool3d(s10[None, None].half(), 5, 1, 2)[0, 0] > 0}
    del s10
    variants = [("avg0", 0.0)] + [(f"s{sg:g}", sg) for sg in SIGMAS]
    SUP = {}                                        # (run, key, k, variant, weight) -> sparse map
    sigma_check = 0
    for (key, run_tag), c in VC.items():
        K_ = run_tag
        S_rec = (run4 if run_tag == 4 else run8)["SETS"][key]
        src = [sources(c, k, geo, z0, tree) for k in range(K_)]
        for k in range(K_):                          # sigma-0 support == the recorded hard set (ids and weights)
            u, vv = support(*src[k], 0.0, False)
            if sigma_check < 40:
                ids = np.sort(np.asarray(S_rec[k][0], np.int64))
                assert np.array_equal(u.cpu().numpy(), ids), ("sigma-0 support != recorded set ids", key, k)
                w_rec = np.asarray(S_rec[k][1])[np.argsort(np.asarray(S_rec[k][0], np.int64))]
                assert np.allclose(vv.cpu().numpy(), w_rec, rtol=0, atol=1e-12), ("sigma-0 weights", key, k)
                sigma_check += 1
        # gate once per view over the largest kernel's cells
        allkeys = []
        maps = {}
        for k in range(K_):
            own = src[k][0]
            for vn, sg in variants:
                for area in (False, True):
                    maps[(k, vn, area)] = support(*src[k], sg, area)
                    allkeys.append(maps[(k, vn, area)][0])
        uk = torch.unique(torch.cat(allkeys)) if allkeys else torch.zeros(0, dtype=torch.int64, device=dev)
        ownall = torch.unique(torch.cat([s[0] for s in src])) if src else torch.zeros(0, dtype=torch.int64, device=dev)
        gate_ok = gate(uk, torch.isin(uk, ownall), c, geo, ctx)
        okset = uk[gate_ok]
        for kk, (u, vv) in maps.items():
            k, vn, area = kk
            own_k = torch.isin(u, src[k][0])
            keep = own_k | torch.isin(u, okset)
            SUP[(run_tag, key, k, vn, area)] = (u[keep], vv[keep])
    tim["soft_build_s"] = time.time() - t2
    t3 = time.time()
    subsets = {("K4", "all4"): (4, [0, 1, 2, 3])}
    for nm, ks in (("first1", [0]), ("first2", [0, 1]), ("first4", [0, 1, 2, 3]), ("first8", list(range(8))), ("quadA", [0, 1, 2, 3]), ("quadB", [4, 5, 6, 7])):
        subsets[("K8", nm)] = (8, ks)
    for k in range(8):
        subsets[("K8", f"single{k}")] = (8, [k])
    soft = {}
    for (rn, sn), (rt, ks) in subsets.items():
        if sn == "quadA":
            continue                                  # == first4
        for vn, _ in variants:
            for area in (False, True):
                Q = {key: agg([SUP[(rt, key, k, vn, area)] for k in ks], len(ks)) for key in allk}
                tagv = f"{vn}_{'area' if area else 'w'}"
                rules = ["max"] + (["clip_sum", "noisy_or"] if sn in ("all4", "first1", "first4") else [])
                for rule in rules:
                    soft[f"{rn}|{sn}|{tagv}|{rule}"] = {"V1": [float(Q[c1][1].sum()) for c1 in keys1], "V12": [union(Q[a], Q[b], rule) for a, b in paths]}
    tim["soft_score_s"] = time.time() - t3
    # diffusion diagnostic (offline, GT): share of each first-layer K4 qbar mass on cells near the GT target surface (cell of a GT point,
    # dilated by one cell) and on the hard cells of the same view
    gc = np.unique(np.floor(S["gt"]).astype(np.int64), axis=0)
    o27 = np.stack(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij"), -1).reshape(-1, 3)
    gd = (gc[:, None] + o27[None]).reshape(-1, 3) + OFF
    gtk = torch.as_tensor(np.unique((gd[:, 0] << 40) | (gd[:, 1] << 20) | gd[:, 2]), device=dev)
    diff = {}
    for vn, _ in variants:
        for area in (False, True):
            tot = near = own = 0.0
            for c1 in keys1:
                u, vv = agg([SUP[(4, c1, k, vn, area)] for k in range(4)], 4)
                hk = torch.unique(torch.cat([SUP[(4, c1, k, "avg0", area)][0] for k in range(4)]))
                tot += float(vv.sum())
                near += float(vv[torch.isin(u, gtk)].sum())
                own += float(vv[torch.isin(u, hk)].sum())
            diff[f"{vn}_{'area' if area else 'w'}"] = {"mass": tot, "near_gt_share": near / tot if tot > 0 else None, "hard_cell_share": own / tot if tot > 0 else None}
    # soft-off check: avg0 one-step == hard one-step mean (same cells and weights, linear mean)
    reg["avg0_onestep_vs_hard_max_abs"] = max(abs(x - float(np.mean(H4[0][c1]))) for x, c1 in zip(soft["K4|all4|avg0_w|max"]["V1"], keys1))
    st = {**S["meta"], "n_legal": len(keys1), "keys1": [list(k) for k in keys1], "paths": [[list(a), list(b)] for a, b in paths],
          "seeds": S["seeds"], "regression": reg, "timing": tim, "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30,
          "planner_K4": pl4, "planner_K8": pl8, "planner_log": S.get("planner_log"),
          "oracle": [None if (x is None or not np.isfinite(x)) else float(x) for x in S["oracle"]], "geo1": geo1, "geo_path": geo_path,
          "hard": {"K4": {"V1": [H4[0][c1] for c1 in keys1], "V12": [H4[1][p] for p in paths]}, "K8": {"V1": [H8[0][c1] for c1 in keys1], "V12": [H8[1][p] for p in paths]},
                   "K4_area": {"V1": [H4a[0][c1] for c1 in keys1], "V12": [H4a[1][p] for p in paths]}, "K8_area": {"V1": [H8a[0][c1] for c1 in keys1], "V12": [H8a[1][p] for p in paths]}},
          "soft": soft, "audit": audit, "diffusion": diff}
    if S.get("planner_log") is not None:
        lg = S["planner_log"]
        st["regression"]["replay_vs_closed_loop_log"] = (len(lg) == len(pl4) and all(abs(a - b) <= 1e-9 * max(1.0, abs(b)) for a, b in zip(pl4, lg)))
        st["regression"]["replay_vs_closed_loop_log_bitwise"] = pl4 == lg
    st.pop("planner_log")
    del run4, run8, VC, SUP, geo, ctx
    torch.cuda.empty_cache()
    return st
