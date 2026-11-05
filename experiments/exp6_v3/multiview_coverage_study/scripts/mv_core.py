"""Multi-View Coverage Reasoning Study -- per-state analysis (seen2scene env, 1 GPU; offline; GT used only to score).

For one planner state (P1, encoder features, noise of the frozen planner) the frozen D-v2 planner is run twice (K = 4 = the planner,
K = 8 = the same seeds plus four more). Every prediction the planner makes -- first-layer candidates and all lookahead children --
is RECORDED from the running planner (hooks on h_policy.value_cov and d_policy.lookahead); nothing of the planner is re-implemented.
The recorded per-sample novel-cell sets are fed back through d_policy.union_sets / q_of and must reproduce the planner's two-step
scores (regression). On top of the recorded predictions this module measures (configs/mv_v1.yaml):
  A  complementarity of candidate pairs (first-layer pairs and the planner's own (c1, c2) paths): predicted vs true overlap,
     union, marginal gain, double counting of the same true surface, pair ranking
  B  K = 1 / 2 / 4 / 8 (first K of the same seeds; also averages over disjoint subsets): ranking quality, sample diversity,
     pose vs noise variation, effective sample count
  C  cross-view geometric consistency of the generated first hits (projection, depth comparison, visibility), per sample and
     distribution level, attributed with the GT
  D  offline Counterfactual View Agreement Rescoring (CVAR) of the two-step score and its ranking quality
Truth: official one-step gain (oracle), value function on GT depth (same units as the prediction), geometric coverage of the
official GT target points (mono rule)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MV = os.path.dirname(HERE)
X = os.path.dirname(os.path.dirname(MV))
R = os.path.dirname(X)
E3 = os.path.join(X, "exp6_v3", "ablation_study", "experiment_E_stability_plateau", "E3_dv2_closed_loop")
sys.path.insert(0, os.path.join(E3, "scripts"))
import dv2_fixed as FX  # noqa: E402  (module setup: b2 P1 patch, V3 planner core; installs nothing)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HP, DP, DV2, PC, PF, MC, MS, TD, LC, PR, RY = FX.HP, FX.DP, FX.DV2, FX.PC, FX.PF, FX.MC, FX.MS, FX.TD, FX.LC, FX.PR, FX.RY
VC0 = HP.value_cov                     # pristine D1 value function (captured before any install)
VS = 0.011
DTH = 4 * np.radians(60.0) / 256.0     # angular size of one scoring block (rad)
EPS = 1.0
VCFG = {"eps_cov": 1.0, "cell_size": 1.0}
KMAX = 8
REC = {"on": False, "vc": {}, "c2_views": {}, "c2_hits": {}}
INST = {}


def sync():
    torch.cuda.synchronize()
    return time.time()


# ------------------------------------------------------------------------------------------------ planner installation + recording
def install(scene, dev):
    """D1 (d_policy d1) + D-v2 (mode full, dv2_v2.yaml) exactly as the E3 planner child, then recording hooks on top."""
    sys.argv += ["--scene", scene]
    DP.install("d1")
    cfg = yaml.safe_load(open(os.path.join(E3, "configs", "dv2_v2.yaml")))["dv2"]
    grid = DV2.install(scene, dev, cfg)
    DV2.STATE["mode"] = "full"
    vc_dv2 = HP.value_cov

    def rec_vc(t, v, geo, z0, cfg_):
        r = vc_dv2(t, v, geo, z0, cfg_)
        if REC["on"]:
            REC["vc"].setdefault(id(v), []).append(r)
        return r
    HP.value_cov = rec_vc
    la = DP.lookahead

    def rec_la(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats):
        def rs(core_, geo_, h_, Rl, Tl, eps_, z0_, keep_hits=True):
            out, tm, views = orig_score(core_, geo_, h_, Rl, Tl, eps_, z0_, keep_hits=keep_hits)
            if REC["on"]:
                for j, v in enumerate(views):
                    # several pose keys can share one camera pose (R, T bitwise equal); all of them get this (identical) view
                    for key in INST["tkey"][(np.asarray(Rl[j], np.float64).tobytes(), np.asarray(Tl[j], np.float64).tobytes())]:
                        REC["c2_views"][key] = v
                        REC["c2_hits"][key] = out["hits"][j]
            return out, tm, views
        return la(core, geo, h, eps, z0, sets, s_one, pc, rs, stats)
    DP.lookahead = rec_la
    INST.update({"grid": grid, "vc_dv2": vc_dv2, "dev": dev})
    return grid


def set_scene(scene):
    """Pose table of the scene (d_policy convention) + a fresh D-v2 observed grid for the scene (the installed wrappers hold the grid object)."""
    z = np.load(os.path.join(R, "work_dirs", "exp6_v3", "H", "pose_tables", f"{scene}.npz"))
    HP.STATE["pose"] = {"lut": {tuple(int(x) for x in r): j for j, r in enumerate(z["idx"])}, "R": z["R"], "T": z["T"], "C": z["C"], "grid": json.loads(str(z["grid"]))}
    tk = {}
    for i in range(len(z["idx"])):
        tk.setdefault((np.asarray(z["R"][i], np.float64).tobytes(), np.asarray(z["T"][i], np.float64).tobytes()), []).append(tuple(int(x) for x in z["idx"][i]))
    INST["tkey"] = tk
    ng = DV2.ObservedGrid(scene, INST["dev"])
    INST["grid"].__dict__.update(ng.__dict__)
    HP.STATE.update({"obs": [], "tree": None, "n_tree": -1, "rec": None, "visited": set()})


def children(path, geo):
    """d_policy.lookahead children rule (unvisited-first neighbours, known-band segment test); used only to know which recorded
    views belong to which root -- the recomputed scores are checked against the planner's own."""
    P = HP.STATE["pose"]
    vis = HP.STATE["visited"] | set(path)
    nb = [tuple(int(x) for x in r) for r in HP.neighbours(path[-1], P["grid"])]
    nb = [n for n in nb if n not in path]
    new = [n for n in nb if n not in vis]
    cand = new if new else nb
    return [c for c in cand if not HP.segment_blocked(geo, P["C"][P["lut"][path[-1]]], P["C"][P["lut"][c]])]


def run_planner(core, geo, h, rec, eps, z0):
    REC.update({"on": True, "vc": {}, "c2_views": {}, "c2_hits": {}})
    t0 = sync()
    out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, z0, keep_hits=True)
    dt = sync() - t0
    REC["on"] = False
    keys1 = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    V = {k: views[j] for j, k in enumerate(keys1)}
    HT = {k: out["hits"][j] for j, k in enumerate(keys1)}
    for k, v in REC["c2_views"].items():
        if k not in V:
            V[k], HT[k] = v, REC["c2_hits"][k]
    K = eps.shape[0]
    SETS = {}
    for k, v in V.items():
        s = REC["vc"].get(id(v), [])
        assert len(s) == K, (k, len(s), K)
        SETS[k] = s
    return {"out": out, "keys1": keys1, "V": V, "HT": HT, "SETS": SETS, "seconds": dt, "K": K, "tm": tm}


def q_paired(S1, S2, ks):
    return float(np.mean([float(DP.union_sets(S1[k], S2[k])[1].sum()) for k in ks]))


def q_unpaired(S1, S2, ks):
    return float(np.mean([float(DP.union_sets(S1[a], S2[b])[1].sum()) for a in ks for b in ks]))


def two_step_scores(keys1, ch, SETS, ks, pair=q_paired, sets2=None):
    """score(c1) = max over children of pair(S(c1), S(c2)) (lambda 0), else the one-step mean (d_policy.lookahead, depth 2)."""
    sc = []
    for c1 in keys1:
        s1 = float(np.mean([float(SETS[c1][k][1].sum()) for k in ks]))
        best = None
        for c2 in ch[c1]:
            S2 = (sets2 or {}).get((c1, c2), SETS[c2])
            q = pair(SETS[c1], S2, ks)
            best = q if best is None or q > best else best
        sc.append(best if best is not None else s1)
    return sc


# ------------------------------------------------------------------------------------------------ geometry helpers
class ViewC:
    """Runtime-frame geometry + predictions + truth of one recorded view."""

    def __init__(self, key, v, ht, fr, kM, dev):
        P = HP.STATE["pose"]
        i = P["lut"][key]
        self.key, self.v = key, v
        self.R = torch.as_tensor(P["R"][i], dtype=torch.float64, device=dev)
        self.T = torch.as_tensor(P["T"][i], dtype=torch.float64, device=dev)
        self.c = torch.as_tensor(np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0], device=dev)
        d = torch.as_tensor(MC.to_y_up(v["d"].double().cpu().numpy()), device=dev)
        self.d = d / d.norm(dim=1, keepdim=True)
        pr = v["pr"]
        self.tcap = v["tcap"].double() * kM
        self.tknown = pr["t_known"].double() * kM
        self.tmin, self.tmax = pr["tmin"].double() * kM, torch.minimum(pr["tmax"].double(), v["tcap"].double()) * kM
        self.munk = pr["n_u"] > 0
        self.dz = v["dz"].double()
        self.tm = ht["t"]                              # [K, R] model units as decoded (the planner's tensor)
        self.kM = kM
        self.t = ht["t"].double() * kM                 # [K, R] runtime, inf = no hit
        self.hu = ht["hit_u"] & torch.isfinite(self.t)   # generated hits
        self.ttrue = None


def project(p, B):
    """Ray index of B's 114 x 64 grid hit by world points p [N, 3] (-1 = outside), depth along that ray, distance to the ray."""
    view = p @ B.R + B.T
    z = view[:, 2]
    zs = torch.where(z.abs() < 1e-9, torch.full_like(z, 1e-9), z)
    nx, ny = view[:, 0] * MC.S_PROJ / zs, view[:, 1] * MC.S_PROJ / zs
    u = (MC.W / MC.M_SHORT - nx) * (MC.M_SHORT - 1) / 2.0
    v = (MC.H / MC.M_SHORT - ny) * (MC.M_SHORT - 1) / 2.0
    i = torch.round((u - 1.5) / 4.0).long()
    j = torch.round((v - 1.5) / 4.0).long()
    ok = (z > 0.05) & (i >= 0) & (i < 114) & (j >= 0) & (j < 64)
    r = torch.where(ok, j * 114 + i, torch.full_like(i, -1))
    rc = r.clamp(min=0)
    q = p - B.c[None]
    tp = (q * B.d[rc]).sum(-1)
    e = (q - tp[:, None] * B.d[rc]).norm(dim=-1)
    return r, tp, e


def tol(tp, e=0.0):
    return 1.0 + e + tp * DTH


def novel_rays(Vc, k, geo, z0, tree, use_dv2=True):
    """Per-ray (novel mask, cell key, weight) of sample k, computed with exactly the operations of h_policy.value_cov (model-frame
    hit point -> runtime, S10 distance, 1.0 cells, w(z)) after the D-v2 visibility test of dv2.py: the rays whose cells form the
    planner's set. Checked against the recorded sets by the caller."""
    t = Vc.tm[k]
    if use_dv2:
        vis = DV2.visible_mask(INST["grid"], geo.fr, Vc.v, t, Vc.kM)
        t = torch.where(vis, t, torch.full_like(t, float("inf")))
    hit = torch.isfinite(t)
    p = Vc.v["o"][None] + Vc.v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    pr = geo.fr.model_to_runtime(p.double().cpu().numpy())
    dd, _ = tree.query(pr, k=1, distance_upper_bound=EPS)
    nov = hit & torch.as_tensor(~np.isfinite(dd), device=hit.device)
    c = np.floor(pr / VCFG["cell_size"]).astype(np.int64) + (1 << 19)
    key = torch.as_tensor((c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2], device=hit.device)
    z = t.double() * Vc.kM * Vc.dz
    w = torch.clamp((z / z0) ** 2, max=1.0)
    return nov, key, torch.where(nov, w, torch.zeros_like(w))


# ------------------------------------------------------------------------------------------------ Task C: consistency of an ordered pair
def consistency(A, B, nmask, wA, K):
    """A's generated hits projected into B. nmask [K, R] D-v2-novel rays of A, wA [K, R] weights. Returns weighted sums."""
    o = {f"{lv}_{c}": 0.0 for lv in ("paired", "unpaired", "dist", "dist_novel") for c in ("agree", "front", "through")}
    o.update({"valid": 0.0, "valid_novel": 0.0, "gen": 0.0, "gen_novel": 0.0,
              "gt_through_Awrong": 0.0, "gt_through_Bwrong": 0.0, "gt_through_bothok": 0.0, "gt_through_other": 0.0,
              "gt_agree_Aok": 0.0, "gt_agree_Awrong": 0.0})
    for ka in range(K):
        a = torch.nonzero(A.hu[ka])[:, 0]
        if a.numel() == 0:
            continue
        w = wA[ka, a]
        nv = nmask[ka, a]
        o["gen"] += float(w.sum()) / K
        o["gen_novel"] += float(w[nv].sum()) / K
        p = A.c[None] + A.d[a] * A.t[ka, a][:, None]
        r, tp, e = project(p, B)
        rc = r.clamp(min=0)
        tl = tol(tp, e)
        ok = (r >= 0) & B.munk[rc] & (tp > B.tmin[rc]) & (tp < B.tmax[rc]) & (tp < B.tknown[rc] - tl)
        if not ok.any():
            continue
        a, w, nv, rc, tp, tl = a[ok], w[ok], nv[ok], rc[ok], tp[ok], tl[ok]
        o["valid"] += float(w.sum()) / K
        o["valid_novel"] += float(w[nv].sum()) / K
        dB = B.t[:, rc]                                                          # [K, n] B's predicted depth (inf = none)
        ag = (dB - tp[None]).abs() <= tl[None]
        fr_ = dB < tp[None] - tl[None]
        th = dB > tp[None] + tl[None]
        for nm, m in (("agree", ag), ("front", fr_), ("through", th)):
            o[f"paired_{nm}"] += float((m[ka].double() * w).sum()) / K
            oth = [kb for kb in range(K) if kb != ka]
            o[f"unpaired_{nm}"] += float((m[oth].double().mean(0) * w).sum()) / K
            o[f"dist_{nm}"] += float((m.double().mean(0) * w).sum()) / K
            o[f"dist_novel_{nm}"] += float((m.double().mean(0) * w * nv.double()).sum()) / K
        if A.ttrue is not None:
            ta = A.ttrue[a]
            a_ok = torch.isfinite(ta) & ((A.t[ka, a] - ta).abs() <= tol(ta))
            tb = B.ttrue[rc]
            b_through = ~torch.isfinite(tb) | (tb > tp + tl)
            thf = th.double().mean(0) * w
            o["gt_through_Awrong"] += float(thf[~a_ok].sum()) / K
            o["gt_through_Bwrong"] += float(thf[a_ok & ~b_through].sum()) / K
            o["gt_through_bothok"] += float(thf[a_ok & b_through].sum()) / K
            agf = ag.double().mean(0) * w
            o["gt_agree_Aok"] += float(agf[a_ok].sum()) / K
            o["gt_agree_Awrong"] += float(agf[~a_ok].sum()) / K
    return o


def incons(o, lv="dist"):
    d = o[f"{lv}_agree"] + o[f"{lv}_through"]
    return o[f"{lv}_through"] / d if d > 0 else None


# ------------------------------------------------------------------------------------------------ Task D: CVAR on one path
def cvar_sets(A, B, K, geo, z0, novB, mode="both"):
    """c2 = B's per-sample t with the CVAR-removed rays set to 'no hit', re-scored with the planner's own D-v2 value function."""
    gen = A.hu[:K]
    tA = torch.where(gen, A.t[:K], torch.full_like(A.t[:K], float("nan")))
    med = torch.nanmedian(tA, dim=0).values
    close = gen & ((A.t[:K] - med[None]).abs() <= tol(torch.nan_to_num(med, nan=0.0))[None])
    rel = (close.sum(0) >= 3) & torch.isfinite(med)
    free_to = A.t[:K].min(0).values                                              # all samples free up to this depth (inf = no surface at all)
    out, nrem = [], 0
    H = None
    if mode in ("both", "occ") and rel.any():
        a = torch.nonzero(rel)[:, 0]
        ph = A.c[None] + A.d[a] * med[a][:, None]
        rh, th_, eh = project(ph, B)
        rc = rh.clamp(min=0)
        okh = (rh >= 0) & B.munk[rc] & (th_ > B.tmin[rc]) & (th_ < B.tmax[rc]) & (th_ < B.tknown[rc] - tol(th_, eh))
        zH = torch.full((B.t.shape[1],), float("inf"), dtype=torch.float64, device=B.t.device)
        if okh.any():
            zH = zH.scatter_reduce(0, rc[okh], th_[okh], reduce="amin")
        H = zH
    for k in range(K):
        rem = torch.zeros(B.t.shape[1], dtype=torch.bool, device=B.t.device)
        nv = novB[k]
        if H is not None:
            rem |= nv & torch.isfinite(H) & (B.t[k] >= H - tol(H.clamp(max=1e6)))
        if mode in ("both", "free") and nv.any():
            b = torch.nonzero(nv)[:, 0]
            p2 = B.c[None] + B.d[b] * B.t[k, b][:, None]
            ra, tpa, ea = project(p2, A)
            rc = ra.clamp(min=0)
            ok = (ra >= 0) & A.munk[rc] & (tpa > A.tmin[rc]) & (tpa < A.tmax[rc]) & (tpa < A.tknown[rc] - tol(tpa, ea))
            fr_ = ok & (free_to[rc] > tpa + tol(tpa, ea))
            rem[b[fr_]] = True
        nrem += int(rem.sum())
        t2 = torch.where(rem, torch.full_like(B.t[k], float("inf")), B.t[k]) / B.kM
        out.append(INST["vc_dv2"](t2.to(B.v["d"].dtype), B.v, geo, z0, VCFG))
    return out, nrem


# ------------------------------------------------------------------------------------------------ the state
def rank_metrics(scores, keys, oracle, scene, start, t):
    """pick (planner rule m_score.choose over all legal), and with an oracle: Top-1 (within 1e-4 pp), regret, Spearman."""
    ch = MS.choose(list(scores), scene, start, t, True)
    pk = ch["pick"]
    if oracle is None:
        return {"pick": pk}
    g = np.asarray(oracle, float)
    ok = np.isfinite(g)
    if pk is None or not ok[pk]:
        return {"pick": pk, "top1": None, "regret": None, "gain": None, "rho": None}
    s = np.asarray(scores, float)[ok]
    rho = spearmanr(s, g[ok]).correlation if ok.sum() > 2 and np.ptp(s) > 0 and np.ptp(g[ok]) > 0 else None
    return {"pick": pk, "top1": bool(g[pk] >= np.nanmax(g) - 1e-4), "regret": float(np.nanmax(g) - g[pk]), "gain": float(g[pk]),
            "rho": None if rho is None or not np.isfinite(rho) else float(rho)}


def analyze_state(S):
    """S: core, geo-building pieces, h, eps8, z0, rec, scene, start, t, kM, fr, isect, gt (points runtime), gt_tree, vis, oracle (list or None),
    meta (dict), save (path or None)."""
    dev = INST["dev"]
    core, rec, z0, kM, fr = S["core"], S["rec"], S["z0"], S["kM"], S["fr"]
    geo = PC.Geometry.from_fusion(fr, S["dims"], S["dom"], S["fu"], dev)
    tree = HP.obs_tree()
    run4 = run_planner(core, geo, S["h"], rec, S["eps8"][:4], z0)
    run8 = run_planner(core, geo, S["h"], rec, S["eps8"], z0)
    keys1 = run4["keys1"]
    ch = {c1: children((c1,), geo) for c1 in keys1}
    allk = set(keys1) | {c for c1 in keys1 for c in ch[c1]}
    assert allk <= set(run4["V"]) and allk <= set(run8["V"]), (sorted(allk - set(run4["V"]))[:5], sorted(set(run4["V"]) - allk)[:5], len(allk), len(run4["V"]), len(run8["V"]))
    # regression: recorded sets -> the planner's two-step scores
    rec4 = two_step_scores(keys1, ch, run4["SETS"], range(4))
    pl4 = [float(x) for x in run4["out"]["distp1"]]
    reg4 = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(rec4, pl4))
    rec8 = two_step_scores(keys1, ch, run8["SETS"], range(8))
    reg8 = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(rec8, [float(x) for x in run8["out"]["distp1"]]))
    # K = 4 run vs the first 4 samples of the K = 8 run (same seeds)
    sub = two_step_scores(keys1, ch, run8["SETS"], range(4))
    k48 = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(sub, pl4))
    # view caches with truth (K = 4 run = the planner)
    VC = {}
    for key in allk:
        v = run4["V"][key]
        c = ViewC(key, v, run4["HT"][key], fr, kM, dev)
        tt = FX.gt_first_hit(S["isect"], c.c.cpu().numpy(), c.d.cpu().numpy(), float(c.tcap.max()))
        tt = torch.as_tensor(tt, device=dev)
        c.ttrue = torch.where(tt <= c.tcap, tt, torch.full_like(tt, float("inf")))
        VC[key] = c
    # projection self-test: points on a view's own rays must project back onto those rays
    c_ = VC[keys1[0]]
    rr, tp_, _ = project(c_.c[None] + c_.d * 10.0, c_)
    proj_ok = float((rr == torch.arange(len(rr), device=dev)).double().mean())
    assert proj_ok > 0.999, ("projection self-test", proj_ok)
    # truth per view: value function on GT depth (same units) + geometric coverage of the GT target points
    lo, hi = S["box"]
    covered0 = S["covered0"]
    TV, NC = {}, {}
    for key, c in VC.items():
        tm = (c.ttrue / kM).to(c.v["d"].dtype)
        TV[key] = VC0(tm, c.v, geo, z0, VCFG)
        fin = torch.isfinite(c.ttrue)
        p = (c.c[None] + c.d * torch.where(fin, c.ttrue, torch.zeros_like(c.ttrue))[:, None]).cpu().numpy()
        zz = (c.ttrue * c.dz).cpu().numpy()
        okp = fin.cpu().numpy() & (zz < 70.0) & np.all((p >= lo) & (p <= hi), 1)
        if okp.sum():
            dd, _ = cKDTree(p[okp]).query(S["gt"], k=1, distance_upper_bound=EPS, workers=8)
            NC[key] = np.isfinite(dd) & ~covered0
        else:
            NC[key] = np.zeros(len(S["gt"]), bool)
    norm = 100.0 / (len(S["gt"]) * S["vis"])
    # per-ray novelty of the recorded predictions (checked against the recorded sets)
    NOV, WNOV, KEY = {}, {}, {}
    chk = 0
    for key, c in VC.items():
        nv, ky, ww = [], [], []
        for k in range(4):
            n_, k_, w_ = novel_rays(c, k, geo, z0, tree)
            nv.append(n_), ky.append(k_), ww.append(w_)
            if chk < 6:
                ids = np.unique(k_[n_].cpu().numpy())
                assert np.array_equal(ids, np.sort(run4["SETS"][key][k][0])), ("novel-ray reconstruction differs", key, k)
                chk += 1
        NOV[key], KEY[key], WNOV[key] = torch.stack(nv), torch.stack(ky), torch.stack(ww)
    S4, S8 = run4["SETS"], run8["SETS"]
    V1 = {c1: [float(S8[c1][k][1].sum()) for k in range(8)] for c1 in keys1}
    oracle = S.get("oracle")
    geo1 = [float(NC[c1].sum()) * norm for c1 in keys1]
    geo2 = [max([float((NC[c1] | NC[c2]).sum()) * norm for c2 in ch[c1]] or [float(NC[c1].sum()) * norm]) for c1 in keys1]
    tv1 = [float(TV[c1][1].sum()) for c1 in keys1]
    tv2 = [max([float(DP.union_sets(TV[c1], TV[c2])[1].sum()) for c2 in ch[c1]] or [tv1[i]]) for i, c1 in enumerate(keys1)]
    st = {**S["meta"], "n_legal": len(keys1), "keys1": [list(k) for k in keys1], "children": {str(list(c1)): [list(c) for c in ch[c1]] for c1 in keys1},
          "n_views": len(allk), "seeds": S["seeds"], "regression": {"recorded_vs_planner_K4": reg4, "recorded_vs_planner_K8": reg8, "K8_first4_vs_K4": k48, "projection_self_test": proj_ok},
          "seconds": {"K4": run4["seconds"], "K8": run8["seconds"]},
          "oracle_one_step_pp": None if oracle is None else [None if not np.isfinite(x) else float(x) for x in oracle],
          "geo_one_step_pp": geo1, "geo_two_step_pp": geo2, "gtvalue_one_step": tv1, "gtvalue_two_step": tv2,
          "pred_one_step_k8": [V1[c1] for c1 in keys1], "planner_scores_K4": pl4}
    # ---------------- B: K subsets
    sc = {}
    for K_ in (1, 2, 4, 8):
        sc[f"two_step_first{K_}"] = two_step_scores(keys1, ch, S8, range(K_))
        sc[f"one_step_first{K_}"] = [float(np.mean(V1[c1][:K_])) for c1 in keys1]
    subsets = {1: [[k] for k in range(8)], 2: [[0, 1], [2, 3], [4, 5], [6, 7]], 4: [[0, 1, 2, 3], [4, 5, 6, 7]]}
    for K_, ss in subsets.items():
        sc[f"two_step_subsets{K_}"] = [two_step_scores(keys1, ch, S8, s_) for s_ in ss]
        sc[f"one_step_subsets{K_}"] = [[float(np.mean([V1[c1][k] for k in s_])) for c1 in keys1] for s_ in ss]
    # sample diversity / pose vs noise
    def wjac(a, b):
        u = DP.union_sets(a, b)[1].sum()
        i = a[1].sum() + b[1].sum() - u
        return float(i / u) if u > 0 else None
    jw, jb_same, jb_diff = [], [], []
    for c1 in keys1:
        js = [wjac(S8[c1][a], S8[c1][b]) for a in range(8) for b in range(a + 1, 8)]
        js = [x for x in js if x is not None]
        jw.append(float(np.mean(js)) if js else None)
    for i, a in enumerate(keys1):
        for b in keys1[i + 1:]:
            x = [wjac(S8[a][k], S8[b][k]) for k in range(8)]
            y = [wjac(S8[a][k], S8[b][(k + 1) % 8]) for k in range(8)]
            jb_same += [z for z in x if z is not None]
            jb_diff += [z for z in y if z is not None]
    Vm = np.asarray([V1[c1] for c1 in keys1])
    depth = []
    for c1 in keys1:                                  # K = 8 depth / hit spread on rays with UNKNOWN samples
        c = ViewC(c1, run8["V"][c1], run8["HT"][c1], fr, kM, dev)
        mu = c.munk
        hu = c.hu[:, mu].double()
        dis = float(((hu.mean(0) > 0) & (hu.mean(0) < 1)).double().mean()) if mu.any() else None
        tt = torch.where(c.hu[:, mu], c.t[:, mu], torch.full_like(c.t[:, mu], float("nan")))
        cnt = c.hu[:, mu].sum(0)
        m2 = cnt >= 2
        if m2.any():
            mean = torch.nanmean(tt[:, m2], 0)
            sd = torch.sqrt(torch.nanmean((tt[:, m2] - mean[None]) ** 2, 0))
            depth.append({"hit_disagreement": dis, "rel_depth_sd_median": float(torch.median(sd / mean)), "depth_sd_median": float(torch.median(sd))})
        else:
            depth.append({"hit_disagreement": dis, "rel_depth_sd_median": None, "depth_sd_median": None})
    rho = np.nanmean([x for x in jw if x is not None]) if any(x is not None for x in jw) else np.nan
    st["K_study"] = {"scores": sc, "within_cam_jaccard": jw, "between_cam_jaccard_same_k": float(np.mean(jb_same)) if jb_same else None,
                     "between_cam_jaccard_diff_k": float(np.mean(jb_diff)) if jb_diff else None,
                     "value_var_within": float(Vm.var(1, ddof=1).mean()), "value_var_between": float(Vm.mean(1).var(ddof=1)) if len(Vm) > 1 else None,
                     "K_eff_8": float(8.0 / (1.0 + 7.0 * rho)) if np.isfinite(rho) else None, "depth": depth}
    # ---------------- A + C: pairs
    def pair_row(a, b, kind):
        SA, SB = S4[a], S4[b]
        VA, VB = float(np.mean([SA[k][1].sum() for k in range(4)])), float(np.mean([SB[k][1].sum() for k in range(4)]))
        U = q_paired(SA, SB, range(4))
        Uu = q_unpaired(SA, SB, range(4))
        TA, TB = TV[a], TV[b]
        Ut = float(DP.union_sets(TA, TB)[1].sum())
        tVA, tVB = float(TA[1].sum()), float(TB[1].sum())
        NA, NB = NC[a], NC[b]
        nA, nB, nU, nI = int(NA.sum()), int(NB.sum()), int((NA | NB).sum()), int((NA & NB).sum())
        # double counting: B's predicted marginal (cells not in A's cells, same k) on rays whose TRUE hit is a truly co-visible novel surface of A
        A_, B_ = VC[a], VC[b]
        fa = torch.isfinite(A_.ttrue)
        pa = (A_.c[None] + A_.d * torch.where(fa, A_.ttrue, torch.zeros_like(A_.ttrue))[:, None])[fa].cpu().numpy()
        fb = torch.isfinite(B_.ttrue)
        pb = (B_.c[None] + B_.d * torch.where(fb, B_.ttrue, torch.zeros_like(B_.ttrue))[:, None]).cpu().numpy()
        dup = np.zeros(len(pb), bool)
        if len(pa):
            dd, _ = cKDTree(pa).query(pb, k=1, distance_upper_bound=EPS)
            dup = np.isfinite(dd) & fb.cpu().numpy()
        dup_t = torch.as_tensor(dup, device=dev)
        marg_w = marg_dup = 0.0
        for k in range(4):
            idsA = torch.as_tensor(np.asarray(SA[k][0], np.int64), device=dev)
            inA = torch.isin(KEY[b][k], idsA)
            m = NOV[b][k] & ~inA
            marg_w += float(WNOV[b][k][m].sum()) / 4
            marg_dup += float(WNOV[b][k][m & dup_t].sum()) / 4
        cAB = consistency(VC[a], VC[b], NOV[a], WNOV[a] + (~NOV[a]).double() * torch.clamp((VC[a].t * VC[a].dz / z0) ** 2, max=1.0).nan_to_num(0.0), 4)
        cBA = consistency(VC[b], VC[a], NOV[b], WNOV[b] + (~NOV[b]).double() * torch.clamp((VC[b].t * VC[b].dz / z0) ** 2, max=1.0).nan_to_num(0.0), 4)
        return {"kind": kind, "a": list(a), "b": list(b), "pred": {"VA": VA, "VB": VB, "U": U, "U_unpaired": Uu, "I": VA + VB - U},
                "gtv": {"VA": tVA, "VB": tVB, "U": Ut, "I": tVA + tVB - Ut},
                "cov": {"nA": nA, "nB": nB, "nU": nU, "nI": nI, "pp_U": nU * norm, "pp_marg_B": (nU - nA) * norm},
                "marg_B_pred": marg_w, "marg_B_on_true_duplicate": marg_dup, "cons_AB": cAB, "cons_BA": cBA}
    pairs = []
    for i, a in enumerate(keys1):
        for b in keys1[i + 1:]:
            pairs.append(pair_row(a, b, "first"))
    for a in keys1:
        for b in ch[a]:
            pairs.append(pair_row(a, b, "path"))
    st["pairs"] = pairs
    # ---------------- D: CVAR (K = 4, the planner's samples)
    var = {}
    for mode in ("both", "occ", "free"):
        s2, nrem = {}, 0
        for a in keys1:
            for b in ch[a]:
                s2[(a, b)], n_ = cvar_sets(VC[a], VC[b], 4, geo, z0, NOV[b], mode)
                nrem += n_
        var[mode] = (two_step_scores(keys1, ch, S4, range(4), sets2=s2), nrem)
    sc4 = {"dv2": pl4, "dv2_recomputed": rec4, "unpaired": two_step_scores(keys1, ch, S4, range(4), pair=q_unpaired),
           "cvar": var["both"][0], "cvar_occ": var["occ"][0], "cvar_free": var["free"][0],
           "one_step": [float(np.mean([S4[c1][k][1].sum() for k in range(4)])) for c1 in keys1]}
    st["D"] = {"scores": sc4, "removed_rays": {m: var[m][1] for m in var}}
    # ranking metrics for every score list
    oracles = {"official": oracle, "geo2": geo2, "geo1": geo1, "gtv2": tv2}
    rk = {}
    for name, s_ in list(sc4.items()) + [(k_, v_) for k_, v_ in sc.items() if not k_.endswith(tuple("0123456789")) or "first" in k_]:
        if isinstance(s_[0], list):
            continue
        rk[name] = {o: rank_metrics(s_, keys1, ov, S["meta"]["scene"], S["meta"]["start"], S["meta"]["t"]) for o, ov in oracles.items()}
    for K_, ss in subsets.items():
        for kind in ("two_step", "one_step"):
            rk[f"{kind}_subsets{K_}"] = {o: [rank_metrics(s_, keys1, ov, S["meta"]["scene"], S["meta"]["start"], S["meta"]["t"]) for s_ in sc[f"{kind}_subsets{K_}"]]
                                         for o, ov in oracles.items()}
    st["rank"] = rk
    if S.get("save"):
        np.savez_compressed(S["save"], keys=np.asarray([list(k) for k in allk]),
                            **{f"t_{'_'.join(map(str, k))}": VC[k].t.float().cpu().numpy() for k in allk},
                            **{f"t8_{'_'.join(map(str, k))}": (run8["HT"][k]["t"].double() * kM).float().cpu().numpy() for k in keys1},
                            **{f"hu_{'_'.join(map(str, k))}": VC[k].hu.cpu().numpy() for k in allk},
                            **{f"nov_{'_'.join(map(str, k))}": NOV[k].cpu().numpy() for k in allk},
                            **{f"tt_{'_'.join(map(str, k))}": VC[k].ttrue.float().cpu().numpy() for k in allk},
                            **{f"c_{'_'.join(map(str, k))}": VC[k].c.cpu().numpy() for k in allk},
                            **{f"d_{'_'.join(map(str, k))}": VC[k].d.float().cpu().numpy() for k in allk},
                            **{f"munk_{'_'.join(map(str, k))}": VC[k].munk.cpu().numpy() for k in allk},
                            **{f"N_{'_'.join(map(str, k))}": np.flatnonzero(NC[k]) for k in allk},
                            covered0=np.flatnonzero(covered0), gt=S["gt"].astype(np.float32),
                            band=np.argwhere(geo.caster.obs_band.cpu().numpy()).astype(np.int16), origin=fr.origin_z, kM=kM)
    del run4, run8, VC, geo
    torch.cuda.empty_cache()
    return st
