"""Experiment CT -- fixed-state prediction + decision evaluation of V3 checkpoints (seen2scene env, 1 GPU per process; offline, GT only to score).
Sets (configs/CT_config.yaml): indoor = MP3D VALSEL 90 states (frozen indoor planner path: H0 DistP1, K 4, Euler 10, R10, local P1 n_win 5);
explore / confirm = the Experiment F outdoor pool (D-v2 two-step on the global P1), Test-3 never included. All checkpoints of one process see the
same replayed state, the same noise and the same candidates. Per state and checkpoint: per-camera confusion on labelled UNKNOWN rays (labels =
geometry.encode_labels of t* on the state's own D100 map, the training rule), depth errors in runtime z-depth, novelty confusion, K = 4 diversity,
per-sample planner values. Truth: outdoor from the Experiment F rows (official gain, geometric truth, children), indoor from oracle.npz.
Output work_dirs/.../experiment_CT_continued_training/states/<group>/<tag>.jsonl (written per trajectory; claims; resumable).
Usage: python ct_fixed.py --set indoor|explore|confirm --group G --ckpts NAME,NAME [--only TAG] [--tmax T] [--out DIR]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
FD = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(FD))))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_F_stochastic_coverage_decision", "scripts"))
import f_run as FR  # noqa: E402

C = FR.C
np, torch, FX = C.np, C.torch, C.FX
HP, PC = C.HP, C.PC
FROZEN_SCORE = PC.score_state                   # before any policy install: the frozen H0 path
from seen2scene.futureview import geometry as G  # noqa: E402
from seen2scene.futureview import scoring as SCO  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_CT_continued_training")
FW = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_F_stochastic_coverage_decision", "states")
CFG = C.yaml.safe_load(open(os.path.join(FD, "configs", "CT_config.yaml")))
EDGES = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf")]       # runtime z-depth (MP3D: 7 units = 1 m)
VS = 0.011
CAT_NAMES = {v: k for k, v in G.CAT.items()}


def ckpt_path(name):
    return os.path.join(R, CFG["checkpoints"][name]["path"])


def _i(x):
    return int(x.item()) if torch.is_tensor(x) else int(x)


def _f(x):
    return float(x.item()) if torch.is_tensor(x) else float(x)


@torch.no_grad()
def camera_metrics(tk, hu, ts, v, geo, kM, new_pred, new_true, eps_cov):
    """tk, hu [K, R] of the decoded samples (model units, inf = no hit; hit_u), ts [R] t* (model units, inf = none); new_pred [K, R], new_true [R]."""
    pr = v["pr"]
    dz = v["dz"].to(tk.dtype)
    lab = G.encode_labels(v["o"], v["d"], pr, ts, dz, kM, geo.struct, geo.domain)
    cat, b = lab["cat"], lab["b"]
    unk = pr["n_u"] > 0
    lm = unk & (b != 0)
    lh = lm & (cat == G.CAT["HIT"])
    ln = lm & ~lh
    K = tk.shape[0]
    p = hu
    tp_m = p & lh[None]
    o = {"K": K, "n_rays": int(cat.numel()), "n_unk": _i(unk.sum()), "n_lm": _i(lm.sum()), "n_lh": _i(lh.sum()),
         "tp": _i(tp_m.sum()), "fp": _i((p & ln[None]).sum()), "fn": _i((~p & lh[None]).sum()), "tn": _i((~p & ln[None]).sum()),
         "fp_nohit": _i((p & (lm & (cat == G.CAT["NOHIT"]))[None]).sum()), "fp_known": _i((p & (lm & (cat == G.CAT["KNOWN"]))[None]).sum()),
         "pred_hit_on_nolabel": _i((p & (unk & ~lm)[None]).sum())}
    pm = p.float().mean(0)
    o["brier"] = _f(((pm - lh.float()) ** 2)[lm].sum()) if lm.any() else 0.0
    zt = ts.double() * kM * dz.double()
    zh = tk.double() * kM * dz.double()
    # depth localisation on true positives (each sample)
    e = (zh - zt[None]).abs()
    o["sae_tp"] = _f(e[tp_m].sum()) if tp_m.any() else 0.0
    o["sse_tp"] = _f((zh - zt[None])[tp_m].sum()) if tp_m.any() else 0.0
    o["sre_tp"] = _f((e / zt.clamp(min=1e-6)[None])[tp_m].sum()) if tp_m.any() else 0.0
    vals = e[tp_m]
    o["hist_tp"] = [_i(((vals >= EDGES[i]) & (vals < EDGES[i + 1])).sum()) for i in range(len(EDGES) - 1)]
    # first-hit MAE (decoded depth, finite truth within range)
    ok = torch.isfinite(zt) & (zt < 70.0)
    ph = torch.isfinite(zh)
    both = ph & ok[None]
    ez = torch.where(both, e, torch.zeros_like(e))
    o["sae_all"], o["n_all"] = _f(ez.sum()), _i(both.sum())
    bu = both & unk[None]
    o["sae_unk"], o["n_mae_unk"] = _f(ez[bu].sum()) if bu.any() else 0.0, _i(bu.sum())
    o["hit_agree"] = _i((ph == ok[None]).sum())
    # K-sample diversity on labelled UNKNOWN rays
    alh = p.all(0) & lm
    anh = p.any(0) & lm
    o["n_anyhit"], o["n_allhit"], o["n_disagree"] = _i(anh.sum()), _i(alh.sum()), _i((anh & ~alh).sum())
    o["pair_sum"] = o["rel_std_sum"] = 0.0
    if alh.any() and K >= 2:
        zz = zh[:, alh]
        pd = torch.stack([(zz[i] - zz[j]).abs() for i in range(K) for j in range(i + 1, K)]).mean(0)
        o["pair_sum"] = _f(pd.sum())
        o["rel_std_sum"] = _f((zz.std(0, unbiased=False) / zz.mean(0).clamp(min=1e-6)).sum())
    m = alh & lh
    o["n_iv"], o["iv_in"] = _i(m.sum()), 0
    o["rho_spread_err"] = None
    if m.any():
        zz, zg = zh[:, m], zt[m]
        tol = (VS * kM * dz.double())[m]
        o["iv_in"] = _i(((zg >= zz.min(0).values - tol) & (zg <= zz.max(0).values + tol)).sum())
        if int(m.sum()) >= 10 and K >= 2:
            sp = zz.std(0, unbiased=False).cpu().numpy()
            er = (zz.mean(0) - zg).abs().cpu().numpy()
            if np.ptp(sp) > 0 and np.ptp(er) > 0:
                r = spearmanr(sp, er).correlation
                o["rho_spread_err"] = float(r) if np.isfinite(r) else None
    a = anh & lh
    o["n_best"], o["sae_best"], o["sae_mean"], o["within_best"], o["within_mean"] = _i(a.sum()), 0.0, 0.0, 0, 0
    if a.any():
        ee = e[:, a].clone()
        pp = p[:, a]
        ee[~pp] = float("inf")
        best = ee.min(0).values
        mean = torch.where(pp, e[:, a], torch.zeros_like(e[:, a])).sum(0) / pp.sum(0).clamp(min=1)
        o["sae_best"], o["sae_mean"] = _f(best.sum()), _f(mean.sum())
        o["within_best"], o["within_mean"] = _i((best <= eps_cov).sum()), _i((mean <= eps_cov).sum())
    # novelty confusion (the domain's scoring rule) over all rays
    o["new_tp"] = _i((new_pred & new_true[None]).sum())
    o["new_fp"] = _i((new_pred & ~new_true[None]).sum())
    o["new_fn"] = _i((~new_pred & new_true[None]).sum())
    o["n_true_new"] = _i(new_true.sum())
    # label categories over all rays and over UNKNOWN rays
    o["cat_all"] = {CAT_NAMES[i]: _i((cat == i).sum()) for i in CAT_NAMES}
    o["cat_unk"] = {CAT_NAMES[i]: _i(((cat == i) & unk).sum()) for i in CAT_NAMES}
    return o


def in_grid(ts, v, geo):
    p = v["o"][None] + v["d"] * torch.where(torch.isfinite(ts), ts, torch.zeros_like(ts))[:, None]
    ijk = torch.floor(p / VS).long()
    dims = torch.as_tensor(geo.dims, device=ijk.device)
    return torch.isfinite(ts) & ((ijk >= 0) & (ijk < dims)).all(1)


def outdoor_truth_new(ts, v, geo, tree, kM):
    """Truly new under the D-v2 value rule: in the grid and the domain, farther than eps from every observed S10 point."""
    ok = in_grid(ts, v, geo)
    out = torch.zeros_like(ok)
    idx = torch.nonzero(ok)[:, 0]
    if idx.numel():
        p = v["o"][None] + v["d"][idx] * ts[idx][:, None]
        ijk = torch.floor(p / VS).long()
        dm = geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        pr = geo.fr.model_to_runtime(p.double().cpu().numpy())
        dd, _ = tree.query(pr, k=1, distance_upper_bound=C.EPS)
        out[idx] = dm & torch.as_tensor(~np.isfinite(dd), device=dm.device)
    return out


def outdoor_state(st_meta, P, geo, rec, h, eps4, cores, Frow, tcache, isect, tree, fr, kM):
    rows = []
    z0 = P["z0"]
    for name, core in cores.items():
        t0 = C.sync()
        run = C.run_planner(core, geo, h[name], rec, eps4, z0)
        t_plan = C.sync() - t0
        keys1 = run["keys1"]
        ch = {c1: C.children((c1,), geo) for c1 in keys1}
        paths = [(a, b) for a in keys1 for b in ch[a]]
        S = run["SETS"]
        V1k = [[float(S[c1][k][1].sum()) for k in range(4)] for c1 in keys1]
        V12k = [[float(C.DP.union_sets(S[a][k], S[b][k])[1].sum()) for k in range(4)] for a, b in paths]
        cams = []
        for j, key in enumerate(keys1):
            v, ht = run["V"][key], run["HT"][key]
            Vc = C.ViewC(key, v, ht, fr, kM, C.INST["dev"])
            if key not in tcache:
                tt = torch.as_tensor(FX.gt_first_hit(isect, Vc.c.cpu().numpy(), Vc.d.cpu().numpy(), float(Vc.tcap.max())), device=Vc.tcap.device)
                tcache[key] = (torch.where(tt <= Vc.tcap, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
            ts = tcache[key]
            new_true = outdoor_truth_new(ts, v, geo, tree, kM)
            new_pred = torch.stack([C.novel_rays(Vc, k, geo, z0, tree)[0] for k in range(4)])
            cams.append(camera_metrics(ht["t"], ht["hit_u"], ts, v, geo, kM, new_pred, new_true, 1.0))
        pl = [float(x) for x in run["out"]["distp1"]]
        row = {**st_meta, "ckpt": name, "ckpt_sha": core.ckpt_sha, "keys1": [list(k) for k in keys1], "paths": [[list(a), list(b)] for a, b in paths],
               "score": pl, "V1k": V1k, "V12k": V12k, "h0k": run["out"].get("distp1_k"), "h0": run["out"].get("h0_distp1"), "cams": cams,
               "seconds_plan": t_plan, "predicted_views": run["out"]["d"]["predicted_views"], "peak_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
        reg = {"paths_eq_F": row["paths"] == Frow["paths"], "keys_eq_F": row["keys1"] == Frow["keys1"]}
        if name == "A_40K":
            reg["scores_bitwise_F"] = pl == Frow["planner_K4"]
            reg["V12_bitwise_F"] = V12k == [list(x) for x in Frow["hard"]["K4"]["V12"]]
        row["regression"] = reg
        rows.append(row)
    return rows


def run_outdoor(a, tag, kind, spec, cores, dev):
    P = FR.spec_of(None, tag, kind, spec, dev)
    Frows = {r["t"]: r for r in map(json.loads, open(os.path.join(FW, f"{tag}.jsonl")))}
    targets = sorted(Frows) if a.tmax is None else [t for t in sorted(Frows) if t <= a.tmax]
    scene = P["scene"]
    fr, dims, dom = P["frame"]
    C.set_scene(scene)
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    kM = float(fr.k_M)
    isect = FX.gt_mesh(scene)
    h, eps4 = {}, None
    out = []
    for t in range(max(targets) + 1):
        rec = C.MC.load_service_record(P["svc"](t))
        fu.integrate_record(rec)
        if t % 10 == 0:
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            coarse = geo.coarse()
            h = {n: c.pred.encode(coarse) for n, c in cores.items()}
            seeds = [int(P["seed"](t, kk)) for kk in range(4)]
            eps4 = C.PR.noise(dev, 4, lambda kk: seeds[kk])
            del geo, coarse
        if t not in targets:
            continue
        geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        tree = HP.obs_tree()
        torch.cuda.reset_peak_memory_stats(dev)
        tb = time.time()
        meta = {"domain": "outdoor", "set": a.set, "tag": tag, "scene": scene, "start": P["meta"]["start"], "t": t, "src": P["meta"]["src"], "stratum": P["meta"].get("stratum")}
        rows = outdoor_state(meta, P, geo, rec, h, eps4, cores, Frows[t], {}, isect, tree, fr, kM)
        out += rows
        print(f"{tag} t={t} " + " | ".join(f"{r['ckpt']} pick {int(np.argmax(r['score']))} reg {r['regression']}" for r in rows) + f" | {time.time() - tb:.1f}s {rows[-1]['peak_gb']:.1f}GB", flush=True)
        del geo
    del fu
    torch.cuda.empty_cache()
    return out


def run_indoor(a, tag, sts, cores, dev):
    import e2_fixed as E2F
    rp = E2F.Rep("valsel", sts[0], dev)
    by = {s["t"]: s for s in sts}
    targets = sorted(by) if a.tmax is None else [t for t in sorted(by) if t <= a.tmax]
    recs = [C.MC.load_service_record(rp.svc(t)) for t in range(max(targets) + 1)]
    ref = {}
    pr_ = os.path.join(R, "work_dirs", "exp6_v3", "eval_fixed", "valsel_s_final", "B0", f"{tag}.jsonl")
    if os.path.exists(pr_):
        ref = {r["sid"]: r["scores"]["T1"] for r in map(json.loads, open(pr_))}
    out = []
    for r in sorted({t - t % 10 for t in targets}):
        region = rp.region_of(recs[r])
        fus = FX.m_p1.P1Fusion(*region, device=dev)
        for q in range(r + 1):
            fus.integrate_record(recs[q])
        geo = PC.Geometry.from_fusion(*region, fus, dev)
        coarse = geo.coarse()
        h = {n: c.pred.encode(coarse) for n, c in cores.items()}
        eps = C.PR.noise(dev, 4, lambda kk, tr=r: C.RY.sha_seed(rp.key(tr, kk)))
        del geo, coarse
        last = max(t for t in targets if t - t % 10 == r)
        for t in range(r, last + 1):
            if t > r:
                fus.integrate_record(recs[t])
            if t not in targets:
                continue
            st, rec = by[t], recs[t]
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            kM = float(geo.fr.k_M)
            orc = np.load(os.path.join(E2F.EP.W, "valsel", "states", "indoor", st["sid"], "oracle.npz"))
            torch.cuda.reset_peak_memory_stats(dev)
            tb = time.time()
            cand = [int(x) for x in st["cand"]]
            gain = [None] * len(rec["legal_R"])
            for j in cand:
                gain[j] = float(orc["gain"][j]) * 100.0
            eps_cov = 0.08 * float(orc["k_M"])
            ideal = {}
            for name, core in cores.items():
                t0 = C.sync()
                o2, _, views = FROZEN_SCORE(core, geo, h[name], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
                t_plan = C.sync() - t0
                cams = {}
                for j in cand:
                    v, ht = views[j], o2["hits"][j]
                    ts = torch.as_tensor(orc["tstar"][j], device=dev).to(v["d"].dtype)
                    dz = v["dz"].to(ts.dtype)
                    igr = in_grid(ts, v, geo)
                    tin = torch.where(igr, ts, torch.full_like(ts, float("inf")))
                    new_true = SCO.new_mask_from_depth(tin, v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)[0]
                    new_pred = torch.stack([SCO.new_mask_from_depth(ht["t"][k], v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)[0] for k in range(4)])
                    cams[j] = camera_metrics(ht["t"], ht["hit_u"], ts, v, geo, kM, new_pred, new_true, eps_cov)
                    if j not in ideal:
                        lab = G.encode_labels(v["o"], v["d"], v["pr"], ts, dz, kM, geo.struct, geo.domain)
                        tg, _ = G.decode(lab["b"][None].to(ts.dtype), lab["y"][None].to(ts.dtype), v["o"], v["d"], v["pr"], geo.struct, geo.domain)
                        ideal[j] = float(SCO.new_set_from_depth(tg[0], v["o"], v["d"], v["dz"], kM, geo.dims, geo.domain, geo.caster.band_dil, rp.z0)[1].sum())
                row = {"domain": "indoor", "set": "indoor", "tag": tag, "scene": st["scene"], "start": st["start"], "t": t, "sid": st["sid"], "ckpt": name, "ckpt_sha": core.ckpt_sha,
                       "n_legal": len(gain), "cand": cand, "gain_pp": gain, "score": [float(x) for x in o2["distp1"]], "h0k": o2["distp1_k"], "cams": {str(k): v_ for k, v_ in cams.items()},
                       "ideal_h0": {str(k): v_ for k, v_ in ideal.items()}, "seconds_plan": t_plan, "peak_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "regression": {}}
                if name == "A_40K" and st["sid"] in ref:
                    a_, b_ = np.asarray(o2["distp1_k"], float).ravel(), np.asarray(ref[st["sid"]], float).ravel()
                    row["regression"] = {"bitwise_eval_fixed": bool(np.array_equal(a_, b_)), "maxrel_eval_fixed": float(np.max(np.abs(a_ - b_) / np.maximum(np.abs(b_), 1e-12))) if a_.shape == b_.shape else None,
                                         "same_argmax": int(np.argmax(np.asarray(o2["distp1_k"]).mean(1))) == int(np.argmax(np.asarray(ref[st["sid"]]).mean(1)))}
                out.append(row)
            print(f"{tag} t={t} n={len(gain)} " + " | ".join(f"{x['ckpt']} reg {x['regression']}" for x in out[-len(cores):]) + f" | {time.time() - tb:.1f}s {out[-1]['peak_gb']:.1f}GB", flush=True)
            del geo
        del fus
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("indoor", "explore", "confirm"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(W, "states", a.group)
    os.makedirs(od, exist_ok=True)
    claims = os.path.join(W, "claims", a.group)
    os.makedirs(claims, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    if a.set == "indoor":
        sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
        import e2_fixed as E2F
        E2F.X.setup("B0")
        J = [(f"{tag}", ("indoor", sts)) for tag, sts in E2F.states("valsel", "B0")]
    else:
        J = FR.jobs(argparse.Namespace(set=a.set))
    if a.only:
        J = [j for j in J if j[0] == a.only]
    if not J:
        return
    names = a.ckpts.split(",")
    if a.set != "indoor":
        kind0, spec0 = J[0][1]
        C.install(spec0[0] if kind0 in ("e3", "conf") else spec0[2][0]["scene"], dev)
    cores = {n: FX.PV3.V3Core(ckpt_path(n), dev, expect_sha=CFG["checkpoints"][n].get("sha256")) for n in names}
    for tag, (kind, spec) in J:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po) and not a.only:
            continue
        if not a.only:
            try:
                fd = os.open(os.path.join(claims, f"{tag}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname()[1]} {os.environ.get('CUDA_VISIBLE_DEVICES')} {time.strftime('%FT%T')}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue
        tb = time.time()
        rows = run_indoor(a, tag, spec, cores, dev) if kind == "indoor" else run_outdoor(a, tag, kind, spec, cores, dev)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {tag} rows {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
