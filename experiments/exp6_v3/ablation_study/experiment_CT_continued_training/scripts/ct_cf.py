"""Experiment CT / CT3 -- decision counterfactuals (seen2scene env, 1 GPU; offline, GT only to score). For one checkpoint, on the same fixed states, the planner's
score is recomputed after removing one error type from the K = 4 decoded samples, label by label (labels = encode_labels of t* on the state's D100 map):
  pred         the samples as decoded (must equal the planner: regression)
  no_fp        false-positive hits (predicted hit, label NOHIT / KNOWN) -> the projection outcome t_known
  no_fn        missed hits (label HIT, sample has no hit) -> the expressible true hit
  hit_oracle   no_fp + no_fn (HIT / NOHIT error removed; predicted depth kept on true positives)
  depth_oracle true positives moved to the expressible true hit (classification errors kept)
  perfect      every labelled ray = expressible true depth (encode_labels -> decode); rays without a label keep the samples
Rays without a target (BLOCKED, NEAR, INF_*, no UNKNOWN samples) are never changed, except that `perfect_all` (decode of the labels on every ray, the Experiment F 'gated' reference) is also computed.
Scores: outdoor = D-v2 two-step on first-layer views and the planner's children (t* of every child from the GT mesh); indoor = H0 DistP1.
Output work_dirs/.../states/<group>/<tag>.jsonl. Usage: python ct_cf.py --set indoor|explore|confirm --group G --ckpt NAME [--only TAG] [--tmax T] [--out DIR]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ct_fixed as CF  # noqa: E402

C, np, torch, FX, G, SCO, PC, HP = CF.C, CF.np, CF.torch, CF.FX, CF.G, CF.SCO, CF.PC, CF.HP
VARS = ("pred", "no_fp", "no_fn", "hit_oracle", "depth_oracle", "perfect", "perfect_all")


@torch.no_grad()
def variants(tk, hu, ts, v, geo, kM):
    """-> {name: t [K, R]} model units."""
    pr = v["pr"]
    dz = v["dz"].to(tk.dtype)
    lab = G.encode_labels(v["o"], v["d"], pr, ts, dz, kM, geo.struct, geo.domain)
    tg, _ = G.decode(lab["b"][None].to(ts.dtype), lab["y"][None].to(ts.dtype), v["o"], v["d"], pr, geo.struct, geo.domain)
    tg = tg[0].to(tk.dtype)
    tk0 = pr["t_known"].to(tk.dtype)
    lm = (pr["n_u"] > 0) & (lab["b"] != 0)
    lh = lm & (lab["cat"] == G.CAT["HIT"])
    ln = lm & ~lh
    K = tk.shape[0]
    fp, fn, tp = hu & ln[None], ~hu & lh[None], hu & lh[None]
    out = {"pred": tk, "no_fp": torch.where(fp, tk0[None].expand(K, -1), tk), "no_fn": torch.where(fn, tg[None].expand(K, -1), tk)}
    out["hit_oracle"] = torch.where(fn, tg[None].expand(K, -1), out["no_fp"])
    out["depth_oracle"] = torch.where(tp, tg[None].expand(K, -1), tk)
    out["perfect"] = torch.where(lm[None], tg[None].expand(K, -1), tk)
    out["perfect_all"] = tg[None].expand(K, -1)
    return out


def run_outdoor(a, tag, kind, spec, core, dev):
    P = CF.FR.spec_of(None, tag, kind, spec, dev)
    Frows = {r["t"]: r for r in map(json.loads, open(os.path.join(CF.FW, f"{tag}.jsonl")))}
    targets = sorted(Frows) if a.tmax is None else [t for t in sorted(Frows) if t <= a.tmax]
    scene, (fr, dims, dom) = P["scene"], P["frame"]
    C.set_scene(scene)
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    kM, z0 = float(fr.k_M), P["z0"]
    isect = FX.gt_mesh(scene)
    vc = C.INST["vc_dv2"]
    h, eps4, out = None, None, []
    for t in range(max(targets) + 1):
        rec = C.MC.load_service_record(P["svc"](t))
        fu.integrate_record(rec)
        if t % 10 == 0:
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            seeds = [int(P["seed"](t, kk)) for kk in range(4)]
            eps4 = C.PR.noise(dev, 4, lambda kk: seeds[kk])
            del geo
        if t not in targets:
            continue
        geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        tb = time.time()
        run = C.run_planner(core, geo, h, rec, eps4, z0)
        keys1 = run["keys1"]
        ch = {c1: C.children((c1,), geo) for c1 in keys1}
        allk = list(dict.fromkeys(list(keys1) + [b for a_ in keys1 for b in ch[a_]]))
        sets = {}
        for key in allk:
            v, ht = run["V"][key], run["HT"][key]
            Vc = C.ViewC(key, v, ht, fr, kM, dev)
            tt = torch.as_tensor(FX.gt_first_hit(isect, Vc.c.cpu().numpy(), Vc.d.cpu().numpy(), float(Vc.tcap.max())), device=dev)
            ts = (torch.where(tt <= Vc.tcap, tt, torch.full_like(tt, float("inf"))) / kM).to(v["d"].dtype)
            var = variants(ht["t"], ht["hit_u"], ts, v, geo, kM)
            sets[key] = {n: [vc(tv[k], v, geo, z0, C.VCFG) for k in range(tv.shape[0])] for n, tv in var.items()}
        res = {}
        for n in CF_VARS:
            V1 = [float(np.mean([sets[c][n][k][1].sum() for k in range(4)])) for c in keys1]
            two = []
            for i, c1 in enumerate(keys1):
                best = None
                for c2 in ch[c1]:
                    q = float(np.mean([C.DP.union_sets(sets[c1][n][k], sets[c2][n][k])[1].sum() for k in range(4)]))
                    best = q if best is None or q > best else best
                two.append(best if best is not None else V1[i])
            res[n] = {"one": V1, "two": two}
        pl = [float(x) for x in run["out"]["distp1"]]
        reg = max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(res["pred"]["two"], pl))
        out.append({"domain": "outdoor", "set": a.set, "tag": tag, "scene": scene, "start": P["meta"]["start"], "t": t, "ckpt": a.ckpt, "keys1": [list(k) for k in keys1], "cf": res, "regression_pred_vs_planner": reg, "seconds": time.time() - tb})
        print(f"{tag} t={t} reg {reg:.1e} {time.time() - tb:.1f}s", flush=True)
        assert reg < 1e-9, (tag, t, reg)
        del geo, run, sets
    del fu
    torch.cuda.empty_cache()
    return out


def run_indoor(a, tag, sts, core, dev):
    import e2_fixed as E2F
    rp = E2F.Rep("valsel", sts[0], dev)
    by = {s["t"]: s for s in sts}
    targets = sorted(by) if a.tmax is None else [t for t in sorted(by) if t <= a.tmax]
    recs = [C.MC.load_service_record(rp.svc(t)) for t in range(max(targets) + 1)]
    out = []
    for r in sorted({t - t % 10 for t in targets}):
        region = rp.region_of(recs[r])
        fus = FX.m_p1.P1Fusion(*region, device=dev)
        for q in range(r + 1):
            fus.integrate_record(recs[q])
        geo = PC.Geometry.from_fusion(*region, fus, dev)
        h = core.pred.encode(geo.coarse())
        eps = C.PR.noise(dev, 4, lambda kk, tr=r: C.RY.sha_seed(rp.key(tr, kk)))
        del geo
        for t in range(r, max(t_ for t_ in targets if t_ - t_ % 10 == r) + 1):
            if t > r:
                fus.integrate_record(recs[t])
            if t not in targets:
                continue
            st, rec = by[t], recs[t]
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            kM = float(geo.fr.k_M)
            orc = np.load(os.path.join(E2F.EP.W, "valsel", "states", "indoor", st["sid"], "oracle.npz"))
            tb = time.time()
            o2, _, views = CF.FROZEN_SCORE(core, geo, h, rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
            cand = [int(x) for x in st["cand"]]
            res = {n: {"one": [None] * len(views)} for n in CF_VARS}
            for j in cand:
                v, ht = views[j], o2["hits"][j]
                ts = torch.as_tensor(orc["tstar"][j], device=dev).to(v["d"].dtype)
                var = variants(ht["t"], ht["hit_u"], ts, v, geo, kM)
                for n, tv in var.items():
                    res[n]["one"][j] = float(np.mean([float(SCO.new_set_from_depth(tv[k], v["o"], v["d"], v["dz"], kM, geo.dims, geo.domain, geo.caster.band_dil, rp.z0)[1].sum()) for k in range(tv.shape[0])]))
            pl = [float(x) for x in o2["distp1"]]
            reg = max(abs(res["pred"]["one"][j] - pl[j]) / max(1.0, abs(pl[j])) for j in cand)
            out.append({"domain": "indoor", "set": "indoor", "tag": tag, "scene": st["scene"], "start": st["start"], "t": t, "sid": st["sid"], "ckpt": a.ckpt, "cand": cand, "cf": res, "regression_pred_vs_planner": reg, "seconds": time.time() - tb})
            print(f"{tag} t={t} reg {reg:.1e} {time.time() - tb:.1f}s", flush=True)
            assert reg < 1e-9, (tag, t, reg)
            del geo
        del fus
        torch.cuda.empty_cache()
    return out


CF_VARS = VARS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("indoor", "explore", "confirm"))
    ap.add_argument("--group", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    od = a.out or os.path.join(CF.W, "states", a.group)
    os.makedirs(od, exist_ok=True)
    claims = os.path.join(CF.W, "claims", a.group)
    os.makedirs(claims, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    if a.set == "indoor":
        sys.path.insert(0, os.path.join(CF.R, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
        import e2_fixed as E2F
        E2F.X.setup("B0")
        J = [(tag, ("indoor", sts)) for tag, sts in E2F.states("valsel", "B0")]
    else:
        J = CF.FR.jobs(argparse.Namespace(set=a.set))
    if a.only:
        J = [j for j in J if j[0] == a.only]
    if not J:
        return
    if a.set != "indoor":
        kind0, spec0 = J[0][1]
        C.install(spec0[0] if kind0 in ("e3", "conf") else spec0[2][0]["scene"], dev)
    core = FX.PV3.V3Core(CF.ckpt_path(a.ckpt), dev, expect_sha=CF.CFG["checkpoints"][a.ckpt].get("sha256"))
    for tag, (kind, spec) in J:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po) and not a.only:
            continue
        if not a.only:
            try:
                fd = os.open(os.path.join(claims, f"{tag}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
        tb = time.time()
        rows = run_indoor(a, tag, spec, core, dev) if kind == "indoor" else run_outdoor(a, tag, kind, spec, core, dev)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {tag} rows {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
