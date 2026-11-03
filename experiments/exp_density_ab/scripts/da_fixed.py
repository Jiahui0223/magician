"""EXP Density A/B -- Tasks 1-2: fixed-state 3 checkpoints x 2 inference densities x 2 scoring densities (seen2scene env,
1 GPU per process; FROZEN_PROTOCOL.md §3).

Checkpoints: B2 = V2-Mix EMA@17.5k (the frozen core of the Task 2 V2M group), E2S = E2-S10 EMA@5k, E2D = E2-D100 EMA@5k.
Densities: S10 = the frozen m_p1.P1Fusion (simulator 10 % sample), D100 = p1_full.P1FusionFull (all valid pixels).
State sets: valsel (MP3D VALSEL, 90 states) and dev3 (Macarons++ Dev-3 states of E1.2, strata B0 / B2, 275 states) --
exactly the sets, regions, noise keys and replay rules of e2_fixed.py (imported, unchanged).

Per episode ONE replay with BOTH maps (S10 and D100 of the same service records, like the closed-loop P1FusionB2 twin):
indoor = n_win = 5 region rebuilt at r = t - t % 10 from records 0..r + increments; outdoor = one scene grid. At every
refresh r: per (checkpoint, density) h = encode(coarse of that density's map); ONE noise eps (same key for all 12 configs).
Per state:
  inference  for x in (D100, S10): geometry of map x; per checkpoint score_state -> K = 4 decoded future depths per
             candidate (kept) and DistP1 with map x's own band_dil
  scoring    the other density's band_dil applied to the SAME decoded depths (scoring.new_set_from_depth, unchanged):
             12 score lists per state = 3 ckpt x 2 inference x 2 scoring
  G1 (GT read only after all scores): per config, rays of every candidate grouped by the ray class in BOTH maps
             (common UNKNOWN / only-S10 UNKNOWN / only-D100 UNKNOWN / own UNKNOWN / all / common-UNKNOWN truly new):
             hit / no-hit confusion vs t*, depth error (u = 1 m indoor, 1 runtime unit outdoor), accurate first hit
             (|err| <= trunc + 1 %), K = 4 spread; the e2_fixed pred_groups for the historical 4 configs
  masks      per config and scoring density: decoded hits in domain / excluded by band_dil (= not novel) / rewarded,
             split by pixel truth (GT point farther than eps from every past 10 % sample = truly new); the same for
             the GT depth itself
Checks (STOP on failure): the four historical configs equal the e2_fixed rows bitwise (B0 = B2|S10|S10,
B2 = B2|D100|S10, S10 = E2S|S10|S10, D100 = E2D|D100|S10), B2|D100|D100 equals the E1.2 own-D100 rows (dev3);
the D100 band_dil recomputed from the map's input volume (the closed-loop scoring-map path) equals the caster's own.
Output work_dirs/exp_density_ab/fixed/<set>/<stratum>/<tag>.jsonl
Usage: python da_fixed.py --set valsel|dev3 --stratum B0|B2 --part i/n [--tags a,b]
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
T3S = os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1")
sys.path.insert(0, os.path.join(T3S, "e2_pilot", "scripts"))
import e2_fixed as EF  # noqa: E402  (imports e2_paths, e12_fixed as EF.X)

EP, X = EF.EP, EF.X
OUT = os.path.join(REPO, "work_dirs", "exp_density_ab", "fixed")
CKPTS = ("B2", "E2S", "E2D")
DENS = (100, 10)
HIST = {"B0": ("B2", 10, 10), "B2": ("B2", 100, 10), "S10": ("E2S", 10, 10), "D100": ("E2D", 100, 10)}
SETS = ("all", "unk_own", "unk_common", "unk_only10", "unk_only100", "unk_union", "new_common")


def cfg(m, xi, xs):
    return f"{m}|{'D100' if xi == 100 else 'S10'}|{'D100' if xs == 100 else 'S10'}"


def distp1_bd(views, hits, geo, band_dil, k_M, z0):
    """X.distp1_k with an explicit band_dil (same dims / domain: they depend on the region only)."""
    SCO, torch = X.SCO, X.torch
    out = []
    for v, hh in zip(views, hits):
        hh = hh.to(v["o"].device)
        out.append([float(SCO.new_set_from_depth(hh[k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain, band_dil, z0)[1].sum())
                    for k in range(hh.shape[0])])
    return out


def flags(geo, band_dil, v, t):
    """(hit in domain, excluded = in band_dil) per ray for depths t [R] (numpy bool)."""
    torch, MC = X.torch, X.MC
    hit = torch.isfinite(t)
    p = v["o"][None] + v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / MC.VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    hit = hit & geo.domain[i, j, k]
    return hit.cpu().numpy(), (hit & band_dil[i, j, k]).cpu().numpy()


def conf(zk, ok, zg, sel, u, tol):
    """Sums over K x rays in sel: confusion vs GT hit, |err| on both-hit, accurate first hits."""
    np = X.np
    ph = np.isfinite(zk)
    s = sel[None]
    both = ph & ok[None] & s
    e = np.abs(np.where(both, zk - np.where(ok, zg, 0)[None], 0.0))
    return {"n": int(s.sum() * zk.shape[0]), "gt": int((ok & sel).sum() * zk.shape[0]), "ph": int((ph & s).sum()),
            "tp": int(both.sum()), "fp": int((ph & ~ok[None] & s).sum()), "fn": int((~ph & ok[None] & s).sum()),
            "tn": int((~ph & ~ok[None] & s).sum()), "sae": float(e[both].sum() / u), "acc": int((both & (e <= tol[None])).sum())}


def addd(a, b):
    for k, v in b.items():
        a[k] = a.get(k, 0) + v


def run_episode(sset, stratum, tag, sts, cores, dev, ref2, ref12):
    np, torch, MC, PC, PR, RY, D = X.np, X.torch, X.MC, X.PC, X.PR, X.RY, X.D
    rp = EF.Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    u = 7.0 if sset == "valsel" else 1.0
    dname = "indoor" if sset == "valsel" else "outdoor"
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, recs, dname
    rows = {s["sid"]: {"sid": s["sid"], "scene": s["scene"], "start": s["start"], "t": s["t"], "tag": tag, "stratum": stratum, "set": sset,
                       "phase": X.phase_of(dname, s["t"]), "scores": {}, "g1": {}, "pred_e2": {}, "masks": {}, "ray_counts": {}, "timing": {}}
            for s in sts.values()}
    torch.cuda.reset_peak_memory_stats()
    tm = {"encode_s": [], "state_s": []}

    def refresh(region, fus, r):
        h = {}
        for x in DENS:
            geo = PC.Geometry.from_fusion(*region, fus[x], dev)
            c = geo.coarse()
            t0 = X.sync()
            for m in CKPTS:
                h[(m, x)] = cores[m].pred.encode(c)
            tm["encode_s"].append(X.sync() - t0)
            del geo, c
        eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
        return h, eps

    def new_twin(region):
        return {x: D.fusion_class(x)(*region, device=dev) for x in DENS}

    if sset == "valsel":
        blocks = sorted({t - t % 10 for t in sts})
        for r in blocks:
            region = rp.region_of(recs[r])
            fus = new_twin(region)
            for q in range(r + 1):
                for x in DENS:
                    fus[x].integrate_record(recs[q])
            h, eps = refresh(region, fus, r)
            last = max(t for t in sts if t - t % 10 == r)
            for t in range(r, last + 1):
                if t > r:
                    for x in DENS:
                        fus[x].integrate_record(recs[t])
                if t in sts:
                    state(sts[t], rows, recs[t], region, fus, h, eps, cores, dev, rp, E, u, tm)
            del fus
            torch.cuda.empty_cache()
    else:
        region = rp.frame
        fus = new_twin(region)
        blocks = {t - t % 10 for t in sts}
        h = eps = None
        for t in range(T + 1):
            for x in DENS:
                fus[x].integrate_record(recs[t])
            if t in blocks:
                h, eps = refresh(region, fus, t)
            if t in sts:
                state(sts[t], rows, recs[t], region, fus, h, eps, cores, dev, rp, E, u, tm)
        del fus
    torch.cuda.empty_cache()
    out = [rows[s["sid"]] for s in sorted(sts.values(), key=lambda s: s["t"])]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for r in out:
        r["episode_peak_gb"] = peak
        r["episode_encode_s_mean"] = float(np.mean(tm["encode_s"])) if tm["encode_s"] else None
        ok = {}
        if r["sid"] in ref2:
            h2 = ref2[r["sid"]]
            for hk, (m, xi, xs) in HIST.items():
                ok[f"{hk}_scores_eq_e2"] = r["scores"][cfg(m, xi, xs)] == h2["scores"][hk]
                ok[f"{hk}_pred_eq_e2"] = r["pred_e2"][f"{m}|{xi}"] == h2["pred"][hk]
        if r["sid"] in ref12:
            ok["B2_D100_D100_eq_e12_own100"] = r["scores"][cfg("B2", 100, 100)] == ref12[r["sid"]]
        r["repro"] = ok
        bad = [k for k, v in ok.items() if not v and "_scores_" in k or (k.startswith("B2_D100") and not v)]
        if bad or not r["band_dil_recompute_equal"]:
            raise SystemExit(f"STOP: reproduction failed {r['sid']} {bad} band_dil_eq={r['band_dil_recompute_equal']}")
    return out


def state(st, rows, rec, region, fus, h, eps, cores, dev, rp, E, u, tm):
    np, torch, MC, PC, G, F = X.np, X.torch, X.MC, X.PC, X.G, X.F
    sid, row = st["sid"], rows[st["sid"]]
    t_state = X.sync()
    hits, cls, bd, views0, geo10 = {}, {}, {}, None, None
    for x in DENS:
        geo = PC.Geometry.from_fusion(*region, fus[x], dev)
        k_M = float(geo.fr.k_M)
        for m in CKPTS:
            out, tmm, views = PC.score_state(cores[m], geo, h[(m, x)], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
            hits[(m, x)] = [hh["t"].detach().cpu() for hh in out["hits"]]
            row["scores"][cfg(m, x, x)] = out["distp1_k"]
            row["timing"][f"{m}|{x}"] = tmm
        cls[x] = [v["pr"]["cls"].cpu().numpy() for v in views]
        if x == 100:
            bd[100] = geo.caster.band_dil.clone()
            v_in = fus[100].input_volume().to(geo.caster.dev).to(geo.caster.dt)       # closed-loop scoring-map path
            ob = (v_in.abs() < MC.TRUNC * 0.999) & geo.caster.domain
            row["band_dil_recompute_equal"] = bool(torch.equal(F.max_pool3d(ob[None, None].half(), 3, 1, 1)[0, 0] > 0, bd[100]))
            del v_in, ob, geo, views
            torch.cuda.empty_cache()
        else:
            bd[10], geo10, views0 = geo.caster.band_dil, geo, views
    geo = geo10
    k_M = float(geo.fr.k_M)
    for m in CKPTS:
        row["scores"][cfg(m, 100, 10)] = X.distp1_k(views0, hits[(m, 100)], geo, k_M, rp.z0)
        row["scores"][cfg(m, 10, 100)] = distp1_bd(views0, hits[(m, 10)], geo, bd[100], k_M, rp.z0)
    row["state_score_s"] = X.sync() - t_state
    # ---------------------------------------------------------------- evaluation side (GT) from here on
    if rp.sset == "valsel":
        root, dname = os.path.join(EP.W, "valsel"), "indoor"
    else:
        root, dname = (X.ORIG_W if row["stratum"] == "B0" else X.P.ROOT_B2), "outdoor"
    orc = np.load(os.path.join(root, "states", dname, sid, "oracle.npz"))
    if dname == "indoor":
        assert abs(float(orc["k_M"]) - k_M) < 1e-9 * k_M
    truth, eps_rt = E.truth_of(st, orc, None, k_M)
    trunc_rt = MC.TRUNC * k_M
    row["oracle_gain_pp"] = (np.asarray(orc["gain"], float) * 100.0).tolist()
    for j in st["cand"]:
        new, ok, zg = truth[j]
        v = views0[j]
        dz = v["dz"].double().cpu().numpy()
        tstar = orc["tstar"][j].astype(np.float64)
        u10, u100 = cls[10][j] == G.CLS_UNKNOWN, cls[100][j] == G.CLS_UNKNOWN
        sel = {"all": np.ones_like(ok), "unk_common": u10 & u100, "unk_only10": u10 & ~u100, "unk_only100": u100 & ~u10,
               "unk_union": u10 | u100, "new_common": u10 & u100 & ok & new}
        tol = trunc_rt + 0.01 * np.where(ok, zg, 0.0)
        rc = row["ray_counts"]
        addd(rc, {"rays": int(len(ok)), "gt_hit": int(ok.sum()), "unk10": int(u10.sum()), "unk100": int(u100.sum()),
                  "unk_common": int(sel["unk_common"].sum()), "unk_only10": int(sel["unk_only10"].sum()),
                  "unk_only100": int(sel["unk_only100"].sum()), "unk_common_gt": int((sel["unk_common"] & ok).sum()),
                  "unk_only10_gt": int((sel["unk_only10"] & ok).sum()), "truth_new": int((ok & new).sum()),
                  "unk10_truth_new": int((u10 & ok & new).sum()), "unk100_truth_new": int((u100 & ok & new).sum())})
        # GT depth itself under both band_dil (scoring-side truth)
        tg = torch.as_tensor(np.where(ok, tstar, np.inf), dtype=v["o"].dtype, device=v["o"].device)
        for xs in DENS:
            hit, exd = flags(geo, bd[xs], v, tg)
            addd(row["masks"].setdefault(f"GT|{xs}", {}), {"hit": int(hit.sum()), "excl": int(exd.sum()),
                                                           "excl_truth_new": int((exd & new & ok).sum()), "reward_truth_old": int((hit & ~exd & ~new & ok).sum()),
                                                           "reward_truth_new": int((hit & ~exd & new & ok).sum()), "truth_new_hit": int((hit & new & ok).sum())})
        for m in CKPTS:
            for xi in DENS:
                tk = hits[(m, xi)][j]
                zk = tk.double().numpy() * k_M * dz[None]
                g = row["g1"].setdefault(f"{m}|{xi}", {})
                for nm in SETS:
                    s = (u10 if xi == 10 else u100) if nm == "unk_own" else sel[nm]
                    addd(g.setdefault(nm, {}), conf(zk, ok, zg, s, u, tol))
                pg = EF.pred_groups(tk, tstar, dz, k_M, u, ok, new, u10 if xi == 10 else u100)
                rho = pg.pop("rho", None)
                pe = row["pred_e2"].setdefault(f"{m}|{xi}", {})
                addd(pe, pg)
                pe.setdefault("_rho", []).append(rho)
                for xs in DENS:
                    mk = row["masks"].setdefault(cfg(m, xi, xs), {})
                    for k in range(tk.shape[0]):
                        hit, exd = flags(geo, bd[xs], v, tk[k].to(v["o"].device))
                        acc = ok & np.isfinite(zk[k]) & (np.abs(zk[k] - np.where(ok, zg, 0)) <= tol)
                        addd(mk, {"hit": int(hit.sum()), "excl": int(exd.sum()), "reward": int((hit & ~exd).sum()),
                                  "excl_truth_new": int((exd & new & ok).sum()), "reward_truth_new": int((hit & ~exd & new & ok).sum()),
                                  "reward_truth_old": int((hit & ~exd & ~new & ok).sum()), "reward_no_gt": int((hit & ~exd & ~ok).sum()),
                                  "reward_accurate": int((hit & ~exd & acc).sum()), "excl_accurate_new": int((exd & acc & new).sum())})
    for key, pe in row["pred_e2"].items():
        rv = [r for r in pe.pop("_rho") if r is not None]
        pe["rho_mean"] = float(np.mean(rv)) if rv else None
    row["eps_rt"] = eps_rt
    del geo, views0, bd, hits
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--tags", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(EP.EXP5, "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    c = X.S2.load_cores(dev)
    cores = {"B2": c["V2M"], "E2S": X.PC.Core(EF.ckpt("s10", 5000), dev), "E2D": X.PC.Core(EF.ckpt("d100", 5000), dev)}
    ref2, ref12 = {}, {}
    for f in glob.glob(os.path.join(EP.W, "fixed", f"{a.set}_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref2[r["sid"]] = {"scores": r["scores"], "pred": r["pred"]}
    if a.set == "dev3":
        for f in glob.glob(os.path.join(X.P.E12, "fixed", a.stratum, "outdoor", "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                ref12[r["sid"]] = r["d"]["100"]["own_k"]
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(a.out, a.set, a.stratum)
    os.makedirs(od, exist_ok=True)
    eps_ = EF.states(a.set, a.stratum)
    if a.tags:
        eps_ = [e for e in eps_ if e[0] in a.tags.split(",")]
    for tag, sts in eps_[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = run_episode(a.set, a.stratum, tag, sts, cores, dev, ref2, ref12)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        nrep = sum(len(r["repro"]) for r in rows)
        print(a.set, a.stratum, tag, len(rows), "states %.0fs" % (time.time() - tb), "repro checks", nrep, "all ok",
              all(all(r["repro"].values()) for r in rows), "peak %.1f GB" % rows[0]["episode_peak_gb"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
