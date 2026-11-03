"""EXP Temporal Refresh -- fixed-state G1 / G2 of the encoder refresh period R in {10, 5, 2, 1} (seen2scene env, 1 GPU per
process; REFRESH_PROTOCOL.md §3).

Checkpoints: B2 (V2-Mix EMA@17.5k) and E2D (E2-D100 EMA@5k); historical P1 protocol b2 = D100 inference, S10 DistP1
scoring. State sets, regions, noise keys and replay rules = e2_fixed.py (imported unchanged): MP3D VALSEL (90 states) and
Macarons++ Dev-3 (275). Both P1 maps (D100, S10) of the same records are kept (twin). Per refresh block r10 = t - t % 10:
indoor region / P1 rebuilt from records 0..r10 exactly as the planner, ONE noise (R10 key of r10, shared by all R). For a
state t and period R the encoder input is the D100 coarse grid of the map at tau_R = R * floor(t / R) (records 0..tau_R in
the r10 region; tau_10 = r10 = historical, tau_1 = t = fresh). All periods score the SAME current geometry (map at t), same
candidates, same noise; only h differs. Identical tau_R share one encoding / one prediction.
Per state, checkpoint and R: DistP1 with the S10 band_dil (protocol) and with the D100 band_dil (free, diagnostic);
G1 on the D100-UNKNOWN rays of every candidate (+ truly new subset, all rays), e2 pred_groups; encoder times.
Checks (STOP): R10 scores and pred_groups == historical e2_fixed rows (B2 = 'B2', E2D = 'D100').
Output work_dirs/exp_temporal_refresh/fixed/<set>/<stratum>/<tag>.jsonl
Usage: python tr_fixed.py --set valsel|dev3 --stratum B0|B2 --part i/n
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
sys.path.insert(0, os.path.join(REPO, "experiments", "exp_density_ab", "scripts"))
import da_fixed as DA  # noqa: E402  (imports e2_fixed as DA.EF and e12_fixed as DA.X)

EF, EP, X = DA.EF, DA.EP, DA.X
OUT = os.path.join(REPO, "work_dirs", "exp_temporal_refresh", "fixed")
CKPTS = ("B2", "E2D")
HISTKEY = {"B2": "B2", "E2D": "D100"}
PERIODS = (10, 5, 2, 1)


def tau(t, R):
    return R * (t // R)


def run_episode(sset, stratum, tag, sts, cores, dev, ref2):
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
                       "phase": X.phase_of(dname, s["t"]), "stale10": s["t"] % 10, "tau": {str(R): tau(s["t"], R) for R in PERIODS},
                       "scores": {}, "scores_d100": {}, "g1": {}, "pred_e2": {}, "timing": {}} for s in sts.values()}
    torch.cuda.reset_peak_memory_stats()
    enc_t = []
    need = {}
    for t in sts:
        for R in PERIODS:
            need.setdefault(t - t % 10, set()).add(tau(t, R))

    def encode_now(region, fus, q, H):
        geo = PC.Geometry.from_fusion(*region, fus[100], dev)
        c = geo.coarse()
        for m in CKPTS:
            X.sync()
            t0 = time.time()
            H[(m, q)] = cores[m].pred.encode(c)
            X.sync()
            enc_t.append(time.time() - t0)
        del geo, c

    def new_twin(region):
        return {x: D.fusion_class(x)(*region, device=dev) for x in (100, 10)}

    def block(region, fus, r, t_from, t_to):
        H = {}
        eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
        for q in range(t_from, t_to + 1):
            if q > t_from or t_from > r:
                for x in (100, 10):
                    fus[x].integrate_record(recs[q])
            if q in need.get(r, ()):
                encode_now(region, fus, q, H)
            if q in sts:
                state(sts[q], rows, recs[q], region, fus, H, eps, cores, dev, rp, E, u)
        return H

    if sset == "valsel":
        for r in sorted(need):
            region = rp.region_of(recs[r])
            fus = new_twin(region)
            for qq in range(r + 1):
                for x in (100, 10):
                    fus[x].integrate_record(recs[qq])
            last = max(t for t in sts if t - t % 10 == r)
            block(region, fus, r, r, last)
            del fus
            torch.cuda.empty_cache()
    else:
        region = rp.frame
        fus = new_twin(region)
        q0 = 0
        for r in sorted(need):
            for qq in range(q0, r + 1):
                for x in (100, 10):
                    fus[x].integrate_record(recs[qq])
            last = max(t for t in sts if t - t % 10 == r)
            block(region, fus, r, r, last)
            q0 = last + 1
        del fus
    torch.cuda.empty_cache()
    out = [rows[s["sid"]] for s in sorted(sts.values(), key=lambda s: s["t"])]
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for r in out:
        r["episode_peak_gb"] = peak
        r["encode_s_mean"] = float(np.mean(enc_t)) if enc_t else None
        ok = {}
        if r["sid"] in ref2:
            for m in CKPTS:
                ok[f"{m}_R10_scores_eq_e2"] = r["scores"][f"{m}|10"] == ref2[r["sid"]]["scores"][HISTKEY[m]]
                ok[f"{m}_R10_pred_eq_e2"] = r["pred_e2"][f"{m}|10"] == ref2[r["sid"]]["pred"][HISTKEY[m]]
        r["repro"] = ok
        bad = [k for k, v in ok.items() if not v and "_scores_" in k]
        if bad:
            raise SystemExit(f"STOP: R10 reproduction failed {r['sid']} {bad}")
    return out


def state(st, rows, rec, region, fus, H, eps, cores, dev, rp, E, u):
    np, torch, MC, PC, G = X.np, X.torch, X.MC, X.PC, X.G
    sid, row = st["sid"], rows[st["sid"]]
    t = st["t"]
    t0 = X.sync()
    geo = PC.Geometry.from_fusion(*region, fus[100], dev)
    k_M = float(geo.fr.k_M)
    hits, views100 = {}, None
    for m in CKPTS:
        done = {}
        for R in PERIODS:
            q = tau(t, R)
            if q in done:
                hits[(m, R)] = hits[(m, done[q])]
                row["scores_d100"][f"{m}|{R}"] = row["scores_d100"][f"{m}|{done[q]}"]
                continue
            ts = X.sync()
            out, tmm, views = PC.score_state(cores[m], geo, H[(m, q)], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
            row["timing"][f"{m}|{R}"] = dict(tmm, total_s=X.sync() - ts)
            hits[(m, R)] = [hh["t"].detach().cpu() for hh in out["hits"]]
            row["scores_d100"][f"{m}|{R}"] = out["distp1_k"]
            done[q] = R
            views100 = views
    cls100 = [v["pr"]["cls"].cpu().numpy() for v in views100]
    del geo, views100
    torch.cuda.empty_cache()
    geo10 = PC.Geometry.from_fusion(*region, fus[10], dev)
    views0 = [geo10.view(R_, T_) for R_, T_ in zip(rec["legal_R"], rec["legal_T"])]
    for m in CKPTS:
        for R in PERIODS:
            row["scores"][f"{m}|{R}"] = X.distp1_k(views0, hits[(m, R)], geo10, k_M, rp.z0)
    row["state_s"] = X.sync() - t0
    # ---------------------------------------------------------------- evaluation side (GT) from here on
    if rp.sset == "valsel":
        root, dname = os.path.join(EP.W, "valsel"), "indoor"
    else:
        root, dname = (X.ORIG_W if row["stratum"] == "B0" else X.P.ROOT_B2), "outdoor"
    orc = np.load(os.path.join(root, "states", dname, sid, "oracle.npz"))
    truth, eps_rt = E.truth_of(st, orc, None, k_M)
    trunc_rt = MC.TRUNC * k_M
    row["oracle_gain_pp"] = (np.asarray(orc["gain"], float) * 100.0).tolist()
    for j in st["cand"]:
        new, ok, zg = truth[j]
        dz = views0[j]["dz"].double().cpu().numpy()
        tstar = orc["tstar"][j].astype(np.float64)
        u100 = cls100[j] == G.CLS_UNKNOWN
        sel = {"unk": u100, "new": u100 & ok & new, "all": np.ones_like(ok)}
        tol = trunc_rt + 0.01 * np.where(ok, zg, 0.0)
        for m in CKPTS:
            for R in PERIODS:
                tk = hits[(m, R)][j]
                zk = tk.double().numpy() * k_M * dz[None]
                g = row["g1"].setdefault(f"{m}|{R}", {})
                for nm, s in sel.items():
                    DA.addd(g.setdefault(nm, {}), DA.conf(zk, ok, zg, s, u, tol))
                pg = EF.pred_groups(tk, tstar, dz, k_M, u, ok, new, u100)
                rho = pg.pop("rho", None)
                pe = row["pred_e2"].setdefault(f"{m}|{R}", {})
                DA.addd(pe, pg)
                pe.setdefault("_rho", []).append(rho)
    for key, pe in row["pred_e2"].items():
        rv = [r for r in pe.pop("_rho") if r is not None]
        pe["rho_mean"] = float(np.mean(rv)) if rv else None
    del geo10, views0, hits
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
    cores = {"B2": c["V2M"], "E2D": X.PC.Core(EF.ckpt("d100", 5000), dev)}
    ref2 = {}
    for f in glob.glob(os.path.join(EP.W, "fixed", f"{a.set}_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref2[r["sid"]] = {"scores": r["scores"], "pred": r["pred"]}
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
        rows = run_episode(a.set, a.stratum, tag, sts, cores, dev, ref2)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "states %.0fs" % (time.time() - tb), "repro", sum(len(r["repro"]) for r in rows),
              "all ok", all(all(r["repro"].values()) for r in rows), "peak %.1f GB" % rows[0]["episode_peak_gb"], flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
