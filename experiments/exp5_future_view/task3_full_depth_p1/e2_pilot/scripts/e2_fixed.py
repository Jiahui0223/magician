"""EXP5 Task 3 / E2 pilot -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process; protocol §4-§5).

Models (same noise, same candidates, DistP1 always with the SPARSE 10 % band_dil):
  B0 = frozen V2-Mix on D10 input   B2 = frozen V2-Mix on D100 input
  S10 = E2-S10 (step N) on D10      D100 = E2-D100 (step N) on D100
Sets: valsel (MP3D VALSEL trajectories, e2_valsel_oracle manifest; planning region n_win = 5 of the Task 2 cards, as the
training data) and dev3 (Macarons++ Dev-3 states of E1.2, strata B0 / B2). Per episode two replays with the planner
rules (indoor: region rebuilt at r = t - t % 10 from records 0..r + increments; outdoor: one grid), D100 then D10, one
map at a time; encoder input of every model at r. Per state: the four score lists; predictions vs t* (evaluation only)
grouped by ray: all / unknown (cls UNKNOWN in the model's own input map) / truly new / known (GT point within the
coverage eps of a past 10 % sample); K = 4 relative spread, hit disagreement, spread-error Spearman.
Check (dev3, stratum B0): B0 and B2 scores == the E1.2 rows (I10 / I100) bitwise, else STOP.
Output e2/fixed/<set>_s<step>/<stratum>/<tag>.jsonl. Usage: python e2_fixed.py --set valsel|dev3 --stratum B0|B2 --step 5000 --part i/n
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_paths as EP  # noqa: E402

sys.path.insert(0, os.path.join(EP.EXP5, "task3_full_depth_p1", "density_ablation", "scripts"))
import e12_fixed as X  # noqa: E402

DEV3 = ("barts", "pantheon", "colosseum")
MODELS = {"B0": ("V2M", 10), "B2": ("V2M", 100), "S10": ("E2S", 10), "D100": ("E2D", 100)}


def ckpt(arm, step):
    return os.path.join(EP.W, "training", arm, f"ema_{step:06d}.pt")


def states(sset, stratum):
    if sset == "valsel":
        sts = json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]
        out = {}
        for s in sts:
            out.setdefault(s["tag"], []).append(s)
        return sorted(out.items())
    return [(tag, s) for tag, s in X.states_of("outdoor", stratum) if s[0]["scene"] in DEV3]


class Rep:
    """Replay adapter: region of a record, z0, noise key, record path."""

    def __init__(self, sset, st, dev):
        if sset == "valsel":
            import t2_common as C
            import t21_build as B21
            card = C.load_card(st["scene"])
            self.region_of = lambda rec: B21.region(card, rec["cam_center"])[:3]
            self.z0 = float(card["z0_rt"])
            self.key = lambda r, k: f"exp5-T3E2|valsel|{st['tag']}|t{r}|k{k}"
            td = os.path.dirname(os.path.join(EP.REPO, st["sources"][0]["svc"]))
            self.svc = lambda t: os.path.join(td, f"svc_t{t}.npz")
            self.frame = None
        else:
            ep = X.S2.episode_of("outdoor", st)
            rp = X.S2.StateReplay("outdoor", ep, "V2M", {}, dev)
            self.frame, self.z0, self.key = rp.frame[:3], rp.z0, rp.seed_key
            self.svc = lambda t: X.A.svc_path("outdoor", "V2M", ep, t)


def pred_groups(t_k, tgt, dz, k_M, u, ok, new, unk):
    np = X.np
    zk = t_k.double().cpu().numpy() * k_M * dz[None]
    zg = tgt * k_M * dz
    ph = np.isfinite(zk)
    both = ph & ok[None]
    e = np.abs(np.where(both, zk - np.where(ok, zg, 0)[None], 0.0)) / u
    out = {"n_rays": int(zk.size), "hit_agree": int((ph == ok[None]).sum())}
    for nm, m in (("all", np.ones_like(ok)), ("unknown", unk), ("new", new & ok), ("known", ~new & ok), ("unknown_new", unk & new & ok)):
        mm = both & m[None]
        out[f"n_{nm}"], out[f"sae_{nm}"] = int(mm.sum()), float(e[mm].sum())
    allh = ph.all(0)
    out["n_allhit"], out["hit_dis"] = int(allh.sum()), int((ph.any(0) & ~allh).sum())
    out["n_unk_rays"], out["hit_dis_unk"] = int(unk.sum()), int((ph.any(0) & ~allh & unk).sum())
    if allh.any():
        zz = zk[:, allh]
        out["rel_std_sum"] = float((zz.std(0) / np.maximum(zz.mean(0), 1e-9)).sum())
        m2 = allh & ok
        if m2.sum() >= 10:
            from scipy.stats import spearmanr
            r = spearmanr(zk[:, m2].std(0), np.abs(zk[:, m2].mean(0) - zg[m2])).correlation
            out["rho"] = float(r) if np.isfinite(r) else None
    return out


def run_episode(sset, stratum, tag, sts, cores, dev, ref):
    np, torch, MC, PC, PR, RY = X.np, X.torch, X.MC, X.PC, X.PR, X.RY
    rp = Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    u = 7.0
    if sset == "dev3":
        u = 1.0
    rows = {s["sid"]: {"sid": s["sid"], "scene": s["scene"], "start": s["start"], "t": s["t"], "tag": tag, "stratum": stratum,
                       "phase": X.phase_of("indoor" if sset == "valsel" else "outdoor", s["t"]), "scores": {}, "pred": {}} for s in sts.values()}
    hits, truth = {}, {}
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, recs, "indoor" if sset == "valsel" else "outdoor"
    for x in (100, 10):
        cls = X.D.fusion_class(x)
        mods = [m for m, (c, d) in MODELS.items() if d == x]
        torch.cuda.reset_peak_memory_stats()
        blocks = sorted({t - t % 10 for t in sts})
        if sset == "valsel":
            for r in blocks:
                region = rp.region_of(recs[r])
                fus = cls(*region, device=dev)
                for q in range(r + 1):
                    fus.integrate_record(recs[q])
                h, eps = refresh(rp, region, fus, r, mods, cores, dev)
                last = max(t for t in sts if t - t % 10 == r)
                for t in range(r, last + 1):
                    if t > r:
                        fus.integrate_record(recs[t])
                    if t in sts:
                        state(x, sts[t], rows, recs[t], region, fus, h, eps, mods, cores, dev, rp, hits, E, u)
                del fus
        else:
            region = rp.frame
            fus = cls(*region, device=dev)
            h = eps = None
            for t in range(T + 1):
                fus.integrate_record(recs[t])
                if t in blocks:
                    h, eps = refresh(rp, region, fus, t, mods, cores, dev)
                if t in sts:
                    state(x, sts[t], rows, recs[t], region, fus, h, eps, mods, cores, dev, rp, hits, E, u)
            del fus
        torch.cuda.empty_cache()
    out = [rows[s["sid"]] for s in sorted(sts.values(), key=lambda s: s["t"])]
    for r in out:
        if ref is not None and r["sid"] in ref:
            ok = {"B0_eq_E12_I10": r["scores"]["B0"] == ref[r["sid"]]["I10"], "B2_eq_E12_I100": r["scores"]["B2"] == ref[r["sid"]]["I100"]}
            r["repro"] = ok
            if not all(ok.values()):
                raise SystemExit(f"STOP: frozen-model scores differ from E1.2 ({r['sid']} {ok})")
    return out


def refresh(rp, region, fus, r, mods, cores, dev):
    PC, PR, RY = X.PC, X.PR, X.RY
    geo = PC.Geometry.from_fusion(*region, fus, dev)
    h = {m: cores[MODELS[m][0]].pred.encode(geo.coarse()) for m in mods}
    eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
    del geo
    return h, eps


def state(x, st, rows, rec, region, fus, h, eps, mods, cores, dev, rp, hits, E, u):
    np, torch, MC, PC = X.np, X.torch, X.MC, X.PC
    sid, row = st["sid"], rows[st["sid"]]
    geo = PC.Geometry.from_fusion(*region, fus, dev)
    k_M = float(geo.fr.k_M)
    views = None
    for m in mods:
        out, tm, views = PC.score_state(cores[MODELS[m][0]], geo, h[m], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
        hits[(sid, m)] = [hh["t"].detach().cpu() for hh in out["hits"]]
        row.setdefault("own_k", {})[m] = out["distp1_k"]
        row.setdefault("timing", {})[m] = tm
        row.setdefault("cls_unknown", {})[m] = [(v["pr"]["cls"] == X.G.CLS_UNKNOWN).cpu().numpy() for v in views]
    if x == 10:
        for m in MODELS:
            row["scores"][m] = row["own_k"][m] if MODELS[m][1] == 10 else X.distp1_k(views, hits[(sid, m)], geo, k_M, rp.z0)
    # ---- evaluation side
    if rp.sset == "valsel":
        root, dname = os.path.join(EP.W, "valsel"), "indoor"
    else:
        root, dname = (X.ORIG_W if row["stratum"] == "B0" else X.P.ROOT_B2), "outdoor"
    orc = np.load(os.path.join(root, "states", dname, sid, "oracle.npz"))
    assert abs(float(orc["k_M"]) - k_M) < 1e-9 * k_M if dname == "indoor" else True
    truth, _ = E.truth_of(st, orc, None, k_M)
    for m in mods:
        acc = {}
        rl = []
        for j in st["cand"]:
            new, ok, zg = truth[j]
            dz = views[j]["dz"].double().cpu().numpy()
            pg = pred_groups(hits[(sid, m)][j], orc["tstar"][j].astype(np.float64), dz, k_M, u, ok, new, row["cls_unknown"][m][j])
            rl.append(pg.pop("rho", None))
            for k, v in pg.items():
                acc[k] = acc.get(k, 0) + v
        rv = [r for r in rl if r is not None]
        acc["rho_mean"] = float(np.mean(rv)) if rv else None
        row["pred"][m] = acc
    for m in mods:
        row["cls_unknown"].pop(m)
    if x == 10:
        row.pop("cls_unknown")
    del geo, views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--step", type=int, default=5000)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(EP.EXP5, "task2", "scripts"))
    np, torch = X.np, X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    cores = X.S2.load_cores(dev)
    cores = {"V2M": cores["V2M"], "E2S": X.PC.Core(ckpt("s10", a.step), dev), "E2D": X.PC.Core(ckpt("d100", a.step), dev)}
    ref = None
    if a.set == "dev3" and a.stratum == "B0":
        ref = {}
        for f in glob.glob(os.path.join(X.P.E12, "fixed", "B0", "outdoor", "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                ref[r["sid"]] = {"I10": r["I_k"]["10"], "I100": r["I_k"]["100"]}
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(EP.W, "fixed", f"{a.set}_s{a.step}", a.stratum)
    os.makedirs(od, exist_ok=True)
    for tag, sts in states(a.set, a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = run_episode(a.set, a.stratum, tag, sts, cores, dev, ref)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
