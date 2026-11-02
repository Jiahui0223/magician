"""EXP6 V5.2 -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process).

Same states, candidates, regions and refresh rules as the V5 / V4 / E2 fixed-state evaluations (exp6_v3/shared/
eval_fixed.py -> e2_fixed / e12_fixed helpers): MP3D VALSEL (90 states), Macarons++ Dev-3 B0 (150) and B2 (125). One
replay per episode with BOTH maps of the closed-loop B2 path: the D100 P1 (model input, structure, FREE / UNKNOWN) and
the S10 P1 of the same records (band_dil novelty). Encoder input at the refresh step r (indoor: region rebuilt at
r = t - t % 10; outdoor: one grid), as in the planners.
Per state, for every legal candidate: V5.2 raw terms S_vis / P_surf / P_space (+ diagnostics) and the scores for
lambda = 0 and lambda = 0.1 (score_v52.json); optionally the V5 DistP1 scores recomputed on the same geometry
(--v5-replay, platform check against the exp6_v5 rows). Evaluation side (after all scores of the state): first-hit
metrics vs the oracle t* of the candidate views; belief dump (256 rays per candidate view, UNKNOWN first, z fp16) for
the multi-surface GT (eval_belief_gt.py). Output work_dirs/exp6_v5_2/eval_fixed/<set>_<tag>/<stratum>/<ep>.jsonl (+ belief_<ep>.npz).
Usage: python eval_g1_g2.py --set valsel|dev3 --stratum B0|B2 --ckpt V52_CKPT [--v5-replay V5_CKPT] [--tag s_final] [--part i/n] [--max-eps N]
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v52_common as VC  # noqa: E402

REPO = VC.REPO
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
SC52 = VC.SC52
N_UNK, N_ALL = 128, 128


def belief_rows(model, h, view, key, j):
    """256 distinct rays of one candidate view (UNKNOWN first) -> z and range for the eval-side multi-surface GT."""
    np, torch, G = X.np, X.torch, X.G
    o, d, tc = view["o"].float(), view["d"].float(), view["tcap"].float()
    cls = view["pr"]["cls"]
    g = torch.Generator(device="cpu"); g.manual_seed(int(hashlib.sha256(f"exp6-V5.2|belief|{key}|{j}".encode()).hexdigest()[:15], 16))
    unk = torch.nonzero(cls.cpu() == G.CLS_UNKNOWN)[:, 0]
    pu = unk[torch.randperm(len(unk), generator=g)[:N_UNK]]
    free = torch.ones(d.shape[0], dtype=torch.bool); free[pu] = False
    others = torch.nonzero(free)[:, 0]
    idx = torch.cat([pu, others[torch.randperm(len(others), generator=g)[:N_UNK + N_ALL - len(pu)]]]).to(d.device)
    ch = G.channels(view["pr"], view["tcap"], view["d"], view["o"]).float().t()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.belief(model.memory(h), o[None].expand(len(idx), 3), d[idx], tc[idx], ch[idx])
    return idx, cls[idx], out["z"].float(), tc[idx]


def run_episode(sset, stratum, tag, sts, model, v5core, dev, cfg, od):
    np, torch, MC, PC, PR, RY, G = X.np, X.torch, X.MC, X.PC, X.PR, X.RY, X.G
    rp = E2F.Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    dataset = "indoor" if sset == "valsel" else "outdoor"
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, recs, dataset
    rows, bel = [], {k: [] for k in ("sid", "t", "view", "is_cand", "R", "T", "k_M", "idx", "cls", "z", "tcap", "tbox")}
    F100, F10 = X.D.fusion_class(100), X.D.fusion_class(10)
    lams = cfg["lambdas"]
    blocks = sorted({t - t % 10 for t in sts})

    def refresh(region, f100, r):
        geo = PC.Geometry.from_fusion(*region, f100, dev)
        coarse = geo.coarse()
        from seen2scene.futureview import model as M
        h52 = model.encode(M.encoder_input(coarse.to(dev)))
        hv5 = v5core.pred.encode(coarse) if v5core is not None else None
        eps = PR.noise(dev, 4, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk))) if v5core is not None else None
        del geo
        return h52, hv5, eps

    def state(st, rec, region, f100, f10, H):
        h52, hv5, eps = H
        geo = VC.b2_geometry(PC, MC, region, f100, f10, dev)
        k_M = float(geo.fr.k_M)
        row = {"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": st["t"], "tag": tag, "stratum": stratum,
               "phase": X.phase_of(dataset, st["t"]), "n_legal": len(rec["legal_R"]), "timing": {}}
        torch.cuda.synchronize(); t0 = time.time()
        views = [geo.view(R, Tt) for R, Tt in zip(rec["legal_R"], rec["legal_T"])]
        raw = SC52.score_state(model, h52, geo, views, k_M, rp.z0, cfg)
        torch.cuda.synchronize(); row["timing"]["v52_score_s"] = time.time() - t0
        row["v52"] = {"s_vis": raw["s_vis"], "p_surf": raw["p_surf"], "p_space": raw["p_space"], "diag": raw["diag"], "state": raw["state"]}
        for lam in lams:
            sc, parts = SC52.combine(raw, lam, cfg["eta"], cfg["floors"])
            row["v52"][f"score_l{lam:g}"] = sc
            row["v52"][f"parts_l{lam:g}"] = parts
        if v5core is not None:
            out, tm, _ = PC.score_state(v5core, geo, hv5, rec["legal_R"], rec["legal_T"], eps, rp.z0)
            row["v5_replay"] = out["distp1_k"]
        # ---------------- evaluation side (after all scores)
        root, dname = (os.path.join(E2F.EP.W, "valsel"), "indoor") if sset == "valsel" else ((X.ORIG_W if stratum == "B0" else X.P.ROOT_B2), "outdoor")
        orc = np.load(os.path.join(root, "states", dname, st["sid"], "oracle.npz"))
        fh = {}
        for j in st["cand"]:
            fh = VC.merge_fh(fh, VC.firsthit_metrics(model, h52, views[j], orc["tstar"][j]))
        row["firsthit"] = fh
        key = hashlib.sha256(f"{st['sid']}".encode()).hexdigest()[:16]
        ext = torch.as_tensor(np.asarray(geo.ext.cpu() if hasattr(geo.ext, "cpu") else geo.ext, np.float64), device=dev).float()
        for j in st["cand"]:
            idx, cls, z, tc = belief_rows(model, h52, views[j], key, j)
            o = views[j]["o"].double().cpu().numpy(); dd = views[j]["d"][idx].double().cpu().numpy(); en = ext.double().cpu().numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (0.0 - o[None]) / dd, (en[None] - o[None]) / dd
                tbox = np.where(dd > 0, t2, np.where(dd < 0, t1, np.inf)).min(1)
            for k, v in (("sid", st["sid"]), ("t", st["t"]), ("view", j), ("is_cand", True), ("R", np.asarray(rec["legal_R"][j], np.float64)),
                         ("T", np.asarray(rec["legal_T"][j], np.float64)), ("k_M", k_M), ("idx", idx.cpu().numpy().astype(np.int16)),
                         ("cls", cls.cpu().numpy().astype(np.int8)), ("z", z.cpu().numpy().astype(np.float16)), ("tcap", tc.cpu().numpy()), ("tbox", tbox.astype(np.float32))):
                bel[k].append(v)
        rows.append(row)
        del geo

    if sset == "valsel":
        for r in blocks:
            region = rp.region_of(recs[r])
            f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
            for q in range(r + 1):
                f100.integrate_record(recs[q]); f10.integrate_record(recs[q])
            H = refresh(region, f100, r)
            last = max(t for t in sts if t - t % 10 == r)
            for t in range(r, last + 1):
                if t > r:
                    f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
                if t in sts:
                    state(sts[t], recs[t], region, f100, f10, H)
            del f100, f10
    else:
        region = rp.frame
        f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
        H = None
        for t in range(T + 1):
            f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
            if t in blocks:
                H = refresh(region, f100, t)
            if t in sts:
                state(sts[t], recs[t], region, f100, f10, H)
        del f100, f10
    torch.cuda.empty_cache()
    if bel["sid"]:
        np.savez_compressed(os.path.join(od, f"belief_{tag}.npz"), scene=np.asarray([rows[0]["scene"]] * len(bel["sid"])),
                            **{k: np.asarray(v) if k in ("sid", "t", "view", "is_cand", "k_M") else np.stack([np.asarray(x) for x in v]) for k, v in bel.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--v5-replay", default=None)
    ap.add_argument("--tag", default="s_final")
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--max-eps", type=int, default=0)
    ap.add_argument("--score-cfg", default="score_v52.json")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5", "scripts"))
    torch = X.torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    model, sha = VC.load_model(a.ckpt, dev)
    cfg = VC.score_cfg(a.score_cfg)
    v5core = None
    if a.v5_replay:
        import v5_core
        v5_core.install()
        torch.use_deterministic_algorithms(True, warn_only=True)
        v5core = X.PC.Core(a.v5_replay, dev)
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(VC.W52, "eval_fixed", f"{a.set}_{a.tag}", a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({"v52": a.ckpt, "v52_sha256": sha, "score_cfg": cfg, "v5_replay": a.v5_replay}, open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    eps = E2F.states(a.set, a.stratum)[pi::pn]
    if a.max_eps:
        eps = eps[:a.max_eps]
    for tag, sts in eps:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        with torch.no_grad():
            rows = run_episode(a.set, a.stratum, tag, sts, model, v5core, dev, cfg, od)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)


if __name__ == "__main__":
    main()
