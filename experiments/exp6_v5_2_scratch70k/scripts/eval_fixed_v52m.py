"""EXP6 V5.2-M -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process; read-only diagnostics).

Same states, candidates, regions, refresh rules and B2 maps (D100 input / structure / FREE, S10 band_dil novelty) as
the V5 / V5.2 fixed-state evaluations (exp6_v5_2/scripts/eval_g1_g2.py, whose replay is reused unchanged in structure):
MP3D VALSEL 90 states, Macarons++ Dev-3 B0 150 + B2 125. Per state and legal candidate: S_vis / P_surf / P_space,
lambda 0 / 0.1 scores (score config of the evaluated checkpoint: floors from its own Train-6 S-c calibration), timings.
Evaluation side (after all scores): first-hit metrics of every candidate view vs the oracle t* (NLL, top interval,
no-hit reliability, high-confidence false no-hit, FREE violations, depth error by distance band, UNKNOWN vs known rays);
belief dump of 256 rays per candidate view (bin logits + edges) for the eval-side multi-surface GT.
Output <out>/<set>/<stratum>/<ep>.jsonl (+ belief_<ep>.npz).
Usage: python eval_fixed_v52m.py --set valsel|dev3 --stratum B0|B2 --ckpt CK --score-cfg CFG.json --out DIR [--part i/n]"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5_2", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "shared"))
import v52_common as VC  # noqa: E402  (b2_geometry)
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
N_UNK, N_ALL = 128, 128
NB = 10


def load_model(ckpt, dev):
    import torch
    from seen2scene.futureview import v52m_model as VM
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert ck["model_cfg"].get("arch") == "v52m"
    m = VM.V52MModel(ck["model_cfg"])
    m.load_state_dict(ck["ema"], strict=True)
    return m.to(dev).eval(), VC.sha256_file(ckpt), int(ck.get("step", -1))


def fh_metrics(out, view, tstar, G):
    """First-hit metrics of all rays of one view (sums / counts)."""
    import numpy as np
    import torch
    ts = torch.as_tensor(np.asarray(tstar, np.float64), device=out["edges"].device).float()
    e = out["edges"]
    B = e.shape[1] - 1
    valid = ~(ts < e[:, 0])
    hit = torch.isfinite(ts) & (ts < e[:, -1])
    idx = torch.searchsorted(e.contiguous(), ts.clamp(max=1e9)[:, None].contiguous())[:, 0] - 1
    idx = torch.where(hit, idx.clamp(0, B - 1), torch.full_like(idx, B))
    ffree = torch.gather(torch.cat([out["free_frac"], torch.zeros_like(out["free_frac"][:, :1])], 1), 1, idx[:, None])[:, 0]
    conflict = hit & (ffree >= 0.999)
    lp = torch.where(idx == B, torch.log(out["p_nohit"].clamp(min=1e-30)), torch.gather(out["log_pf"], 1, idx.clamp(max=B - 1)[:, None])[:, 0])
    allp = torch.cat([out["p_first"], out["p_nohit"][:, None]], 1)
    top = allp.argmax(1) == idx
    pn = out["p_nohit"]
    mode_t = torch.gather(out["t_mid"], 1, out["p_first"].argmax(1, keepdim=True))[:, 0]
    unk = view["pr"]["cls"] == G.CLS_UNKNOWN
    acc = {}
    dist = ts / G.VS
    for grp, m0 in (("all", valid & ~conflict), ("unknown", valid & ~conflict & unk), ("known", valid & ~conflict & ~unk)):
        a = {"n": int(m0.sum()), "nll": float(-lp[m0].sum()), "top": int(top[m0].sum()), "n_hit": int((hit & m0).sum()),
             "hc_nohit_pred": int(((pn > 0.95) & m0).sum()), "hc_nohit_true_hit": int(((pn > 0.95) & m0 & hit).sum()),
             "cal_n": [0] * NB, "cal_p": [0.0] * NB, "cal_y": [0.0] * NB, "n_conflict": int((conflict & (unk if grp == "unknown" else (~unk if grp == "known" else torch.ones_like(unk)))).sum()),
             "free_mass_on_hit": 0.0}
        bi = torch.clamp((pn * NB).long(), max=NB - 1)
        for k in range(NB):
            s = (bi == k) & m0
            a["cal_n"][k] += int(s.sum()); a["cal_p"][k] += float(pn[s].sum()); a["cal_y"][k] += float((~hit[s]).float().sum())
        for lab, lo, hi in (("near", 0, 64), ("mid", 64, 256), ("far", 256, 1e9)):
            mm = m0 & hit & (pn < 0.5) & (dist >= lo) & (dist < hi)
            a[f"n_err_{lab}"] = int(mm.sum())
            a[f"err_{lab}"] = float(((mode_t - ts).abs() / G.VS)[mm].sum())
        acc[grp] = a
    return acc


def merge(a, b):
    for g, v in b.items():
        if g not in a:
            a[g] = json.loads(json.dumps(v))
            continue
        for k, x in v.items():
            a[g][k] = [p + q for p, q in zip(a[g][k], x)] if isinstance(x, list) else a[g][k] + x
    return a


def run_episode(sset, stratum, tag, sts, model, dev, cfg, od):
    np, torch, MC, PC, PR, RY, G = X.np, X.torch, X.MC, X.PC, X.PR, X.RY, X.G
    from seen2scene.futureview import model as M
    from seen2scene.futureview import v52m_score as SCM
    rp = E2F.Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    dataset = "indoor" if sset == "valsel" else "outdoor"
    rows = []
    bel = {k: [] for k in ("sid", "t", "view", "R", "T", "k_M", "idx", "cls", "logit_bins", "edges", "tcap", "tbox")}
    F100, F10 = X.D.fusion_class(100), X.D.fusion_class(10)
    blocks = sorted({t - t % 10 for t in sts})

    def refresh(region, f100):
        geo = PC.Geometry.from_fusion(*region, f100, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model.encode(M.encoder_input(geo.coarse().to(dev)))
        del geo
        return h

    def state(st, rec, region, f100, f10, h):
        geo = VC.b2_geometry(PC, MC, region, f100, f10, dev)
        k_M = float(geo.fr.k_M)
        row = {"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": st["t"], "tag": tag, "stratum": stratum,
               "phase": X.phase_of(dataset, st["t"]), "n_legal": len(rec["legal_R"]), "timing": {}}
        torch.cuda.synchronize(); t0 = time.time()
        views = [geo.view(Rr, Tt) for Rr, Tt in zip(rec["legal_R"], rec["legal_T"])]
        raw = SCM.score_state(model, h, geo, views, k_M, rp.z0, cfg)
        torch.cuda.synchronize(); row["timing"]["v52m_score_s"] = time.time() - t0
        row["v52m"] = {"s_vis": raw["s_vis"], "p_surf": raw["p_surf"], "p_space": raw["p_space"], "diag": raw["diag"], "state": raw["state"], "free_frac_mean": raw["free_frac_mean"]}
        for lam in cfg["lambdas"]:
            sc, parts = SCM.combine(raw, lam, cfg["eta"], cfg["floors"])
            row["v52m"][f"score_l{lam:g}"] = sc
            row["v52m"][f"parts_l{lam:g}"] = parts
        # ---------------- evaluation side
        root, dname = (os.path.join(E2F.EP.W, "valsel"), "indoor") if sset == "valsel" else ((X.ORIG_W if stratum == "B0" else X.P.ROOT_B2), "outdoor")
        orc = np.load(os.path.join(root, "states", dname, st["sid"], "oracle.npz"))
        fh = {}
        mem = model.memory(h)
        ext = np.asarray(geo.ext.cpu() if hasattr(geo.ext, "cpu") else geo.ext, np.float64)
        key = hashlib.sha256(st["sid"].encode()).hexdigest()[:16]
        for j in st["cand"]:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model.view_belief(h, views[j], mem)
            fh = merge(fh, fh_metrics(out, views[j], orc["tstar"][j], G))
            cls = views[j]["pr"]["cls"]
            g = torch.Generator(device="cpu"); g.manual_seed(int(hashlib.sha256(f"exp6-V52M|belief|{key}|{j}".encode()).hexdigest()[:15], 16))
            unk = torch.nonzero(cls.cpu() == G.CLS_UNKNOWN)[:, 0]
            pu = unk[torch.randperm(len(unk), generator=g)[:N_UNK]]
            free = torch.ones(cls.shape[0], dtype=torch.bool); free[pu] = False
            others = torch.nonzero(free)[:, 0]
            idx = torch.cat([pu, others[torch.randperm(len(others), generator=g)[:N_UNK + N_ALL - len(pu)]]])
            o = views[j]["o"].double().cpu().numpy(); dd = views[j]["d"][idx.to(dev)].double().cpu().numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = (0.0 - o[None]) / dd, (ext[None] - o[None]) / dd
                tbox = np.where(dd > 0, t2, np.where(dd < 0, t1, np.inf)).min(1)
            ii = idx.to(dev)
            for k, v in (("sid", st["sid"]), ("t", st["t"]), ("view", j), ("R", np.asarray(rec["legal_R"][j], np.float64)), ("T", np.asarray(rec["legal_T"][j], np.float64)),
                         ("k_M", k_M), ("idx", idx.numpy().astype(np.int16)), ("cls", cls[ii].cpu().numpy().astype(np.int8)),
                         ("logit_bins", out["logit_b"][ii].float().cpu().numpy().astype(np.float16)), ("edges", out["edges"][ii].float().cpu().numpy()),
                         ("tcap", views[j]["tcap"][ii].float().cpu().numpy()), ("tbox", tbox.astype(np.float32))):
                bel[k].append(v)
        row["firsthit"] = fh
        rows.append(row)
        del geo

    if sset == "valsel":
        for r in blocks:
            region = rp.region_of(recs[r])
            f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
            for q in range(r + 1):
                f100.integrate_record(recs[q]); f10.integrate_record(recs[q])
            h = refresh(region, f100)
            last = max(t for t in sts if t - t % 10 == r)
            for t in range(r, last + 1):
                if t > r:
                    f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
                if t in sts:
                    state(sts[t], recs[t], region, f100, f10, h)
            del f100, f10
    else:
        region = rp.frame
        f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
        h = None
        for t in range(T + 1):
            f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
            if t in blocks:
                h = refresh(region, f100)
            if t in sts:
                state(sts[t], recs[t], region, f100, f10, h)
        del f100, f10
    torch.cuda.empty_cache()
    if bel["sid"]:
        np.savez_compressed(os.path.join(od, f"belief_{tag}.npz"), scene=np.asarray([rows[0]["scene"]] * len(bel["sid"])),
                            **{k: np.asarray(v) if k in ("sid", "t", "view", "k_M") else np.stack([np.asarray(x) for x in v]) for k, v in bel.items()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--score-cfg", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--max-eps", type=int, default=0)
    a = ap.parse_args()
    X.setup(a.stratum)
    torch = X.torch
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    model, sha, step = load_model(a.ckpt, dev)
    cfg = json.load(open(a.score_cfg))
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(a.out, a.set, a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({"ckpt": a.ckpt, "sha256": sha, "step": step, "score_cfg": cfg}, open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    eps = E2F.states(a.set, a.stratum)[pi::pn]
    if a.max_eps:
        eps = eps[:a.max_eps]
    for tag, sts in eps:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        with torch.no_grad():
            rows = run_episode(a.set, a.stratum, tag, sts, model, dev, cfg, od)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)


if __name__ == "__main__":
    main()
