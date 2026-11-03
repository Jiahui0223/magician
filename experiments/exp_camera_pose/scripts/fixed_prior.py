"""Camera Pose Phase 1 -- Stage B, part 1: FrontierPrior of every legal candidate on the development fixed states
(seen2scene env, 1 GPU; no network). Same states, same legal candidates (in the same order as the historical score
lists), same refresh / region rules and the same B2 planner geometry (b2_geometry: D100 struct / FREE / cast volume,
S10 band_dil) as the V5 / V5.2 / V5.2-M fixed-state evaluations (exp6_v3 shared eval_fixed + e2 pilot replay).
Sets: MP3D VALSEL (90 states), Macarons++ Dev-3 B0 (150) and B2 (125). Output <out>/<set>/<stratum>/<episode>.jsonl:
per state sid, n_legal, F list, feature dicts, timings. Usage: python fixed_prior.py --set valsel|dev3 [--stratum B0|B2] --out DIR"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, R)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v5_2", "scripts"))
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v3", "shared"))
import v52_common as VC  # noqa: E402  (b2_geometry)
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F


def run_episode(sset, stratum, tag, sts, dev):
    import frontier as FR
    torch, MC, PC = X.torch, X.MC, X.PC
    rp = E2F.Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    F100, F10 = X.D.fusion_class(100), X.D.fusion_class(10)
    blocks = sorted({t - t % 10 for t in sts})
    rows = []

    def state(st, rec, region, f100, f10):
        geo = VC.b2_geometry(PC, MC, region, f100, f10, dev)
        torch.cuda.synchronize(); t0 = time.time()
        views = [geo.view(Rr, Tt) for Rr, Tt in zip(rec["legal_R"], rec["legal_T"])]
        torch.cuda.synchronize(); t1 = time.time()
        feats = FR.features(views, geo)
        t2 = time.time()
        rows.append({"sid": st["sid"], "scene": st["scene"], "start": st["start"], "t": st["t"], "tag": tag, "stratum": stratum, "n_legal": len(rec["legal_R"]),
                     "F": [f["F"] for f in feats], "feats": feats, "view_s": t1 - t0, "frontier_s": t2 - t1})
        del geo, views

    if sset == "valsel":
        for r in blocks:
            region = rp.region_of(recs[r])
            f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
            for q in range(r + 1):
                f100.integrate_record(recs[q]); f10.integrate_record(recs[q])
            last = max(t for t in sts if t - t % 10 == r)
            for t in range(r, last + 1):
                if t > r:
                    f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
                if t in sts:
                    state(sts[t], recs[t], region, f100, f10)
            del f100, f10
    else:
        region = rp.frame
        f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
        for t in range(T + 1):
            f100.integrate_record(recs[t]); f10.integrate_record(recs[t])
            if t in sts:
                state(sts[t], recs[t], region, f100, f10)
        del f100, f10
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--max-eps", type=int, default=0)
    a = ap.parse_args()
    X.setup(a.stratum)
    dev = X.torch.device("cuda")
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(a.out, a.set, a.stratum)
    os.makedirs(od, exist_ok=True)
    eps = E2F.states(a.set, a.stratum)[pi::pn]
    if a.max_eps:
        eps = eps[:a.max_eps]
    for tag, sts in eps:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        with X.torch.no_grad():
            rows = run_episode(a.set, a.stratum, tag, sts, dev)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), flush=True)


if __name__ == "__main__":
    main()
