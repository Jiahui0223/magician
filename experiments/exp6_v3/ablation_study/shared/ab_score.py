"""EXP6 V3 Ablation -- K = 4 predictions, DistP1 and GT-depth DistP1 of the frozen models on the fixed states of both
pools (seen2scene env, 1 GPU per process; protocol.md v1.0 §1-§2).

Pool cl : EXP5 A1 / A2 closed-loop states (Task 2 service records of V2O / V2M / MAG-G / MAG-B), noise = the EXP5 seeds
Pool fx : EXP6 G2 states -- valsel (E2 trajectories, E2 seeds) and dev3 stratum B2 (Task 3 B2 trajectories, A1 seeds)
Models  : B2 (V2-Mix, D100 input), E2D (E2-D100), T0, T1 (V3; D100 -> VAE); every model scored with the SPARSE band_dil.
Planner side first (records only): D100 and sparse fusions of records 0..t (indoor: n_win = 5 region of the refresh
step r = t - t % 10; outdoor: one grid), encoder input and noise at r, PC.score_state(keep_hits) on the D100 geometry,
distp1_k on the sparse geometry. Evaluation side afterwards: GT-depth DistP1 from oracle.npz:tstar (EXP5 S2 rule),
regression against the EXP6 / E2 fixed-state rows where the state exists there.
Output work_dirs/exp6_v3/ablation_study/cache/<pool>/<dataset>/<sid>.npz (+ .json)
Usage: python ab_score.py --pool cl|fx --dataset indoor|outdoor --part i/n [--limit N] [--tags a,b]
"""
import argparse
import glob
import hashlib
import json
import os
import socket
import sys
import time

import torch

from seen2scene import PROJECT_ROOT
from seen2scene.v3 import predictor as V3P

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import e2_fixed as E2F  # noqa: E402

X = E2F.X
W6 = os.path.join(REPO, "work_dirs", "exp6_v3")
WA = os.path.join(W6, "ablation_study")
E2D_CKPT = os.path.join(E2F.EP.W, "training", "d100", "ema_005000.pt")
T0_CKPT = os.path.join(W6, "task0_vae", "checkpoints", "final.pt")
T1_CKPT = os.path.join(W6, "task1_xattn", "checkpoints", "final.pt")
MODELS = ("B2", "E2D", "T0", "T1")


class V3Core:
    def __init__(self, ckpt, dev):
        self.model, self.pred, self.ckpt_sha = V3P.load_core(ckpt, dev)
        self.ckpt = ckpt

    def encode_for(self, fus, geo):
        return self.pred.encode_fusion(fus)


class V2Core:
    def __init__(self, core, ckpt):
        self.core, self.pred, self.ckpt_sha, self.ckpt = core, core.pred, core.ckpt_sha, ckpt

    def encode_for(self, fus, geo):
        return self.pred.encode(geo.coarse())


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_cores(dev):
    v2 = X.S2.load_cores(dev)["V2M"]
    return {"B2": V2Core(v2, os.path.join(REPO, X.A.MODELS["V2M"]["path"])),
            "E2D": V2Core(X.PC.Core(E2D_CKPT, dev), E2D_CKPT),
            "T0": V3Core(T0_CKPT, dev), "T1": V3Core(T1_CKPT, dev)}


# ------------------------------------------------------------------------------------------------- state pools
class Episode:
    """One source episode of a pool: states, record path, region / frame, z0, noise key, oracle dir."""

    def __init__(self, pool, dataset, tag, sts, dev):
        self.pool, self.dataset, self.tag, self.sts = pool, dataset, tag, sts
        st0 = sts[0]
        if pool == "cl":
            src = st0["sources"][0]["source"]
            ep = X.S2.episode_of(dataset, st0)
            rp = X.S2.StateReplay(dataset, ep, src, {}, dev)
            self.svc = lambda t: X.A.svc_path(dataset, src, ep, t)
            self.key = rp.seed_key
            self.z0 = rp.z0
            self.source = src
            if dataset == "indoor":
                self.region_of = lambda rec: rp.F.region_geometry(rp.card, rec["cam_center"])[:3]
                self.frame = None
            else:
                self.frame = rp.frame[:3]
            self.oracle_dir = lambda sid: os.path.join(X.ORIG_W, "states", dataset, sid)
        else:
            sset = "valsel" if dataset == "indoor" else "dev3"
            rp = E2F.Rep(sset, st0, dev)
            self.svc, self.key, self.z0, self.frame = rp.svc, rp.key, rp.z0, rp.frame
            self.source = st0["sources"][0].get("source", st0.get("policy", "E2"))
            if dataset == "indoor":
                self.region_of = rp.region_of
                self.oracle_dir = lambda sid: os.path.join(E2F.EP.W, "valsel", "states", "indoor", sid)
            else:
                self.oracle_dir = lambda sid: os.path.join(X.P.ROOT_B2, "states", "outdoor", sid)


def episodes(pool, dataset):
    if pool == "cl":
        man = json.load(open(os.path.join(X.ORIG_W, "manifests", f"states_{dataset}.json")))
        return [(f"{src}|{tag}", sts) for (src, tag), sts in X.S2.groups(man)]
    if dataset == "indoor":
        return [(f"E2|{tag}", sts) for tag, sts in E2F.states("valsel", "B0")]
    return [(f"B2|{tag}", sts) for tag, sts in E2F.states("dev3", "B2")]


def out_paths(pool, dataset, sid):
    d = os.path.join(WA, "cache", pool, dataset)
    return os.path.join(d, f"{sid}.npz"), os.path.join(d, f"{sid}.json")


# ------------------------------------------------------------------------------------------------- regression rows
def reference_rows(pool, dataset):
    """sid -> {model: distp1_k} from the EXP6 (T0 / T1 / B2) and E2 (D100 -> E2D) fixed-state rows."""
    ref = {}
    sets = [("valsel", "B0")] if dataset == "indoor" else ([("dev3", "B0")] if pool == "cl" else [("dev3", "B2")])
    for sset, stratum in sets:
        for f in glob.glob(os.path.join(W6, "eval_fixed", f"{sset}_s_final", stratum, "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                ref.setdefault(r["sid"], {}).update({m: r["scores"][m] for m in ("B2", "T0", "T1") if m in r["scores"]})
        for f in glob.glob(os.path.join(E2F.EP.W, "fixed", f"{sset}_s5000", stratum, "*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                if "D100" in r["scores"]:
                    ref.setdefault(r["sid"], {})["E2D"] = r["scores"]["D100"]
    return ref


# ------------------------------------------------------------------------------------------------- one episode
@torch.no_grad()
def run_episode(pool, dataset, tag, sts, cores, dev, ref, log):
    torch, np, MC, PC, PR, RY, SCO = X.torch, X.np, X.MC, X.PC, X.PR, X.RY, X.SCO
    ep = Episode(pool, dataset, tag, sts, dev)
    todo = [s for s in sts if not os.path.exists(out_paths(pool, dataset, s["sid"])[1])]
    if not todo:
        return 0
    by_t = {}
    for s in todo:
        by_t.setdefault(int(s["t"]), []).append(s)
    T = max(by_t)
    recs = [MC.load_service_record(ep.svc(t)) for t in range(T + 1)]
    for t, r in enumerate(recs):
        assert int(r["t"]) == t, (tag, t, int(r["t"]))
    F100, F10 = X.D.fusion_class(100), X.D.fusion_class(10)
    blocks = sorted({t - t % 10 for t in by_t})
    n_done = 0
    fus100 = fus10 = region = None
    if dataset == "outdoor":
        region = ep.frame
        fus100, fus10 = F100(*region, device=dev), F10(*region, device=dev)
        integrated = -1
    for r in blocks:
        if dataset == "indoor":
            region = ep.region_of(recs[r])
            fus100, fus10 = F100(*region, device=dev), F10(*region, device=dev)
            for q in range(r + 1):
                fus100.integrate_record(recs[q])
                fus10.integrate_record(recs[q])
        else:
            while integrated < r:
                integrated += 1
                fus100.integrate_record(recs[integrated])
                fus10.integrate_record(recs[integrated])
        geo100 = PC.Geometry.from_fusion(*region, fus100, dev)
        h = {m: cores[m].encode_for(fus100, geo100) for m in MODELS}
        eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(ep.key(tr, kk)))
        del geo100
        last = max(t for t in by_t if t - t % 10 == r)
        for t in range(r, last + 1):
            if t > r:
                if dataset == "indoor":
                    fus100.integrate_record(recs[t]); fus10.integrate_record(recs[t])
                else:
                    while integrated < t:
                        integrated += 1
                        fus100.integrate_record(recs[integrated]); fus10.integrate_record(recs[integrated])
            if t not in by_t:
                continue
            rec = recs[t]
            for st in by_t[t]:
                t_state0 = time.time()
                sid = st["sid"]
                res = {"sid": sid, "pool": pool, "dataset": dataset, "tag": tag, "source": ep.source, "scene": st["scene"],
                       "start": st["start"], "t": t, "refresh": r, "n_legal": len(rec["legal_idx"]), "cand": list(st["cand"]),
                       "models": {}, "regression": {}, "timing": {}}
                # ---- planner side: predictions on the D100 geometry, DistP1 on the sparse geometry
                geo100 = PC.Geometry.from_fusion(*region, fus100, dev)
                hits = {}
                for m in MODELS:
                    tb = time.time()
                    out, tm, _ = PC.score_state(cores[m], geo100, h[m], rec["legal_R"], rec["legal_T"], eps, ep.z0, keep_hits=True)
                    torch.cuda.synchronize()
                    hits[m] = out["hits"]
                    res["timing"][m] = dict(tm, total_s=time.time() - tb)
                del geo100
                geo10 = PC.Geometry.from_fusion(*region, fus10, dev)
                k_M = float(geo10.fr.k_M)
                views = [geo10.view(R, Tt) for R, Tt in zip(rec["legal_R"], rec["legal_T"])]
                save = {}
                for m in MODELS:
                    sc = X.distp1_k(views, [hh["t"] for hh in hits[m]], geo10, k_M, ep.z0)
                    res["models"][m] = {"distp1_k": sc, "distp1": [float(np.mean(s)) for s in sc]}
                    save[f"{m}_t"] = np.stack([hh["t"].float().cpu().numpy() for hh in hits[m]])
                    save[f"{m}_hit_u"] = np.stack([hh["hit_u"].cpu().numpy() for hh in hits[m]])
                    if sid in ref and m in ref[sid]:
                        a, b = np.asarray(sc, float), np.asarray(ref[sid][m], float)
                        res["regression"][m] = {"bitwise": bool(a.shape == b.shape and np.array_equal(a, b)),
                                                "maxdiff": float(np.max(np.abs(a - b))) if a.shape == b.shape else None}
                pr_keys = ("valid", "cls", "n_u", "d_lo", "d_hi", "t_known", "tmax")
                geo_np = {k: np.stack([v["pr"][k].float().cpu().numpy() if v["pr"][k].dtype.is_floating_point else v["pr"][k].cpu().numpy()
                                       for v in views]) for k in pr_keys}
                geo_np.update({"o": np.stack([v["o"].float().cpu().numpy() for v in views]), "d": np.stack([v["d"].float().cpu().numpy() for v in views]),
                               "tcap": np.stack([v["tcap"].float().cpu().numpy() for v in views]), "dz": np.stack([v["dz"].float().cpu().numpy() for v in views])})
                # ---- evaluation side (GT) only from here: GT-depth DistP1 of the same candidates (EXP5 S2 rule)
                orc = np.load(os.path.join(ep.oracle_dir(sid), "oracle.npz"))
                assert np.array_equal(np.asarray(orc["legal_idx"]), np.asarray(rec["legal_idx"])), ("legal_idx differs from oracle", sid)
                gtd = []
                for j, v in enumerate(views):
                    ts = torch.as_tensor(orc["tstar"][j], device=dev, dtype=v["d"].dtype)
                    tt = torch.where(ts <= v["pr"]["tmax"], ts, torch.full_like(ts, float("inf")))
                    _, w = SCO.new_set_from_depth(tt, v["o"], v["d"], v["dz"], k_M, geo10.dims, geo10.domain, geo10.caster.band_dil, ep.z0)
                    gtd.append(float(w.sum()))
                res["gtd_distp1"] = gtd
                res["oracle_gain"] = [float(g) for g in orc["gain"]]
                res["k_M"], res["z0"], res["seconds"] = k_M, float(ep.z0), time.time() - t_state0
                res["ckpt_sha256"] = {m: cores[m].ckpt_sha for m in MODELS}
                pn, pj = out_paths(pool, dataset, sid)
                os.makedirs(os.path.dirname(pn), exist_ok=True)
                np.savez_compressed(pn + ".tmp.npz", **save, **geo_np, k_M=np.asarray(k_M), z0=np.asarray(ep.z0))
                os.replace(pn + ".tmp.npz", pn)
                json.dump(res, open(pj + ".tmp", "w"), default=float)
                os.replace(pj + ".tmp", pj)
                del geo10, views, hits
                torch.cuda.empty_cache()
                n_done += 1
                reg = {m: v["bitwise"] for m, v in res["regression"].items()}
                log(f"{pool} {dataset} {tag} t={t} sid={sid} {res['seconds']:.0f}s regression={reg or '-'}")
    del fus100, fus10
    torch.cuda.empty_cache()
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["cl", "fx"], required=True)
    ap.add_argument("--dataset", choices=["indoor", "outdoor"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=0, help="states per episode (smoke)")
    ap.add_argument("--tags", default="", help="comma-separated episode tags (smoke)")
    a = ap.parse_args()
    X.setup("B2" if (a.pool == "fx" and a.dataset == "outdoor") else "B0")
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    cores = load_cores(dev)
    ref = reference_rows(a.pool, a.dataset)
    eps_all = episodes(a.pool, a.dataset)
    if a.tags:
        eps_all = [(tg, s) for tg, s in eps_all if tg.split("|", 1)[1] in a.tags.split(",") or tg in a.tags.split(",")]
    pi, pn = map(int, a.part.split("/"))
    os.makedirs(os.path.join(WA, "cache", "claims"), exist_ok=True)
    os.makedirs(os.path.join(WA, "manifests"), exist_ok=True)
    man = {"pool": a.pool, "dataset": a.dataset, "part": a.part, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
           "ckpt": {m: {"path": cores[m].ckpt, "sha256": cores[m].ckpt_sha} for m in MODELS}, "reference_states": len(ref),
           "started": time.strftime("%F %T")}
    json.dump(man, open(os.path.join(WA, "manifests", f"score_{a.pool}_{a.dataset}_{pi}of{pn}_{int(time.time())}.json"), "w"), indent=1)
    lp = os.path.join(WA, "logs", f"score_{a.pool}_{a.dataset}_{pi}of{pn}.log")
    os.makedirs(os.path.dirname(lp), exist_ok=True)

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        open(lp, "a").write(line + "\n")

    total = 0
    for tg, sts in eps_all[pi::pn]:
        if a.limit:
            sts = sts[:a.limit]
        if all(os.path.exists(out_paths(a.pool, a.dataset, s["sid"])[1]) for s in sts):
            continue
        claim = os.path.join(WA, "cache", "claims", f"{a.pool}_{a.dataset}_{tg.replace('|', '_')}.claim")
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.strftime('%F %T')}\n".encode()); os.close(fd)
        except FileExistsError:
            continue
        tb = time.time()
        n = run_episode(a.pool, a.dataset, tg, sts, cores, dev, ref, log)
        total += n
        log(f"episode {tg}: {n} states in {time.time() - tb:.0f}s; peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
    log(f"DONE part {a.part}: {total} states")


if __name__ == "__main__":
    main()
