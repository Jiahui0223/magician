"""EXP6 V3 Task 2 / Experiment 1 -- all scorer variants on the fixed states (seen2scene env, 1 GPU per process; protocol §3-§4).

Reuses the Experiment A replay (ablation_study/shared/ab_score.py: Pool CL = EXP5 A1 / A2 closed-loop states, Pool FX =
EXP6 G2 states; sparse + D100 fusions of the records 0..t, Task 2 region / refresh rules) and the Experiment A cache
(K = 4 decoded first-hit distances of B2 / E2D / T0 / T1 per candidate, same noise as the frozen runs) -- no model
inference. Per state and candidate: the scorer variants of t2_scorers on (a) the true depth t* of oracle.npz (E1-GT,
+ the global variants) and (b) every cached sample of every model (E1-Pred; K-mean reported by the analysis).
Checks: S0 of t* == the Experiment A gtd_distp1 (bitwise), S0 of the cached samples == the cached distp1_k (bitwise).
Output work_dirs/exp6_v3/task2/exp1/scores/<pool>/<dataset>/<sid>.json
Usage: python e1_score_fixed.py --pool cl|fx --dataset indoor|outdoor --part i/n [--limit N] [--tags a,b]
"""
import argparse
import json
import os
import socket
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
T2 = os.path.dirname(os.path.dirname(HERE))
E6 = os.path.dirname(T2)
sys.path.insert(0, os.path.join(T2, "common"))
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
import t2_scorers as TS  # noqa: E402
import ab_score as AB  # noqa: E402

X = AB.X
REPO = AB.REPO
WT2 = os.path.join(REPO, "work_dirs", "exp6_v3", "task2", "exp1")
MODELS = ("B2", "T1", "T0", "E2D")
EPS_MODEL_INDOOR = 0.08          # mp3d_sim.EPS_MODEL (coverage tolerance, model units) -> runtime = 0.08 * k_I
EPS_RT_OUTDOOR = 1.0             # 2 * TEST_RESOLUTION * scene_scale_factor (m_t6_sim: sim.eps)


def out_path(pool, dataset, sid):
    return os.path.join(WT2, "scores", pool, dataset, f"{sid}.json")


@torch.no_grad()
def run_episode(pool, dataset, tag, sts, dev, log):
    torch_, np_, MC, PC = X.torch, X.np, X.MC, X.PC
    ep = AB.Episode(pool, dataset, tag, sts, dev)
    todo = [s for s in sts if not os.path.exists(out_path(pool, dataset, s["sid"]))]
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
    hist = TS.Hist()
    hist_n = 0
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
            while hist_n <= t:                                   # historical observed surface = records 0..t
                hist.add(recs[hist_n]["sample_points"])
                hist_n += 1
            hist.build()
            rec = recs[t]
            geo10 = PC.Geometry.from_fusion(*region, fus10, dev)
            geo100 = PC.Geometry.from_fusion(*region, fus100, dev)
            fr = geo10.fr
            k_M = float(fr.k_M)
            eps_rt = EPS_MODEL_INDOOR * k_M if dataset == "indoor" else EPS_RT_OUTDOOR
            for st in by_t[t]:
                t_state0 = time.time()
                sid = st["sid"]
                cache = np.load(os.path.join(AB.WA, "cache", pool, dataset, f"{sid}.npz"))
                cj = json.load(open(os.path.join(AB.WA, "cache", pool, dataset, f"{sid}.json")))
                orc = np.load(os.path.join(ep.oracle_dir(sid), "oracle.npz"))
                assert np.array_equal(np.asarray(orc["legal_idx"]), np.asarray(rec["legal_idx"])), ("legal_idx", sid)
                res = {"sid": sid, "pool": pool, "dataset": dataset, "tag": tag, "source": ep.source, "scene": st["scene"], "start": st["start"],
                       "t": t, "n_legal": len(rec["legal_idx"]), "cand": list(st["cand"]), "eps_rt": eps_rt, "z0": float(ep.z0), "k_M": k_M,
                       "n_hist_points": hist.n, "oracle_gain": [float(g) for g in orc["gain"]], "scores": {}, "diag": {}, "checks": {}}
                views10 = [geo10.view(R, Tt) for R, Tt in zip(rec["legal_R"], rec["legal_T"])]
                views100 = [geo100.view(R, Tt) for R, Tt in zip(rec["legal_R"], rec["legal_T"])]
                for v in views100:
                    v["struct"] = geo100.struct
                # ---- E1-GT: true depth (evaluation side)
                gt_scores, gt_diag = [], []
                for j, (v10, v100) in enumerate(zip(views10, views100)):
                    ts = torch_.as_tensor(orc["tstar"][j], device=dev, dtype=v10["d"].dtype)
                    tt = torch_.where(ts <= v10["pr"]["tmax"], ts, torch_.full_like(ts, float("inf")))
                    tg = torch_.where(ts <= v10["tcap"], ts, torch_.full_like(ts, float("inf")))
                    sc, dg = TS.score_view(tt, v10, v100, geo10, hist, eps_rt, ep.z0, fr, t_global=tg, want_global=True)
                    gt_scores.append(sc); gt_diag.append(dg)
                res["scores"]["GT"] = {k: [s[k] for s in gt_scores] for k in gt_scores[0]} if gt_scores else {}
                res["diag"]["GT"] = gt_diag
                res["checks"]["GT_S0_eq_A"] = bool(np.array_equal(np.asarray(res["scores"]["GT"].get("S0", []), float), np.asarray(cj["gtd_distp1"], float)))
                res["checks"]["GT_S0_maxrel_A"] = float(np.max(np.abs(np.asarray(res["scores"]["GT"]["S0"]) - np.asarray(cj["gtd_distp1"])) / np.maximum(1e-300, np.abs(cj["gtd_distp1"])))) if gt_scores else 0.0
                # ---- E1-Pred: cached K = 4 decoded depths of every model (planner side quantities only)
                for m in MODELS:
                    tk = torch_.as_tensor(cache[f"{m}_t"], device=dev)              # [n_legal, K, R]
                    K = tk.shape[1]
                    per = []                                                       # [n_legal][K] dict
                    dgs = []
                    for j, (v10, v100) in enumerate(zip(views10, views100)):
                        row, drow = [], []
                        for k in range(K):
                            sc, dg = TS.score_view(tk[j, k].to(v10["d"].dtype), v10, v100, geo10, hist, eps_rt, ep.z0, fr)
                            row.append(sc); drow.append(dg)
                        per.append(row); dgs.append(drow)
                    res["scores"][m] = {kk: [[per[j][k][kk] for k in range(K)] for j in range(len(per))] for kk in per[0][0]} if per else {}
                    res["diag"][m] = [{kk: float(np.mean([d[kk] for d in drow if d[kk] is not None])) if any(d[kk] is not None for d in drow) else None
                                       for kk in ("n_hit", "n_new0", "n_in_band", "n_behind_dense", "n_thin_front", "mean_cos", "mean_z_rt", "mean_novelty_lin", "frac_novel_hard", "n_hit_in_dense_empty")}
                                      for drow in dgs]
                    ref = np.asarray(cj["models"][m]["distp1_k"], float)
                    new = np.asarray(res["scores"][m]["S0"], float) if per else np.zeros((0, K))
                    res["checks"][f"{m}_S0_eq_cache"] = bool(new.shape == ref.shape and np.array_equal(new, ref))
                    res["checks"][f"{m}_S0_maxrel_cache"] = float(np.max(np.abs(new - ref) / np.maximum(1e-300, np.abs(ref)))) if new.size else 0.0
                res["seconds"] = time.time() - t_state0
                p = out_path(pool, dataset, sid)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                json.dump(res, open(p + ".tmp", "w"), default=float)
                os.replace(p + ".tmp", p)
                n_done += 1
                log(f"{pool} {dataset} {tag} t={t} sid={sid} {res['seconds']:.1f}s GT_S0==A {res['checks']['GT_S0_eq_A']} B2_S0==cache {res['checks']['B2_S0_eq_cache']} hist {hist.n}")
                del views10, views100
            del geo10, geo100
            torch_.cuda.empty_cache()
    del fus100, fus10
    torch_.cuda.empty_cache()
    return n_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["cl", "fx"], required=True)
    ap.add_argument("--dataset", choices=["indoor", "outdoor"], required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tags", default="")
    a = ap.parse_args()
    X.setup("B2" if (a.pool == "fx" and a.dataset == "outdoor") else "B0")
    torch_ = X.torch
    torch_.use_deterministic_algorithms(True, warn_only=True)
    torch_.backends.cudnn.deterministic = True
    torch_.backends.cudnn.benchmark = False
    dev = torch_.device("cuda")
    eps_all = AB.episodes(a.pool, a.dataset)
    if a.tags:
        eps_all = [(tg, s) for tg, s in eps_all if tg.split("|", 1)[1] in a.tags.split(",") or tg in a.tags.split(",")]
    pi, pn = map(int, a.part.split("/"))
    os.makedirs(os.path.join(WT2, "claims"), exist_ok=True)
    os.makedirs(os.path.join(WT2, "logs"), exist_ok=True)
    lp = os.path.join(WT2, "logs", f"score_{a.pool}_{a.dataset}_{pi}of{pn}.log")

    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        open(lp, "a").write(line + "\n")

    total = 0
    for tg, sts in eps_all[pi::pn]:
        if a.limit:
            sts = sts[:a.limit]
        if all(os.path.exists(out_path(a.pool, a.dataset, s["sid"])) for s in sts):
            continue
        claim = os.path.join(WT2, "claims", f"{a.pool}_{a.dataset}_{tg.replace('|', '_')}.claim")
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.strftime('%F %T')}\n".encode()); os.close(fd)
        except FileExistsError:
            continue
        tb = time.time()
        n = run_episode(a.pool, a.dataset, tg, sts, dev, log)
        total += n
        log(f"episode {tg}: {n} states in {time.time() - tb:.0f}s; peak {torch_.cuda.max_memory_allocated() / 2 ** 30:.1f} GB")
        torch_.cuda.reset_peak_memory_stats()
    log(f"DONE part {a.part}: {total} states")


if __name__ == "__main__":
    main()
