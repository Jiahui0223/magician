"""EXP6 V3 -- G1 / G2 on the development fixed states (seen2scene env, 1 GPU per process; protocol.md §4).

Same states, candidates, noise and replay rules as the E2 pilot (task3_full_depth_p1/e2_pilot/scripts/e2_fixed.py,
imported for its helpers): MP3D VALSEL (90 states) and Macarons++ Dev-3 (strata B0 / B2 of E1.2). Per episode two
replays (D100 map, then the sparse 10 % map), one map on the GPU at a time:
  pass D100   V3 models (T0 / T1 EMA checkpoints) encode the live D100 fusion (partial_tsdf.aggregate_fusion, the
              training input by construction); B2 (frozen V2-Mix on the D100 coarse grid) is re-scored as a replay
              check against the E2 rows (bitwise); predictions (K = 4, same noise) kept
  pass sparse DistP1 novelty of every model's predictions with the sparse band_dil (the frozen scoring map)
Evaluation side (after scoring): prediction errors vs t* grouped by ray (all / unknown / truly new / known), K = 4
spread, hit disagreement, spread-error Spearman (e2_fixed.pred_groups). Output e3/fixed/<set>/<stratum>/<tag>.jsonl.
Usage: python eval_fixed.py --set valsel|dev3 --stratum B0|B2 --t0 CKPT --t1 CKPT --part i/n
"""
import argparse
import glob
import json
import numpy as np
import os
import sys
import time

from seen2scene import PROJECT_ROOT
from seen2scene.v3 import predictor as V3P

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
import e2_fixed as E2F  # noqa: E402

W = os.path.join(REPO, "work_dirs", "exp6_v3", "eval_fixed")
X = E2F.X


class V3Core:
    def __init__(self, ckpt, dev):
        self.model, self.pred, self.ckpt_sha = V3P.load_core(ckpt, dev)
        self.ckpt = ckpt

    def encode_for(self, fus, geo):
        return self.pred.encode_fusion(fus)


class V2Core:
    def __init__(self, core):
        self.core, self.pred, self.ckpt_sha = core, core.pred, core.ckpt_sha

    def encode_for(self, fus, geo):
        return self.pred.encode(geo.coarse())


def run_episode(sset, stratum, tag, sts, cores, dev, ref):
    np, torch, MC, PC, PR, RY = X.np, X.torch, X.MC, X.PC, X.PR, X.RY
    rp = E2F.Rep(sset, sts[0], dev)
    rp.sset = sset
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    u = 7.0 if sset == "valsel" else 1.0
    rows = {s["sid"]: {"sid": s["sid"], "scene": s["scene"], "start": s["start"], "t": s["t"], "tag": tag, "stratum": stratum,
                       "phase": X.phase_of("indoor" if sset == "valsel" else "outdoor", s["t"]), "scores": {}, "pred": {}, "timing": {}}
            for s in sts.values()}
    hits, unk = {}, {}
    E = X.Episode.__new__(X.Episode)
    E.truth, E.recs, E.dataset = {}, recs, "indoor" if sset == "valsel" else "outdoor"
    names = list(cores)
    for x in (100, 10):
        cls = X.D.fusion_class(x)
        blocks = sorted({t - t % 10 for t in sts})

        def refresh(region, fus, r):
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            h = {m: cores[m].encode_for(fus, geo) for m in names} if x == 100 else {}
            eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
            del geo
            return h, eps

        def state(st, rec, region, fus, h, eps):
            sid, row = st["sid"], rows[st["sid"]]
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            k_M = float(geo.fr.k_M)
            if x == 100:
                for m in names:
                    t0 = time.time()
                    out, tm, views = PC.score_state(cores[m], geo, h[m], rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
                    torch.cuda.synchronize()
                    hits[(sid, m)] = [hh["t"].detach().cpu() for hh in out["hits"]]
                    row["timing"][m] = dict(tm, total_s=time.time() - t0)
                    unk[(sid, m)] = [(v["pr"]["cls"] == X.G.CLS_UNKNOWN).cpu().numpy() for v in views]
                row["views_dense"] = None
            else:
                views = [geo.view(R, T) for R, T in zip(rec["legal_R"], rec["legal_T"])]
                for m in names:
                    row["scores"][m] = X.distp1_k(views, hits[(sid, m)], geo, k_M, rp.z0)
                # evaluation side
                root, dname = (os.path.join(E2F.EP.W, "valsel"), "indoor") if sset == "valsel" else \
                    ((X.ORIG_W if stratum == "B0" else X.P.ROOT_B2), "outdoor")
                orc = np.load(os.path.join(root, "states", dname, sid, "oracle.npz"))
                truth, _ = E.truth_of(st, orc, None, k_M)
                for m in names:
                    acc, rl = {}, []
                    for j in st["cand"]:
                        new, ok, zg = truth[j]
                        dz = views[j]["dz"].double().cpu().numpy()
                        pg = E2F.pred_groups(hits[(sid, m)][j], orc["tstar"][j].astype(np.float64), dz, k_M, u, ok, new, unk[(sid, m)][j])
                        rl.append(pg.pop("rho", None))
                        for k, val in pg.items():
                            acc[k] = acc.get(k, 0) + val
                    rv = [r for r in rl if r is not None]
                    acc["rho_mean"] = float(np.mean(rv)) if rv else None
                    row["pred"][m] = acc
                if ref is not None and sid in ref and "B2" in names:
                    row["repro_B2_eq_E2"] = row["scores"]["B2"] == ref[sid]
                    # scores["B2"] = [n_cand][K] per-world novelty; the choice compares candidate means
                    cand = lambda x: (lambda a: a.mean(1) if a.ndim == 2 else a)(np.asarray(x, dtype=float))  # noqa: E731
                    try:
                        va, vb = np.asarray(row["scores"]["B2"], float).ravel(), np.asarray(ref[sid], float).ravel()
                        same_shape = va.shape == vb.shape
                        row["repro_B2_maxdiff"] = float(np.max(np.abs(va - vb))) if same_shape else None
                        row["repro_B2_maxrel"] = float(np.max(np.abs(va - vb) / np.maximum(np.abs(vb), 1e-12))) if same_shape else None
                        row["repro_B2_same_argmax"] = bool(same_shape and int(cand(row["scores"]["B2"]).argmax()) == int(cand(ref[sid]).argmax()))
                    except (TypeError, ValueError):
                        row["repro_B2_maxdiff"] = row["repro_B2_maxrel"] = None; row["repro_B2_same_argmax"] = False
                    # EVAL_FIXED_REPLAY_TOL (default 0 = bitwise, the frozen rule): a documented tolerance on the relative
                    # score difference, still requiring the same arg-max; anything else is the STOP of the frozen protocol
                    tol = float(os.environ.get("EVAL_FIXED_REPLAY_TOL", "0"))
                    within = row["repro_B2_maxrel"] is not None and row["repro_B2_maxrel"] <= tol and row["repro_B2_same_argmax"]
                    if not row["repro_B2_eq_E2"] and not within:
                        raise SystemExit(f"STOP: B2 replay differs from the E2 rows ({sid}; max|d| {row['repro_B2_maxdiff']}, "
                                         f"maxrel {row['repro_B2_maxrel']}, same argmax {row['repro_B2_same_argmax']})")
            del geo

        if sset == "valsel":
            for r in blocks:
                region = rp.region_of(recs[r])
                fus = cls(*region, device=dev)
                for q in range(r + 1):
                    fus.integrate_record(recs[q])
                h, eps = refresh(region, fus, r)
                last = max(t for t in sts if t - t % 10 == r)
                for t in range(r, last + 1):
                    if t > r:
                        fus.integrate_record(recs[t])
                    if t in sts:
                        state(sts[t], recs[t], region, fus, h, eps)
                del fus
        else:
            region = rp.frame
            fus = cls(*region, device=dev)
            h = eps = None
            for t in range(T + 1):
                fus.integrate_record(recs[t])
                if t in blocks:
                    h, eps = refresh(region, fus, t)
                if t in sts:
                    state(sts[t], recs[t], region, fus, h, eps)
            del fus
        torch.cuda.empty_cache()
    return [rows[s["sid"]] for s in sorted(sts.values(), key=lambda s: s["t"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["valsel", "dev3"], required=True)
    ap.add_argument("--stratum", choices=["B0", "B2"], default="B0")
    ap.add_argument("--t0", required=True)
    ap.add_argument("--t1", required=True)
    ap.add_argument("--tag", default="s_final")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    v2 = X.S2.load_cores(dev)["V2M"]
    cores = {"B2": V2Core(v2), "T0": V3Core(a.t0, dev), "T1": V3Core(a.t1, dev)}
    ref = {}
    for f in glob.glob(os.path.join(E2F.EP.W, "fixed", f"{a.set}_s5000", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref[r["sid"]] = r["scores"]["B2"]
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(W, f"{a.set}_{a.tag}", a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({"t0": a.t0, "t0_sha256": cores["T0"].ckpt_sha, "t1": a.t1, "t1_sha256": cores["T1"].ckpt_sha, "v2m_sha256": v2.ckpt_sha},
              open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    for tag, sts in E2F.states(a.set, a.stratum)[pi::pn]:
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
