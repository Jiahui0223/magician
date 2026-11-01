"""Experiment A -- fixed-state A0 / A1 scoring of the frozen V4 (seen2scene env, 1 GPU per process).

Same states, candidates and K = 4 noise as the V4 G1/G2 rows (exp6_v3/shared/eval_fixed.py): per episode a D100 replay
(P1FusionFull of the executed records; V4 encodes the D100 coarse grid; one prediction per candidate, kept) and a sparse
replay (P1Fusion S10 of the same records). The SAME predicted depths are scored twice: A1 = DistP1 with the dense map's
band_dil (D100 pass), A0 = DistP1 with the sparse map's band_dil (S10 pass). Also per state: fraction of predicted hit
rays judged known (inside band_dil) under each map, band_dil voxel counts, and the G0 check that A0 equals the stored
V4 rows bitwise. Output work_dirs/exp6_v4/ablation_study/experiment_A/fixed/<set>/<stratum>/<tag>.jsonl.
Usage: python a_fixed.py --set valsel|dev3 --stratum B0|B2 [--part i/n]"""
import argparse
import glob
import json
import os
import sys
import time

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import eval_fixed as EF  # noqa: E402

X, E2F = EF.X, EF.E2F
W = os.path.join(REPO, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "fixed")
REF = os.path.join(REPO, "work_dirs", "exp6_v4", "eval_fixed")
CK = os.path.join(REPO, "work_dirs", "exp6_v4", "checkpoints", "final.pt")


def known_stats(views, hits, geo):
    """Sum over candidates and K of (predicted hit rays in domain, of which NEW under geo.caster.band_dil)."""
    torch, SCO = X.torch, X.SCO
    n_hit = n_new = 0
    for v, hh in zip(views, hits):
        hh = hh.to(v["o"].device)
        for k in range(hh.shape[0]):
            new, _, hit = SCO.new_mask_from_depth(hh[k], v["o"], v["d"], geo.dims, geo.domain, geo.caster.band_dil)
            n_hit += int(hit.sum()); n_new += int(new.sum())
    return n_hit, n_new


def run_episode(sset, stratum, tag, sts, core, dev, ref):
    np, torch, MC, PC, PR, RY = X.np, X.torch, X.MC, X.PC, X.PR, X.RY
    rp = E2F.Rep(sset, sts[0], dev)
    sts = {s["t"]: s for s in sts}
    T = max(sts)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    rows = {s["sid"]: {"sid": s["sid"], "scene": s["scene"], "start": s["start"], "t": s["t"], "tag": tag, "stratum": stratum,
                       "phase": X.phase_of("indoor" if sset == "valsel" else "outdoor", s["t"]), "scores": {}, "known": {}, "band_vox": {}, "timing": {}}
            for s in sts.values()}
    hits = {}
    for x in (100, 10):
        cls = X.D.fusion_class(x)
        blocks = sorted({t - t % 10 for t in sts})

        def refresh(region, fus, r):
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            h = core.encode_for(fus, geo) if x == 100 else None
            eps = PR.noise(dev, X.A.K_MAIN, lambda kk, tr=r: RY.sha_seed(rp.key(tr, kk)))
            del geo
            return h, eps

        def state(st, rec, region, fus, h, eps):
            sid, row = st["sid"], rows[st["sid"]]
            geo = PC.Geometry.from_fusion(*region, fus, dev)
            assert fus.n_frames == st["t"] + 1, ("fusion must hold exactly the executed records 0..t", fus.n_frames, st["t"])
            k_M = float(geo.fr.k_M)
            if x == 100:
                t0 = time.time()
                out, tm, views = PC.score_state(core, geo, h, rec["legal_R"], rec["legal_T"], eps, rp.z0, keep_hits=True)
                torch.cuda.synchronize()
                hits[sid] = [hh["t"].detach().cpu() for hh in out["hits"]]
                row["timing"]["predict_s"] = time.time() - t0
                t1 = time.time()
                row["scores"]["A1"] = X.distp1_k(views, hits[sid], geo, k_M, rp.z0)            # dense band_dil
                row["timing"]["score_dense_s"] = time.time() - t1
                assert row["scores"]["A1"] == out["distp1_k"], "dense scoring must equal the planner's in-process scores (mode full)"
                nh, nn = known_stats(views, hits[sid], geo)
                row["known"]["A1"] = {"n_hit": nh, "n_new": nn}
                row["band_vox"]["A1"] = int(geo.caster.band_dil.sum())
                row["n_cand"] = len(views)
                row["n_unknown_rays"] = out["n_unknown_rays"]
            else:
                views = [geo.view(R, T) for R, T in zip(rec["legal_R"], rec["legal_T"])]
                t1 = time.time()
                row["scores"]["A0"] = X.distp1_k(views, hits[sid], geo, k_M, rp.z0)            # sparse band_dil, same hits
                row["timing"]["score_sparse_s"] = time.time() - t1
                nh, nn = known_stats(views, hits[sid], geo)
                row["known"]["A0"] = {"n_hit": nh, "n_new": nn}
                row["band_vox"]["A0"] = int(geo.caster.band_dil.sum())
                if sid in ref:
                    row["repro_A0_eq_V4rows"] = row["scores"]["A0"] == ref[sid]
                    if not row["repro_A0_eq_V4rows"]:
                        va, vb = np.asarray(row["scores"]["A0"], float).ravel(), np.asarray(ref[sid], float).ravel()
                        row["repro_A0_maxrel"] = float(np.max(np.abs(va - vb) / np.maximum(np.abs(vb), 1e-12))) if va.shape == vb.shape else None
                        raise SystemExit(f"STOP: A0 (sparse scoring of the fresh V4 predictions) differs from the stored V4 rows at {sid} "
                                         f"(maxrel {row['repro_A0_maxrel']})")
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
    ap.add_argument("--tag", default="a_fixed")
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    X.setup(a.stratum)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task2", "scripts"))
    torch = X.torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    v4core = X.PC.Core(CK, dev)
    assert v4core.model.ray_xattn is not None
    core = EF.V2Core(v4core)
    ref = {}
    for f in glob.glob(os.path.join(REF, f"{a.set}_s_final_v4", a.stratum, "*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            ref[r["sid"]] = r["scores"]["V4"]
    pi, pn = map(int, a.part.split("/"))
    od = os.path.join(W, a.set, a.stratum)
    os.makedirs(od, exist_ok=True)
    json.dump({"ckpt": CK, "sha256": core.ckpt_sha, "ref_rows": len(ref)}, open(os.path.join(od, f"models_{pi}.json"), "w"), indent=1)
    for tag, sts in E2F.states(a.set, a.stratum)[pi::pn]:
        po = os.path.join(od, f"{tag}.jsonl")
        if os.path.exists(po):
            continue
        tb = time.time()
        rows = run_episode(a.set, a.stratum, tag, sts, core, dev, ref)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(a.set, a.stratum, tag, len(rows), "%.0fs" % (time.time() - tb), "repro", sum(bool(r.get("repro_A0_eq_V4rows")) for r in rows), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
