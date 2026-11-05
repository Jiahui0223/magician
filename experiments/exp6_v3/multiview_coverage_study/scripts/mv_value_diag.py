"""Multi-View Coverage Reasoning Study -- value-function diagnostic (seen2scene env, 1 GPU; offline; GT only to score).
Same states and replay as mv_run.py. For every first-layer candidate, one-step values from the FROZEN first-layer prediction (the
planner's K = 4 noise) and from the TRUE first hits of the same 114 x 64 rays, each with
  w      the planner's value function (h_policy.value_cov: novel 1.0 cells, weight min(1, (z / z0)^2); D-v2 filter on predictions)
  u      the same cells with weight 1 (no depth weighting)
  gt     the geometric coverage of the official GT target points (true depth only; = geo_one_step of mv_run)
  area   the same coverage rule on AREA-UNIFORM samples of the target surface (audit a1 mesh samples on target faces; true depth only)
  occ    the planner's cells / weights, but a hit is new only if no DENSE observed depth point (D100, the D-v2 observed 0.5 grid,
         3 x 3 x 3 neighbourhood) is near it -- online-computable (true depth and predictions)
ranked against the official one-step gain. Output results/value_diag.json and the per-state rows in work_dirs/.../value_diag/."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mv_core as C  # noqa: E402
import mv_run as RN  # noqa: E402

np, torch, FX = C.np, C.torch, C.FX
OUT = os.path.join(C.R, "work_dirs", "exp6_v3", "multiview_coverage_study", "value_diag")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("e3", "g2"))
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    a.tmax, a.save, a.only = None, "", None
    os.makedirs(OUT, exist_ok=True)
    dev = torch.device("cuda")
    frozen_score = C.PC.score_state                              # the frozen first-layer scorer (before d_policy replaces it)
    J = RN.jobs(a)
    pi, pn = map(int, a.part.split("/"))
    J = J[pi::pn]
    C.install(J[0][1][0] if a.set == "e3" else J[0][1][2][0]["scene"], dev)
    hcfg = C.yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(C.R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])

    A1 = os.path.join(C.R, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")
    area_cache = {}

    def value_occ(t, v, geo, z0, occd):
        """h_policy.value_cov with the S10-distance novelty replaced by the dilated dense observed grid."""
        hit = torch.isfinite(t)
        p = v["o"][None] + v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
        ijk = torch.floor(p / C.VS).long()
        for a_ in range(3):
            ijk[:, a_] = ijk[:, a_].clamp(0, geo.dims[a_] - 1)
        hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        pr = torch.as_tensor(geo.fr.model_to_runtime(p.double().cpu().numpy()), device=dev)
        g = C.INST["grid"]
        q = torch.floor((pr - g.lo_t) / C.DV2.CFG["cell"]).long()
        ok = (q >= 0).all(-1) & (q < g.n_t).all(-1)
        qc = torch.minimum(q.clamp(min=0), g.n_t - 1)
        seen = ok & occd[qc[:, 0], qc[:, 1], qc[:, 2]]
        nov = hit & ~seen
        if not nov.any():
            return 0.0
        c = np.floor(pr[nov].cpu().numpy() / 1.0).astype(np.int64) + (1 << 19)
        key = (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]
        z = (t[nov].double() * float(geo.fr.k_M) * v["dz"][nov].double()).cpu().numpy()
        w = np.minimum((z / float(z0)) ** 2, 1.0)
        ids, inv = np.unique(key, return_inverse=True)
        wv = np.zeros(len(ids))
        np.maximum.at(wv, inv, w)
        return float(wv.sum())

    def analyze(S):
        geo = C.PC.Geometry.from_fusion(S["fr"], S["dims"], S["dom"], S["fu"], dev)
        sc_ = S["meta"]["scene"]
        if sc_ not in area_cache:
            za = np.load(os.path.join(A1, f"{sc_}.npz"))
            area_cache.clear()
            area_cache[sc_] = za["mesh_sample"][za["mesh_sample_target_face"]].astype(np.float64)
        area = area_cache[sc_]
        da, _ = C.HP.obs_tree().query(area, k=1, distance_upper_bound=C.EPS)
        area_cov0 = np.isfinite(da)
        occd = torch.nn.functional.max_pool3d(C.INST["grid"].occ[None, None].half(), 3, 1, 1)[0, 0] > 0
        rec, z0, kM = S["rec"], S["z0"], S["kM"]
        out, _, views = frozen_score(core, geo, S["h"], rec["legal_R"], rec["legal_T"], S["eps8"][:4], z0, keep_hits=True)
        keys = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
        lo, hi = S["box"]
        res = {k_: [] for k_ in ("pred_w", "pred_u", "true_w", "true_u", "true_gt", "pred_w_d1", "true_area", "true_occ", "pred_occ")}
        for j, v in enumerate(views):
            vc = C.ViewC(keys[j], v, out["hits"][j], S["fr"], kM, dev)
            tt = torch.as_tensor(FX.gt_first_hit(S["isect"], vc.c.cpu().numpy(), vc.d.cpu().numpy(), float(vc.tcap.max())), device=dev)
            tt = torch.where(tt <= vc.tcap, tt, torch.full_like(tt, float("inf")))
            tm = (tt / kM).to(v["d"].dtype)
            res["true_w"].append(float(C.VC0(tm, v, geo, z0, C.VCFG)[1].sum()))
            res["true_u"].append(float(C.VC0(tm, v, geo, 1e-9, C.VCFG)[1].sum()))
            res["pred_w"].append(float(np.mean([C.INST["vc_dv2"](out["hits"][j]["t"][k], v, geo, z0, C.VCFG)[1].sum() for k in range(4)])))
            res["pred_u"].append(float(np.mean([C.INST["vc_dv2"](out["hits"][j]["t"][k], v, geo, 1e-9, C.VCFG)[1].sum() for k in range(4)])))
            res["pred_w_d1"].append(float(np.mean([C.VC0(out["hits"][j]["t"][k], v, geo, z0, C.VCFG)[1].sum() for k in range(4)])))
            fin = torch.isfinite(tt)
            p = (vc.c[None] + vc.d * torch.where(fin, tt, torch.zeros_like(tt))[:, None]).cpu().numpy()
            okp = fin.cpu().numpy() & ((tt * vc.dz).cpu().numpy() < 70.0) & np.all((p >= lo) & (p <= hi), 1)
            n = 0
            if okp.sum():
                dd, _ = C.cKDTree(p[okp]).query(S["gt"], k=1, distance_upper_bound=C.EPS, workers=8)
                n = int((np.isfinite(dd) & ~S["covered0"]).sum())
            res["true_gt"].append(n * 100.0 / (len(S["gt"]) * S["vis"]))
            na = 0
            if okp.sum():
                dd2, _ = C.cKDTree(p[okp]).query(area, k=1, distance_upper_bound=C.EPS, workers=8)
                na = int((np.isfinite(dd2) & ~area_cov0).sum())
            res["true_area"].append(na * 100.0 / len(area))
            res["true_occ"].append(value_occ(tm, v, geo, z0, occd))
            res["pred_occ"].append(float(np.mean([value_occ(torch.where(C.DV2.visible_mask(C.INST["grid"], geo.fr, v, out["hits"][j]["t"][k], kM), out["hits"][j]["t"][k],
                                                                          torch.full_like(out["hits"][j]["t"][k], float("inf"))), v, geo, z0, occd) for k in range(4)])))
        del geo, out, views, occd
        torch.cuda.empty_cache()
        return {**S["meta"], "keys1": [list(k) for k in keys], "oracle": [None if not np.isfinite(x) else float(x) for x in S["oracle"]], **res}

    orig = C.analyze_state
    C.analyze_state = analyze
    for tag, spec in J:
        po = os.path.join(OUT, f"{a.set}_{tag}.jsonl")
        if os.path.exists(po):
            continue
        rows = RN.run_traj(a, tag, spec, core, dev, po) if False else None
        rows = []
        # run_traj prints regression fields that this diagnostic does not produce; reuse its replay through a thin wrapper
        def collect(S):
            r = analyze(S)
            rows.append(r)
            return {"regression": {"recorded_vs_planner_K4": 0.0, "recorded_vs_planner_K8": 0.0, "K8_first4_vs_K4": 0.0}, "n_legal": len(r["keys1"]), "n_views": len(r["keys1"]),
                    "seconds": {"K4": 0.0, "K8": 0.0}}
        C.analyze_state = collect
        RN.run_traj(a, tag, spec, core, dev, po)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(po + ".tmp", po)
        print("DONE", tag, len(rows), flush=True)
    C.analyze_state = orig


if __name__ == "__main__":
    main()
