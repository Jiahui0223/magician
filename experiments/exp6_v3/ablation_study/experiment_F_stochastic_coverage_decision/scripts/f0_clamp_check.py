"""Experiment F / F0.1 supplementary check (seen2scene env, 1 GPU; offline). For a few exploration states: where do the hits of the
planner value function on RAW true depth lie? value_cov clamps voxel indices into the grid before its domain test, so a true hit outside
the P1 grid is counted on a boundary voxel. Per first-layer view: value of all true hits vs of the hits inside the grid, by label category.
Output results/F0_regression/F0_clamp_check.json. Usage: python f0_clamp_check.py [TAG:t ...]"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f_core as F  # noqa: E402
import f_run as FR  # noqa: E402

C = F.C
np, torch, FX, G = C.np, C.torch, C.FX, F.G
DEFAULT = ["e3_colosseum_s0:15", "e3_pisa_s1:40", "g2_B2_barts_s0:15", "g2_B0_pantheon_s1:0"]


def main():
    want = sys.argv[1:] or DEFAULT
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    J = dict(FR.jobs(type("A", (), {"set": "explore"})()))
    first = want[0].split(":")[0]
    kind0, spec0 = J[first]
    C.install(spec0[0] if kind0 == "e3" else spec0[2][0]["scene"], dev)
    hcfg = C.yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(F.R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    out = []
    for w in want:
        tag, t_ = w.split(":")
        kind, spec = J[tag]
        P = FR.spec_of(None, tag, kind, spec, dev)
        fr, dims, dom = P["frame"]
        C.set_scene(P["scene"])
        fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
        kM = float(fr.k_M)
        isect = FX.gt_mesh(P["scene"])
        for t in range(int(t_) + 1):
            rec = C.MC.load_service_record(P["svc"](t))
            fu.integrate_record(rec)
        geo = C.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        HP = C.HP
        tree = HP.obs_tree()
        for j, (Rm, Tm) in enumerate(zip(rec["legal_R"], rec["legal_T"])):
            v = geo.view(Rm, Tm)
            o_rt = np.asarray(fr.model_to_runtime(v["o"].double().cpu().numpy()[None]))[0]
            d_rt = C.MC.to_y_up(v["d"].double().cpu().numpy())
            d_rt = d_rt / np.linalg.norm(d_rt, axis=1, keepdims=True)
            tcap = (v["tcap"].double() * kM).cpu().numpy()
            tt = FX.gt_first_hit(isect, o_rt, d_rt, float(tcap.max()))
            tt = np.where(tt <= tcap, tt, np.inf)
            tm = torch.as_tensor(tt / kM, device=dev).to(v["d"].dtype)
            p = v["o"][None] + v["d"] * torch.where(torch.isfinite(tm), tm, torch.zeros_like(tm))[:, None]
            ijk = torch.floor(p / F.VS).long()
            dims_t = torch.as_tensor(geo.dims, device=dev)
            ingrid = ((ijk >= 0) & (ijk < dims_t)).all(1)
            lab = G.encode_labels(v["o"], v["d"], v["pr"], tm, v["dz"].to(v["d"].dtype), kM, geo.struct, geo.domain)
            row = {"tag": tag, "t": int(t_), "cand": j}
            inf = torch.full_like(tm, float("inf"))
            row["cells_all"] = int(len(C.VC0(tm, v, geo, P["z0"], C.VCFG)[0]))
            row["cells_in_grid_hits"] = int(len(C.VC0(torch.where(ingrid, tm, inf), v, geo, P["z0"], C.VCFG)[0]))
            row["value_all"] = float(C.VC0(tm, v, geo, P["z0"], C.VCFG)[1].sum())
            row["value_in_grid_hits"] = float(C.VC0(torch.where(ingrid, tm, inf), v, geo, P["z0"], C.VCFG)[1].sum())
            for nm, cid in G.CAT.items():
                m = lab["cat"] == cid
                if m.any():
                    row[f"value_{nm}"] = float(C.VC0(torch.where(m, tm, inf), v, geo, P["z0"], C.VCFG)[1].sum())
                    row[f"value_{nm}_in_grid"] = float(C.VC0(torch.where(m & ingrid, tm, inf), v, geo, P["z0"], C.VCFG)[1].sum())
            out.append(row)
        print(tag, t_, "views", len(rec["legal_R"]), "value in-grid share", np.mean([r["value_in_grid_hits"] / max(r["value_all"], 1e-9) for r in out if r["tag"] == tag]), flush=True)
        del geo, fu
        torch.cuda.empty_cache()
    agg = {"views": len(out), "value_in_grid_share_mean": float(np.mean([r["value_in_grid_hits"] / max(r["value_all"], 1e-9) for r in out]))}
    for nm in G.CAT:
        a = [r.get(f"value_{nm}_in_grid", 0.0) / r[f"value_{nm}"] for r in out if r.get(f"value_{nm}", 0) > 0]
        if a:
            agg[f"in_grid_share_{nm}"] = float(np.mean(a))
    p = os.path.join(F.FD, "results", "F0_regression", "F0_clamp_check.json")
    json.dump({"note": "value_cov clamps indices into the grid before the domain test; out-of-grid true hits are counted on boundary voxels",
               "summary": agg, "views": out}, open(p, "w"), indent=1)
    print(agg)


if __name__ == "__main__":
    main()
