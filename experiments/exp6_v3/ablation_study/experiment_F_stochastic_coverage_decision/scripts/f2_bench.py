"""Experiment F / F2 cost micro-benchmark (seen2scene env, 1 GPU; offline). Wall time and peak GPU memory of the selected soft variant
(results/selection.json: sigma 0.25 tau_cov, depth weight, max-union) on the planner's K = 4 samples, measured on top of the recorded planner
run, against the planner's own scoring time. Output results/F2_soft_coverage/F2_bench.json. Usage: python f2_bench.py [TAG] [t,t,...]"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f_core as F  # noqa: E402
import f_run as FR  # noqa: E402

C = F.C
np, torch, FX = C.np, C.torch, C.FX


def soft_once(run, keys1, ch, geo, z0, tree, fr, kM, dev, sigma, area):
    t0 = C.sync()
    g = C.INST["grid"]
    s10 = torch.zeros(tuple(g.n), dtype=torch.bool, device=dev)
    obs = np.concatenate(C.HP.STATE["obs"]) if C.HP.STATE["obs"] else np.zeros((0, 3))
    if len(obs):
        q = torch.floor((torch.as_tensor(obs, device=dev) - g.lo_t) / F.DV2.CFG["cell"]).long()
        q = q[(q >= 0).all(1) & (q < g.n_t).all(1)]
        s10[q[:, 0], q[:, 1], q[:, 2]] = True
    ctx = {"s10dil": torch.nn.functional.max_pool3d(s10[None, None].half(), 5, 1, 2)[0, 0] > 0}
    del s10
    allk = set(keys1) | {b for a in keys1 for b in ch[a]}
    Q = {}
    for key in allk:
        c = C.ViewC(key, run["V"][key], run["HT"][key], fr, kM, dev)
        src = [F.sources(c, k, geo, z0, tree) for k in range(4)]
        maps = [F.support(*s, sigma, area) for s in src]
        uk = torch.unique(torch.cat([m[0] for m in maps]))
        own = torch.unique(torch.cat([s[0] for s in src]))
        ok = uk[F.gate(uk, torch.isin(uk, own), c, geo, ctx)]
        maps = [(u[torch.isin(u, s[0]) | torch.isin(u, ok)], v[torch.isin(u, s[0]) | torch.isin(u, ok)]) for (u, v), s in zip(maps, src)]
        Q[key] = F.agg(maps, 4)
    sc = []
    for a in keys1:
        best = max([F.union(Q[a], Q[b], "max") for b in ch[a]], default=float(Q[a][1].sum()))
        sc.append(best)
    return C.sync() - t0, sc


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "conf_eiffel_s0"
    ts = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,10,20,30,40").split(",")]
    sel = json.load(open(os.path.join(F.FD, "results", "selection.json")))
    v = sel["F2_variant_star"]
    sigma, area = float(v.split("_")[0][1:]), v.endswith("area")
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    J = dict(FR.jobs(type("A", (), {"set": "confirm" if tag.startswith("conf_") else "explore"})()))
    kind, spec = J[tag]
    C.install(spec[0] if kind in ("e3", "conf") else spec[2][0]["scene"], dev)
    hcfg = C.yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(F.R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    P = FR.spec_of(None, tag, kind, spec, dev)
    fr, dims, dom = P["frame"]
    C.set_scene(P["scene"])
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    kM = float(fr.k_M)
    rows = []
    h = eps = None
    for t in range(max(ts) + 1):
        rec = C.MC.load_service_record(P["svc"](t))
        fu.integrate_record(rec)
        if t % 10 == 0:
            geo = C.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            seeds = [int(P["seed"](t, kk)) for kk in range(4)]
            eps = C.PR.noise(dev, 4, lambda kk: seeds[kk])
        if t not in ts:
            continue
        geo = C.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
        tree = C.HP.obs_tree()
        torch.cuda.reset_peak_memory_stats(dev)
        run = C.run_planner(core, geo, h, rec, eps, P["z0"])
        mem_planner = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        keys1 = run["keys1"]
        ch = {a: C.children((a,), geo) for a in keys1}
        torch.cuda.reset_peak_memory_stats(dev)
        dt, sc = soft_once(run, keys1, ch, geo, P["z0"], tree, fr, kM, dev, sigma, area)
        rows.append({"t": t, "n_legal": len(keys1), "n_views": len(set(keys1) | {b for a in keys1 for b in ch[a]}), "planner_K4_s": run["seconds"], "soft_extra_s": dt,
                     "planner_peak_gb": mem_planner, "soft_peak_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30})
        print(rows[-1], flush=True)
    out = {"tag": tag, "variant": v, "rows": rows, "planner_K4_s_mean": float(np.mean([r["planner_K4_s"] for r in rows])),
           "soft_extra_s_mean": float(np.mean([r["soft_extra_s"] for r in rows])), "soft_peak_gb_max": float(max(r["soft_peak_gb"] for r in rows)),
           "note": "soft_extra_s = rebuilding the soft maps from the recorded K = 4 hits of every first-layer view and child + max-union scoring; unoptimised Python/torch"}
    json.dump(out, open(os.path.join(F.FD, "results", "F2_soft_coverage", "F2_bench.json"), "w"), indent=1)
    print({k: v for k, v in out.items() if k != "rows"})


if __name__ == "__main__":
    main()
