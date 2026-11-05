"""Multi-View Coverage Reasoning Study -- state drivers (seen2scene env, 1 GPU per process; offline).
  --set e3  D1 Table 1 trajectories (rep 0): colosseum s0-s4, pisa s1; states t % 5 == 0 with an E-B oracle row
  --set g2  EXP6 V3 G2 Dev-3 fixed states (barts, pantheon, colosseum; strata B0, B2), replayed as eval_fixed.py does
Replay = the planner state: records 0..t through the installed fusion (D-policy capture, b2 fusion, D-v2 grid), R10 encoder refresh,
noise = sha_seed of the planner's own keys for k = 0..7 (k < 4 = the planner's K = 4 noise). Per state mv_core.analyze_state.
Output work_dirs/exp6_v3/multiview_coverage_study/states/<set>_<tag>.jsonl (one file per trajectory, written at the end; resumable).
Usage: python mv_run.py --set e3|g2 [--part i/n] [--tmax T] [--only TAG] [--save t1,t2,...]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mv_core as C  # noqa: E402

np, torch, FX = C.np, C.torch, C.FX
R = C.R
OUT = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "states")
CASES = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "cases")
E3_EPISODES = [("colosseum", k) for k in range(5)] + [("pisa", 1)]
A1 = os.path.join(R, "work_dirs", "cvpr_results", "geometry_roi_audit", "a1")
AUDIT = os.path.join(R, "experiments", "cvpr_results", "geometry_roi_audit", "results", "a1_gt_target.json")


def box_of(scene):
    sb = json.load(open(os.path.join(FX.DATA, scene, "settings.json")))["scene"]
    return np.asarray(sb["x_min"], float) * 10.0, np.asarray(sb["x_max"], float) * 10.0


def jobs(a):
    if a.set == "e3":
        return [(f"{s}_s{k}", (s, k)) for s, k in E3_EPISODES]
    sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts"))
    import e2_fixed as E2F
    out = []
    for stratum in ("B0", "B2"):
        E2F.X.setup(stratum)
        for tag, sts in E2F.states("dev3", stratum):
            out.append((f"{stratum}_{tag}", (stratum, tag, sts)))
    return out


def g2_oracle(stratum, st):
    import e2_fixed as E2F
    X = E2F.X
    root = X.ORIG_W if stratum == "B0" else X.P.ROOT_B2
    man = json.load(open(os.path.join(root, "manifests", "states_outdoor.json")))["states"]
    m = next(s for s in man if s["sid"] == st["sid"])
    o = np.load(os.path.join(root, "states", "outdoor", st["sid"], "oracle.npz"))
    Cc = np.asarray(m["cand"], int)
    return Cc, np.asarray(o["gain"], float)[Cc] * 100.0


def run_traj(a, tag, spec, core, dev, out_path):
    rows = []
    if a.set == "e3":
        s, k = spec
        rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
        ep = json.load(open(os.path.join(R, rec1["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(R, rec1["episode_file"])))
        svc = lambda t: os.path.join(root, "svc", ep["tag"], f"svc_{s}_s{k}_t{t}.npz")  # noqa: E731
        orc = json.load(open(os.path.join(FX.ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.json")))
        orow = {r["t"]: r for r in orc["steps"]}
        gt = np.load(os.path.join(FX.ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.npz"))["gt_points"].astype(np.float64)
        vis = float(orc["vis"])
        fr, dims, dom, _ = FX.TD.scene_geometry(s)
        sb = json.load(open(os.path.join(FX.DATA, s, "settings.json")))["scene"]
        z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
        seed = lambda tr, kk: C.RY.sha_seed(C.LC.seed_key(s, k, 0, tr, kk))  # noqa: E731
        T = len(ep["steps"]) - 1
        targets = [t for t in range(0, T, 5) if t in orow]
        oracle_of = lambda t, keys: [next((c["dcov_pp"] for c in orow[t]["cands"] if tuple(json.loads(c["key"])) == kk), np.nan) for kk in keys]  # noqa: E731
        scene, start, meta0 = s, k, {"set": "e3", "tag": tag, "scene": s, "start": k}
    else:
        stratum, tg, sts = spec
        import e2_fixed as E2F
        E2F.X.setup(stratum)
        rp = E2F.Rep("dev3", sts[0], dev)
        fr, dims, dom = rp.frame
        z0 = rp.z0
        s = sts[0]["scene"]
        svc = rp.svc
        seed = lambda tr, kk: C.RY.sha_seed(rp.key(tr, kk))  # noqa: E731
        byt = {x["t"]: x for x in sts}
        targets = sorted(byt)
        T = max(targets) + 1
        z = np.load(os.path.join(A1, f"{s}.npz"))
        gt = z["gt_points"].astype(np.float64)
        vis = float(json.load(open(AUDIT))[s]["roi_box"]["visibility_ratio"])

        def oracle_of(t, keys):
            Cc, g = g2_oracle(stratum, byt[t])
            o = np.full(len(keys), np.nan)
            o[Cc] = g
            return list(o)
        scene, start, meta0 = s, int(sts[0]["start"]), {"set": "g2", "tag": tag, "scene": s, "start": int(sts[0]["start"]), "stratum": stratum}
    if a.tmax is not None:
        targets = [t for t in targets if t <= a.tmax]
    if not targets:
        return []
    C.set_scene(scene)
    lo, hi = box_of(scene)
    gt_tree = C.cKDTree(gt)
    covered = np.zeros(len(gt), bool)
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    kM = float(fr.k_M)
    isect = FX.gt_mesh(scene)
    h, eps8, seeds = None, None, None
    save = {int(x) for x in a.save.split(",")} if a.save else set()
    for t in range(max(targets) + 1):
        rec = C.MC.load_service_record(svc(t))
        fu.integrate_record(rec)                       # D-policy capture + b2 fusion + D-v2 grid
        sp = np.asarray(rec["sample_points"], np.float64).reshape(-1, 3)
        sp = sp[np.all((sp >= lo) & (sp <= hi), 1)]
        if len(sp):
            dd, _ = C.cKDTree(sp).query(gt, k=1, distance_upper_bound=C.EPS, workers=8)
            covered |= np.isfinite(dd)
        if t % 10 == 0:
            geo = C.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            seeds = [int(seed(t, kk)) for kk in range(C.KMAX)]
            eps8 = C.PR.noise(dev, C.KMAX, lambda kk: seeds[kk])
            del geo
        if t not in targets or len(rec["legal_R"]) == 0:
            continue
        keys = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
        tb = time.time()
        st = C.analyze_state({"core": core, "fr": fr, "dims": dims, "dom": dom, "fu": fu, "h": h, "eps8": eps8, "z0": z0, "rec": rec, "kM": kM,
                              "isect": isect, "gt": gt, "covered0": covered.copy(), "vis": vis, "box": (lo, hi), "oracle": oracle_of(t, keys), "seeds": [str(x) for x in seeds],
                              "meta": {**meta0, "t": t}, "save": os.path.join(CASES, f"{tag}_t{t}.npz") if t in save else None})
        st["seconds_total"] = time.time() - tb
        st["peak_mem_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        rows.append(st)
        r = st["regression"]
        print(f"{tag} t={t} n={st['n_legal']} views={st['n_views']} reg K4 {r['recorded_vs_planner_K4']:.1e} K8 {r['recorded_vs_planner_K8']:.1e} "
              f"K8[:4]-K4 {r['K8_first4_vs_K4']:.1e} | {st['seconds']['K4']:.1f}s/{st['seconds']['K8']:.1f}s total {st['seconds_total']:.0f}s mem {st['peak_mem_gb']:.1f}GB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("e3", "g2"))
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--only", default=None)
    ap.add_argument("--save", default="")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(CASES, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    J = jobs(a)
    if a.only:
        J = [j for j in J if j[0] == a.only]
    pi, pn = map(int, a.part.split("/"))
    J = J[pi::pn]
    first_scene = J[0][1][0] if a.set == "e3" else J[0][1][2][0]["scene"]
    C.install(first_scene, dev)
    hcfg = C.yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    for tag, spec in J:
        po = os.path.join(a.out, f"{a.set}_{tag}.jsonl")
        if os.path.exists(po) and not a.only:
            continue
        tb = time.time()
        rows = run_traj(a, tag, spec, core, dev, po)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {a.set} {tag} states {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
