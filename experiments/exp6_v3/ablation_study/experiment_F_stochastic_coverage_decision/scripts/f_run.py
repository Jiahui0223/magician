"""Experiment F -- state drivers (seen2scene env, 1 GPU per process; offline). Per state f_core.analyze_state.
  --set explore   the 395 multiview-study states (E3 D1 trajectories colosseum s0-s4 + pisa s1, t % 5 == 0 with an E-B oracle row;
                  G2 Dev-3 B0 / B2 manifest states), replayed exactly as multiview_coverage_study/scripts/mv_run.py
  --set confirm   configs/confirmation_set.json: D-v2's own Table 1 closed-loop episodes (V3-DV2-KSV, rep 0), t = 0, 5, ..., 95
Replay = the planner state (records 0..t through the installed fusion, R10 encoder refresh, noise = sha_seed of the planner's own keys,
k = 0..7). Output work_dirs/.../experiment_F_stochastic_coverage_decision/states/<set>_<tag>.jsonl (one file per trajectory; resumable).
Usage: python f_run.py --set explore|confirm [--part i/n] [--only TAG] [--tmax T]"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import f_core as F  # noqa: E402

C = F.C
sys.path.insert(0, os.path.join(F.R, "experiments", "exp6_v3", "multiview_coverage_study", "scripts"))
import mv_run as MR  # noqa: E402

np, torch, FX = C.np, C.torch, C.FX
R = F.R
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_F_stochastic_coverage_decision")
OUT = os.path.join(W, "states")
MV_STATES = os.path.join(R, "work_dirs", "exp6_v3", "multiview_coverage_study", "states")


def jobs(a):
    if a.set == "explore":
        J = []
        for st, nm in (("e3", "e3"), ("g2", "g2")):
            a2 = argparse.Namespace(set=st)
            J += [(f"{nm}_{tag}", (st, spec)) for tag, spec in MR.jobs(a2)]
        return J
    conf = json.load(open(os.path.join(F.FD, "configs", "confirmation_set.json")))
    return [(f"conf_{e['scene']}_s{e['start']}", ("conf", (e["scene"], e["start"], e["states"]))) for e in conf["episodes"]]


def mv_scores(tag):
    """the multiview study's stored D-v2 K = 4 scores of this trajectory (bitwise reference)"""
    p = os.path.join(MV_STATES, f"{tag}.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["t"]: r["planner_scores_K4"] for r in map(json.loads, open(p))}


def spec_of(a, tag, kind, spec, dev):
    """everything run_traj needs: scene, start, svc(t), seed(tr, kk), targets, oracle_of(t, keys), gt, vis, frame, z0, meta, ref scores"""
    if kind == "e3":
        s, k = spec
        rec1 = json.load(open(os.path.join(FX.EP, "V3-D1-H3-NORISK", f"{s}_s{k}.json")))
        ep = json.load(open(os.path.join(R, rec1["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(R, rec1["episode_file"])))
        orc = json.load(open(os.path.join(FX.ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.json")))
        gt = np.load(os.path.join(FX.ORA, "V3-D1-H3-NORISK", f"{s}_s{k}_r0.npz"))["gt_points"].astype(np.float64)
        T = len(ep["steps"]) - 1
        orow = {r["t"]: r for r in orc["steps"]}
        targets = [t for t in range(0, T, 5) if t in orow]
        ref, ref_kind = mv_scores(f"e3_{s}_s{k}"), "multiview"
        meta = {"set": "explore", "src": "e3", "tag": tag, "scene": s, "start": k}
    elif kind == "conf":
        s, k, sts = spec
        rec1 = json.load(open(os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes", "V3-DV2-KSV", f"{s}_s{k}.json")))
        ep = json.load(open(os.path.join(R, rec1["episode_file"])))
        root = os.path.dirname(os.path.dirname(os.path.join(R, rec1["episode_file"])))
        orc = json.load(open(os.path.join(W, "oracle", f"{s}_s{k}_r0.json")))
        gt = np.load(os.path.join(W, "oracle", f"{s}_s{k}_r0.npz"))["gt_points"].astype(np.float64)
        orow = {r["t"]: r for r in orc["steps"]}
        targets = [t for t in sts if t in orow]
        L = [json.loads(x) for x in open(os.path.join(root, "planner_logs", f"pl_{ep['tag']}.steps.jsonl"))]
        ref, ref_kind = {r["t"]: r["scores"] for r in L}, "closed_loop_log"
        meta = {"set": "confirm", "src": "dv2_table1", "tag": tag, "scene": s, "start": k}
    if kind in ("e3", "conf"):
        svc = lambda t: os.path.join(root, "svc", ep["tag"], f"svc_{s}_s{k}_t{t}.npz")  # noqa: E731
        vis = float(orc["vis"])
        fr, dims, dom, _ = FX.TD.scene_geometry(s)
        sb = json.load(open(os.path.join(FX.DATA, s, "settings.json")))["scene"]
        z0 = sum((b - a_) * 10.0 for a_, b in zip(sb["x_min"], sb["x_max"])) / 3.0 / 2.0
        seed = lambda tr, kk: C.RY.sha_seed(C.LC.seed_key(s, k, 0, tr, kk))  # noqa: E731
        oracle_of = lambda t, keys: [next((c["dcov_pp"] for c in orow[t]["cands"] if tuple(json.loads(c["key"])) == kk), np.nan) for kk in keys]  # noqa: E731
        return dict(scene=s, svc=svc, seed=seed, targets=targets, oracle_of=oracle_of, gt=gt, vis=vis, frame=(fr, dims, dom), z0=z0, meta=meta, ref=ref, ref_kind=ref_kind)
    stratum, tg, sts = spec
    import e2_fixed as E2F
    E2F.X.setup(stratum)
    rp = E2F.Rep("dev3", sts[0], dev)
    s = sts[0]["scene"]
    byt = {x["t"]: x for x in sts}
    z = np.load(os.path.join(MR.A1, f"{s}.npz"))

    def oracle_of(t, keys):
        Cc, g = MR.g2_oracle(stratum, byt[t])
        o = np.full(len(keys), np.nan)
        o[Cc] = g
        return list(o)
    return dict(scene=s, svc=rp.svc, seed=lambda tr, kk: C.RY.sha_seed(rp.key(tr, kk)), targets=sorted(byt), oracle_of=oracle_of,
                gt=z["gt_points"].astype(np.float64), vis=float(json.load(open(MR.AUDIT))[s]["roi_box"]["visibility_ratio"]), frame=rp.frame, z0=rp.z0,
                meta={"set": "explore", "src": "g2", "tag": tag, "scene": s, "start": int(sts[0]["start"]), "stratum": stratum},
                ref=mv_scores(f"g2_{stratum}_{tg}"), ref_kind="multiview")


def run_traj(a, tag, kind, spec, core, dev):
    P = spec_of(a, tag, kind, spec, dev)
    targets = P["targets"] if a.tmax is None else [t for t in P["targets"] if t <= a.tmax]
    if not targets:
        return []
    scene = P["scene"]
    fr, dims, dom = P["frame"]
    C.set_scene(scene)
    lo, hi = MR.box_of(scene)
    gt = P["gt"]
    covered = np.zeros(len(gt), bool)
    fu = FX.m_p1.P1Fusion(fr, dims, dom, device=dev)
    kM = float(fr.k_M)
    isect = FX.gt_mesh(scene)
    h = eps8 = seeds = None
    rows = []
    for t in range(max(targets) + 1):
        rec = C.MC.load_service_record(P["svc"](t))
        fu.integrate_record(rec)
        sp = np.asarray(rec["sample_points"], np.float64).reshape(-1, 3)
        sp = sp[np.all((sp >= lo) & (sp <= hi), 1)]
        if len(sp):
            dd, _ = C.cKDTree(sp).query(gt, k=1, distance_upper_bound=C.EPS, workers=8)
            covered |= np.isfinite(dd)
        if t % 10 == 0:
            geo = C.PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            h = core.pred.encode(geo.coarse())
            seeds = [int(P["seed"](t, kk)) for kk in range(C.KMAX)]
            eps8 = C.PR.noise(dev, C.KMAX, lambda kk: seeds[kk])
            del geo
        if t not in targets or len(rec["legal_R"]) == 0:
            continue
        keys = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
        tb = time.time()
        torch.cuda.reset_peak_memory_stats(dev)
        st = F.analyze_state({"core": core, "fr": fr, "dims": dims, "dom": dom, "fu": fu, "h": h, "eps8": eps8, "z0": P["z0"], "rec": rec, "kM": kM,
                              "isect": isect, "gt": gt, "covered0": covered.copy(), "vis": P["vis"], "box": (lo, hi), "oracle": P["oracle_of"](t, keys),
                              "seeds": [str(x) for x in seeds], "meta": {**P["meta"], "t": t}, "planner_log": P["ref"].get(t)})
        st["regression"]["reference"] = P["ref_kind"]
        if P["ref_kind"] == "multiview":
            r_ = P["ref"].get(t)
            st["regression"]["replay_vs_multiview_bitwise"] = None if r_ is None else (r_ == st["planner_K4"])
            st["regression"].pop("replay_vs_closed_loop_log", None)
            st["regression"].pop("replay_vs_closed_loop_log_bitwise", None)
        st["seconds_total"] = time.time() - tb
        rows.append(st)
        r = st["regression"]
        print(f"{tag} t={t} n={st['n_legal']} paths={len(st['paths'])} rec4 {r['recorded_vs_planner_K4']:.1e} g1 {r['gamma1_bitwise']} "
              f"ref {r.get('replay_vs_multiview_bitwise', r.get('replay_vs_closed_loop_log_bitwise'))} p1 {r['p1_unchanged_by_scoring']} "
              f"avg0 {r['avg0_onestep_vs_hard_max_abs']:.1e} | {st['seconds_total']:.0f}s mem {st['peak_mem_gb']:.1f}GB", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=("explore", "confirm"))
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--only", default=None)
    ap.add_argument("--tmax", type=int, default=None)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    J = jobs(a)
    if a.only:
        J = [j for j in J if j[0] == a.only]
    pi, pn = map(int, a.part.split("/"))
    J = J[pi::pn]
    if not J:
        return
    kind0, spec0 = J[0][1]
    first_scene = spec0[0] if kind0 in ("e3", "conf") else spec0[2][0]["scene"]
    C.install(first_scene, dev)
    hcfg = C.yaml.safe_load(open(os.environ["H_CONFIG"]))
    core = FX.PV3.V3Core(os.path.join(R, hcfg["model"]["checkpoint"]), dev, expect_sha=hcfg["model"]["sha256"])
    for tag, (kind, spec) in J:
        po = os.path.join(a.out, f"{tag}.jsonl")
        if os.path.exists(po) and not a.only:
            continue
        claim = os.path.join(W, "claims", f"state_{tag}.claim")
        if not a.only:
            try:
                fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.uname()[1]} {os.environ.get('CUDA_VISIBLE_DEVICES')} {time.strftime('%FT%T')}\n".encode())
                os.close(fd)
            except FileExistsError:
                continue
        tb = time.time()
        rows = run_traj(a, tag, kind, spec, core, dev)
        with open(po + ".tmp", "w") as f:
            for r in rows:
                f.write(json.dumps(r, default=float) + "\n")
        os.replace(po + ".tmp", po)
        print(f"DONE {tag} states {len(rows)} {time.time() - tb:.0f}s", flush=True)


if __name__ == "__main__":
    main()
