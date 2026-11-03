"""Camera Pose C2 -- Task 0 off-grid smoke (two processes, no closed loop):
  --make  (magician env, 1 GPU): VALSEL state 8194nk5LbLH t=5; execute its C2c oracle-best (off-grid) pose: render with the
          frozen sensor, 10 % sample with the trajectory seed, build a service record in the planner format whose legal
          list is the C2c candidate set of the NEW pose (fine lattice, segment-tested), save it.
  --check (seen2scene env, 1 GPU): replay records 0..t of the trajectory + the off-grid record into the planner's D100 / S10
          P1 fusions (same region rule), build the B2 planner geometry, the views, FrontierPrior and the B2 DistP1
          (registry checkpoint, K = 4) of the new legal list; report finite / non-empty values and the change of the P1.
Output work_dirs/exp_camera_pose/C2/smoke/{svc_offgrid.npz, check.json}."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
OUT = os.path.join(REPO, "work_dirs", "exp_camera_pose", "C2", "smoke")
SID_TAG, T = "8194nk5LbLH_L0_ogr_p10", 5


def make():
    import torch
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
    sys.path.insert(0, os.path.join(EXP5, "task3_full_depth_p1", "e2_pilot", "scripts"))
    import c2_poses as CP
    import e2_paths as EP
    import t2_common as C
    import t21_traj as TJ
    import t21_levels as LV
    S, T12, GE = TJ.S, TJ.T12, TJ.GE
    dev = torch.device("cuda")
    man = json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]
    st = next(s for s in man if s["tag"] == SID_TAG and s["t"] == T)
    orc = json.load(open(os.path.join(REPO, "work_dirs", "exp_camera_pose", "C2", "oracle", f"{st['sid']}.json")))
    best = max([c for c in orc["candidates"] if c["family"] == "C2c"], key=lambda c: c["gain"])
    card = C.load_card(st["scene"])
    v, f = LV.mesh_zup(st["scan"])
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    blk = T12.Blocker(v, f)
    g = GE.grid_def(card)
    fn = tuple(best["f"])
    e, az = CP.angles(g, fn)
    R, Tt = sen.RT(CP.position(g, fn), e, az)
    fr = sen.render(R, Tt)
    pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(st["scene"], int(st["start"]), T + 1, "sample"))
    legal, n_gen, n_oob, n_blk = CP.candidates(g, fn, "C2c", blk.blocked, S.rt_to_zup_m)
    LR, LT, LC = [], [], []
    for c in legal:
        ee, aa = CP.angles(g, c["f"])
        r_, t_ = sen.RT(CP.position(g, c["f"]), ee, aa)
        LR.append(r_[0].cpu().numpy()); LT.append(t_[0].cpu().numpy()); LC.append(CP.position(g, c["f"]))
    td = os.path.join(EP.T2W, "mp3d", "traj", st["scene"], st["traj"])
    prev = np.load(os.path.join(td, "svc", f"svc_t{T}.npz"))
    hist = [CP.to_fine(r) for r in prev["executed_pose_idx_history"]] + [fn]
    rec = {"scene_key": np.asarray(f"{st['scene']}/{st['start']}"), "start": np.asarray(int(st["start"])), "t": np.asarray(T + 1),
           "zbuf": fr["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32), "mask": fr["mask"][0, ..., 0].cpu().numpy().astype(bool),
           "R": fr["R"][0].cpu().numpy().astype(np.float64), "T": fr["T"][0].cpu().numpy().astype(np.float64),
           "cam_center": fr["fov"].get_camera_center()[0].cpu().numpy().astype(np.float64), "pose_idx": np.asarray(fn, np.int64),
           "sample_pixel_flat": flat.numpy().astype(np.int64), "sample_points": pts.cpu().numpy().astype(np.float32),
           "executed_pose_idx_history": np.asarray(hist, np.int64), "legal_idx": np.asarray([c["f"] for c in legal], np.int64).reshape(-1, 5),
           "legal_R": np.asarray(LR, np.float64).reshape(-1, 3, 3), "legal_T": np.asarray(LT, np.float64).reshape(-1, 3), "legal_center": np.asarray(LC, np.float64).reshape(-1, 3)}
    rec = S.MC.validate_service_record(rec)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, "svc_offgrid.npz"), **rec)
    info = {"sid": st["sid"], "executed_fine_pose": list(fn), "family": best["family"], "trans_m": best["trans_m"], "dyaw": best["dyaw"], "dpitch": best["dpitch"],
            "on_original_lattice": best["on_original_lattice"], "oracle_gain_pp": 100 * best["gain"], "n_points_sampled": int(len(pts)),
            "new_legal": len(legal), "generated": n_gen, "out_of_bounds": n_oob, "blocked": n_blk, "record_validated": True}
    json.dump(info, open(os.path.join(OUT, "make.json"), "w"), indent=1)
    print(json.dumps(info, indent=1))


def check():
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v5_2", "scripts"))
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
    sys.path.insert(0, os.path.join(REPO, "experiments", "exp_camera_pose", "scripts"))
    import v52_common as VC
    import eval_fixed as EF
    import frontier as FR
    X, E2F = EF.X, EF.E2F
    X.setup("B0")
    torch, MC, PC, PR, RY = X.torch, X.MC, X.PC, X.PR, X.RY
    dev = torch.device("cuda")
    sts = dict(E2F.states("valsel", "B0"))[SID_TAG]
    st = next(s for s in sts if s["t"] == T)
    rp = E2F.Rep("valsel", sts[0], dev)
    recs = [MC.load_service_record(rp.svc(t)) for t in range(T + 1)]
    off = MC.load_service_record(os.path.join(OUT, "svc_offgrid.npz"))
    region = rp.region_of(recs[0 if T < 10 else T - T % 10])
    F100, F10 = X.D.fusion_class(100), X.D.fusion_class(10)
    f100, f10 = F100(*region, device=dev), F10(*region, device=dev)
    for r in recs:
        f100.integrate_record(r); f10.integrate_record(r)
    g_before = VC.b2_geometry(PC, MC, region, f100, f10, dev)
    s_before = {k: int((g_before.struct == v).sum()) for k, v in (("EMPTY", 0), ("UNKNOWN", 1), ("BAND", 2))}
    f100.integrate_record(off); f10.integrate_record(off)
    geo = VC.b2_geometry(PC, MC, region, f100, f10, dev)
    s_after = {k: int((geo.struct == v).sum()) for k, v in (("EMPTY", 0), ("UNKNOWN", 1), ("BAND", 2))}
    views = [geo.view(R, Tt) for R, Tt in zip(off["legal_R"], off["legal_T"])]
    feats = FR.features(views, geo)
    core = PC.Core(os.path.join(REPO, "work_dirs", "checkpoints", "exp6_v2", "B2", "v2mix_ema17500.pt"), dev,
                   expect_sha="343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516")
    h = core.pred.encode(geo.coarse())
    eps = PR.noise(dev, X.A.K_MAIN, lambda kk: RY.sha_seed(f"exp-camera-pose|C2|smoke|k{kk}"))
    with torch.no_grad():
        out, tm, _ = PC.score_state(core, geo, h, off["legal_R"], off["legal_T"], eps, rp.z0)
    D = out["distp1"]
    res = {"struct_before": s_before, "struct_after": s_after, "p1_changed": s_before != s_after, "n_candidates": len(D),
           "distp1_finite": bool(np.all(np.isfinite(D))), "distp1_nonzero": int(sum(x > 0 for x in D)), "distp1": [float(x) for x in D],
           "F": [f["F"] for f in feats], "score_time_s": {k: float(v) for k, v in tm.items()}}
    res["ok"] = res["p1_changed"] and res["distp1_finite"] and res["n_candidates"] > 0
    json.dump(res, open(os.path.join(OUT, "check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("distp1", "F")}, indent=1))


if __name__ == "__main__":
    make() if "--make" in sys.argv else check()
