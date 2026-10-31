"""EXP6 V3 Ablation Experiment C -- MP3D closed loop with GT side channel (magician env; simulator / GT side).

The Task 2 indoor closed loop (task2/scripts/t23_in_sim.py: T0.12 grid, legality, start snapping, frozen sensor,
sampling seeds, C_obs evaluator, service records, C0 + 150 moves) with the loop body copied unchanged and two
additions per step, both AFTER the service record is written and BEFORE the pick is executed:
  1. GT query for the planner (C1: t* of every legal candidate; C2: true C_obs gain of every legal candidate) ->
     <episode>/cq/cq_t<t>.npz, path sent to the child together with the service record; simulator state hash before /
     after the query must be identical (no leakage of the query into the environment);
  2. diagnostics computed AFTER the child answered (never sent to it): true gains of all legal candidates (C1; C2
     reuses its query), oracle best / regret of the executed pick, remaining work and its share inside the Task 2
     planning region, true gains of the GT-blocked neighbours (candidate-space diagnostic, R3).
Checks at every step: pick legal and inside the prefer-unvisited set; the executed candidate's precomputed sample
equals the next frame's sample bitwise and its gain equals the coverage increment (|d| <= 1e-12); C_obs monotone.
Groups: C1 | C2 (child c_in_planner.py) | B2replay (child = Task 3 B2 planner t32_in_planner.py, T3_P1_MODE=b2;
only for the G0 baseline-replay test; no GT channel).
Usage: python c_in_sim.py --group C1 --scan S --level L --start K [--moves N] [--out DIR] [--no-diag]
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c_common as CC  # noqa: E402

sys.path.insert(0, os.path.join(CC.EXP5, "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

import numpy as np  # noqa: E402

S, C, T12, GE, S2E = T23.S, T23.C, T23.T12, T23.GE, T23.S2E


def child(group, key, tag, od, cfg):
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if group == "B2replay":
        g = cfg["groups"]["V2M"]
        env["T3_P1_MODE"] = "b2"
        cmd = [f"{S2E}/bin/python", os.path.join(CC.T3S, "t32_in_planner.py"), "--group", "V2M", "--card", key, "--tag", tag,
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    else:
        cmd = [f"{S2E}/bin/python", os.path.join(HERE, "c_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
               "--svc-dir", os.path.join(od, "svc"), "--cq-dir", os.path.join(od, "cq")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello
    hello.setdefault("info", {})["sim_group"] = group
    return p, hello


def main():
    import torch
    from c_gt import IndoorGT
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--group", choices=list(CC.GROUPS) + ["B2replay"], required=True)
    ap.add_argument("--moves", type=int, default=CC.MOVES_IN)
    ap.add_argument("--config", default="eval_indoor_v1.json")
    ap.add_argument("--out", default=os.path.join(CC.WEVAL, "closed_loop", "indoor"))
    ap.add_argument("--no-diag", action="store_true")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(C.T2, "configs", a.config)))
    dev = torch.device("cuda")
    t_start = time.time()
    L = S.Level(a.scan, a.level)
    v, f = S.mesh_zup(a.scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
    g = GE.grid_def(card)
    blk = T12.Blocker(v, f)
    tag = f"{L.key}_S{a.start}"
    od = os.path.join(a.out, a.group, tag)
    assert not os.path.exists(os.path.join(od, "episode.json")), od
    os.makedirs(os.path.join(od, "svc"), exist_ok=True)
    gt = None
    if a.group in CC.GROUPS:
        os.makedirs(os.path.join(od, "cq"), exist_ok=True)
        gt = IndoorGT(S, L, sen, cov, card, v, f, dev)
    proc, hello = child(a.group, L.key, tag, od, cfg)
    json.dump(hello.get("info", {}), open(os.path.join(od, "planner_info.json"), "w"), indent=1)
    s_local = next(i for i, p in enumerate(L.panos) if p["idx"] == int(a.start))
    pose, snap_m = T12.start_pose(g, S.rt_to_zup_m(np.asarray(L.pano_rt[s_local], np.float64)), blk)
    hist, covs, steps = [], [], []
    plog = open(os.path.join(od, "planner.jsonl"), "w")
    expect = None                  # (sample points, gain) of the executed candidate, checked at the next frame
    checks = {"executed_sample_equal": 0, "executed_gain_equal": 0, "state_hash_equal": 0, "pick_in_candidates": 0, "n": 0}
    refresh_center = None
    for t in range(a.moves + 1):
        hist.append(pose)
        e, az = GE.angles(g, pose)
        R, T = sen.RT(GE.position(g, pose), e, az)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(L.key, a.start, t, "sample"))
        c_prev = covs[-1] if covs else None
        covs.append(cov.update(pts.cpu().numpy()))
        if expect is not None:
            eq_s = bool(np.array_equal(pts.cpu().numpy().astype(np.float32), expect[0]))
            eq_g = bool(abs((covs[-1] - c_prev) - expect[1]) <= 1e-12)
            checks["executed_sample_equal"] += eq_s
            checks["executed_gain_equal"] += eq_g
            if not (eq_s and eq_g):
                raise RuntimeError(f"STOP: executed candidate differs from the next frame at t={t} (sample {eq_s}, gain {eq_g})")
        if c_prev is not None and covs[-1] < c_prev:
            raise RuntimeError(f"STOP: C_obs decreased at t={t}")
        legal = T12.legal_actions(g, pose, blk)
        rec = T12.service_record(L.key, a.start, t, fr, pts, flat, pose, hist, legal, g, sen)
        sp = os.path.join(od, "svc", f"svc_t{t}.npz")
        np.savez(sp, **rec)
        if t % 10 == 0:
            refresh_center = rec["cam_center"]
        if t == a.moves:
            break
        tq = time.time()
        req = {"t": t, "svc": sp}
        gt_s, gains, pts_all, news, rem = 0.0, None, None, None, None
        if gt is not None:
            h0 = gt.state_hash()
            if a.group == "C1":
                CC.write_cq(os.path.join(od, "cq", CC.cq_name(t)), "C1", t, rec["legal_idx"], tstar=gt.tstar(rec["legal_R"], rec["legal_T"]))
            else:
                gains, pts_all, news, rem = gt.gains(rec["legal_R"], rec["legal_T"], a.start, t)
                CC.write_cq(os.path.join(od, "cq", CC.cq_name(t)), "C2", t, rec["legal_idx"], gain=gains)
            if gt.state_hash() != h0:
                raise RuntimeError(f"STOP: GT query changed the simulator state at t={t}")
            checks["state_hash_equal"] += 1
            gt_s = time.time() - tq
            req["cq"] = os.path.join(od, "cq", CC.cq_name(t))
        proc.stdin.write(json.dumps(req) + "\n")
        proc.stdin.flush()
        ans = json.loads(proc.stdout.readline())
        if "error" in ans:
            raise RuntimeError(f"planner error at t={t}: {ans['error']}")
        plog.write(json.dumps(ans) + "\n")
        plog.flush()
        if ans["guard_denied"]:
            raise RuntimeError(f"INVALID: planner guard denied {ans['guard_denied']}")
        pk = ans["pick"]
        if pk is not None and not (0 <= pk < len(legal)):
            raise RuntimeError(f"INVALID: illegal pick {pk} of {len(legal)}")
        step = {"t": t, "pose": list(pose), "n_legal": len(legal), "pick": pk, "fallback": ans["fallback"],
                "decision_s": ans["timing"]["decision_s"], "roundtrip_s": time.time() - tq, "gt_query_s": gt_s, "cov": covs[-1]}
        if "nv_candidates" in ans:
            ok = pk is None or pk in ans["nv_candidates"]
            checks["pick_in_candidates"] += ok
            if not ok:
                raise RuntimeError(f"INVALID: pick {pk} outside the prefer-unvisited candidate set at t={t}")
        # ---- diagnostics (after the answer; never sent to the child)
        if gt is not None and not a.no_diag:
            td = time.time()
            if gains is None:
                gains, pts_all, news, rem = gt.gains(rec["legal_R"], rec["legal_T"], a.start, t)
            hist_set = {tuple(int(x) for x in r) for r in rec["executed_pose_idx_history"][:-1]}
            cand = [j for j, p in enumerate(legal) if tuple(int(x) for x in p) not in hist_set] or list(range(len(legal)))
            diag = {"gain": gains.tolist(), "cand": cand, **CC.oracle_summary(gains, cand, pk), **gt.region_diag(refresh_center, rem, news)}
            blocked = [q for q in GE.neighbours(g, pose) if q not in legal]
            if blocked:
                bR, bT = [], []
                for q in blocked:
                    Rq, Tq = sen.RT(GE.position(g, q), *GE.angles(g, q))
                    bR.append(Rq[0].cpu().numpy()); bT.append(Tq[0].cpu().numpy())
                gb, _, _, _ = gt.gains(bR, bT, a.start, t)
                diag["gain_blocked"] = gb.tolist()
                diag["gstar_blocked"] = float(gb.max())
            else:
                diag["gain_blocked"], diag["gstar_blocked"] = [], None
            if gt.state_hash() != h0:
                raise RuntimeError(f"STOP: diagnostic GT query changed the simulator state at t={t}")
            diag["diag_s"] = time.time() - td
            step["diag"] = diag
        if gains is not None and pk is not None:
            expect = (pts_all[pk], float(gains[pk]))
        else:
            expect = None
        checks["n"] += 1
        steps.append(step)
        pose = legal[pk] if pk is not None else pose
        print(tag, a.group, t, f"cov {covs[-1]:.3f}", f"dec {ans['timing']['decision_s']:.2f}s gt {gt_s:.2f}s", flush=True)
    proc.stdin.write(json.dumps({"bye": True}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=120)
    c = np.asarray(covs)
    ep = {"tag": tag, "group": a.group, "level_key": L.key, "start": a.start, "moves": a.moves, "config": a.config,
          "grid": {"n": list(g["n"]), "step_m": list(g["step_m"]), "start_pose": list(hist[0]), "start_snap_m": snap_m},
          "coverage": c.tolist(), "auc100": float(np.trapz(c[:101], dx=1.0) / 100) if a.moves >= 100 else None,
          "auc150": float(np.trapz(c, dx=1.0) / a.moves), "cov_final": float(c[-1]),
          **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100, 150) if x <= a.moves},
          "executed": [list(p) for p in hist], "steps": steps, "checks": checks, "wall_s": time.time() - t_start,
          "host": os.uname()[1], "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "planner_exit": proc.returncode}
    json.dump(ep, open(os.path.join(od, "episode.json"), "w"), indent=1, default=float)
    print(tag, a.group, "AUC %.3f Final %.3f wall %.0fs" % (ep["auc150"], c[-1], ep["wall_s"]), checks, flush=True)


if __name__ == "__main__":
    main()
