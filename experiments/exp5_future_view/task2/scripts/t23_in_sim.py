"""EXP5 Task 2 -- indoor (MP3D) closed loop of the final comparison (magician env; simulator / GT side; protocol §4, §6).

= the T0.12 grid simulator (task0/magician_comparison/scripts/t12_sim.py; grid, legality, start snapping, sensor,
sampling seeds, C_obs evaluator and service-record format imported unchanged) with the Task 2 groups:
  V2O    G0  V2 Original + DistP1 + Greedy-NV, n_win = 5   (t23_v2_in_planner.py, seen2scene env)
  V2M    G1  V2-Mix      + DistP1 + Greedy-NV, n_win = 5   (t23_v2_in_planner.py)
  MAG-G  G2  MAGICIAN official e697009, beam 1 x 1        (magician_planner_grid.py, unchanged)
  MAG-B  G3  MAGICIAN official e697009, beam 10 x 10      (magician_planner_grid.py, unchanged)
Budget: C0 + 150 moves = 151 frames for every group. Checkpoints / sha256 from task2/configs/eval_indoor_v1.json.
Output W2/eval/indoor/closed_loop/<group>/<level>_S<start>/{svc/, episode.json, planner.jsonl, planner.err}.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

MCS = os.path.join(C.EXP5, "task0", "magician_comparison")
sys.path.insert(0, os.path.join(MCS, "scripts"))
sys.path.insert(0, os.path.join(MCS, "adapters"))
import t12_sim as T12  # noqa: E402
import grid_env as GE  # noqa: E402

import numpy as np  # noqa: E402

S = T12.S
S2E, MGE = T12.S2E, T12.MGE
OUT = os.path.join(C.W2, "eval", "indoor", "closed_loop")


def child(group, key, tag, od, cfg):
    if group.startswith("MAG-"):
        env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
        cmd = [f"{MGE}/bin/python", "-I", os.path.join(MCS, "adapters", "magician_planner_grid.py"), "--method", group, "--card", key, "--tag", tag]
    else:
        g = cfg["groups"][group]
        env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
        cmd = [f"{S2E}/bin/python", os.path.join(HERE, "t23_v2_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    return p, hello


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--group", choices=["V2O", "V2M", "MAG-G", "MAG-B"], required=True)
    ap.add_argument("--moves", type=int, default=C.MOVES_INDOOR)
    ap.add_argument("--config", default="eval_indoor_v1.json")
    ap.add_argument("--out", default=OUT)
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
    proc, hello = child(a.group, L.key, tag, od, cfg)
    json.dump(hello.get("info", {}), open(os.path.join(od, "planner_info.json"), "w"), indent=1)
    s_local = next(i for i, p in enumerate(L.panos) if p["idx"] == int(a.start))
    pose, snap_m = T12.start_pose(g, S.rt_to_zup_m(np.asarray(L.pano_rt[s_local], np.float64)), blk)
    hist, covs, steps = [], [], []
    plog = open(os.path.join(od, "planner.jsonl"), "w")
    for t in range(a.moves + 1):
        hist.append(pose)
        e, az = GE.angles(g, pose)
        R, T = sen.RT(GE.position(g, pose), e, az)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(L.key, a.start, t, "sample"))
        covs.append(cov.update(pts.cpu().numpy()))
        legal = T12.legal_actions(g, pose, blk)
        rec = T12.service_record(L.key, a.start, t, fr, pts, flat, pose, hist, legal, g, sen)
        sp = os.path.join(od, "svc", f"svc_t{t}.npz")
        np.savez(sp, **rec)
        if t == a.moves:
            break
        tq = time.time()
        proc.stdin.write(json.dumps({"t": t, "svc": sp}) + "\n")
        proc.stdin.flush()
        ans = json.loads(proc.stdout.readline())
        plog.write(json.dumps(ans) + "\n")
        plog.flush()
        if ans["guard_denied"]:
            raise RuntimeError(f"INVALID: planner guard denied {ans['guard_denied']}")
        pk = ans["pick"]
        if pk is not None and not (0 <= pk < len(legal)):
            raise RuntimeError(f"INVALID: illegal pick {pk} of {len(legal)}")
        steps.append({"t": t, "pose": list(pose), "n_legal": len(legal), "pick": pk, "fallback": ans["fallback"],
                      "decision_s": ans["timing"]["decision_s"], "roundtrip_s": time.time() - tq, "cov": covs[-1]})
        pose = legal[pk] if pk is not None else pose
        print(tag, a.group, t, f"cov {covs[-1]:.3f}", f"dec {ans['timing']['decision_s']:.1f}s", flush=True)
    proc.stdin.write(json.dumps({"bye": True}) + "\n")
    proc.stdin.flush()
    proc.wait(timeout=120)
    c = np.asarray(covs)
    ep = {"tag": tag, "group": a.group, "level_key": L.key, "start": a.start, "moves": a.moves, "config": a.config,
          "grid": {"n": list(g["n"]), "step_m": list(g["step_m"]), "start_pose": list(hist[0]), "start_snap_m": snap_m},
          "coverage": c.tolist(), "auc100": float(np.trapz(c[:101], dx=1.0) / 100), "auc150": float(np.trapz(c, dx=1.0) / a.moves),
          "cov_final": float(c[-1]), **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100, 150) if x <= a.moves},
          "executed": [list(p) for p in hist], "steps": steps, "wall_s": time.time() - t_start}
    json.dump(ep, open(os.path.join(od, "episode.json"), "w"), indent=1, default=float)
    print(tag, a.group, "AUC150 %.3f Cov150 %.3f wall %.0fs" % (ep["auc150"], c[-1], ep["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
