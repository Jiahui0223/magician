"""EXP5 Task 0.6 -- indoor closed-loop simulator (magician env; simulator / GT side), protocol M.

One episode: level (scan, level), start panorama, method; 100 moves on the panorama graph action set (rotate +-1 azim /
+-1 elev bin, or move to a graph neighbour keeping the orientation; GT line of sight = collision rule), observation
= frozen sensor on the GT mesh, level-scale C_obs per step (101 values). The planner is a child process
(seen2scene env, t06_planner.py) that receives only {t, path of the whitelisted service record}; coverage and GT
never reach it. An empty legal list -> hover (stay; same rule as MAGICIAN's dead-end patch).
Outputs W/closed_loop/<method>/<tag>/{svc/, episode.json, planner.jsonl}. Re-running reproduces the episode bitwise.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402

S2E = "/bsuscratch/jiahuizhang/envs/seen2scene"


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--method", choices=["frozen", "C", "V2K4", "V2K1"], required=True)
    ap.add_argument("--moves", type=int, default=100)
    ap.add_argument("--out", default=os.path.join(S.W, "closed_loop"))
    a = ap.parse_args()
    dev = torch.device("cuda")
    t_start = time.time()
    L = S.Level(a.scan, a.level)
    L.build_graph()
    v, f = S.mesh_zup(a.scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    tag = f"{L.key}_S{a.start}"
    od = os.path.join(a.out, a.method, tag)
    os.makedirs(os.path.join(od, "svc"), exist_ok=True)
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
    child = subprocess.Popen([f"{S2E}/bin/python", os.path.join(HERE, "t06_planner.py"), "--method", a.method,
                              "--card", L.key, "--tag", tag], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    assert json.loads(child.stdout.readline())["ready"]
    s_local = next(i for i, p in enumerate(L.panos) if p["idx"] == int(a.start))
    pose = (s_local, 0, 0, 1, 0)
    hist, covs, steps = [], [], []
    plog = open(os.path.join(od, "planner.jsonl"), "w")
    for t in range(a.moves + 1):
        hist.append(pose)
        R, T = sen.RT(L.pano_rt[pose[0]], S.ELEVS[pose[3]], 360.0 * pose[4] / S.N_AZIM)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(L.key, a.start, t, "sample"))
        covs.append(cov.update(pts.cpu().numpy()))
        legal = S.legal_actions(L, pose)
        rec = S.service_record(L.key, a.start, t, fr, pts, flat, pose, hist, legal, L, sen)
        sp = os.path.join(od, "svc", f"svc_t{t}.npz")
        np.savez(sp, **rec)
        if t == a.moves:
            break
        tq = time.time()
        child.stdin.write(json.dumps({"t": t, "svc": sp}) + "\n")
        child.stdin.flush()
        ans = json.loads(child.stdout.readline())
        plog.write(json.dumps(ans) + "\n")
        plog.flush()
        if ans["guard_denied"]:
            raise RuntimeError(f"INVALID: planner guard denied {ans['guard_denied']}")
        pk = ans["pick"]
        if pk is not None and not (0 <= pk < len(legal)):
            raise RuntimeError(f"INVALID: illegal pick {pk} of {len(legal)}")
        steps.append({"t": t, "pose": list(pose), "n_legal": len(legal), "pick": pk, "fallback": ans["fallback"],
                      "decision_s": ans["timing"]["decision_s"], "roundtrip_s": time.time() - tq,
                      "refresh_s": (ans["refresh"] or {}).get("refresh_s"), "cov": covs[-1]})
        pose = legal[pk] if pk is not None else pose
        print(tag, a.method, t, f"cov {covs[-1]:.3f}", f"dec {ans['timing']['decision_s']:.1f}s", flush=True)
    child.stdin.write(json.dumps({"bye": True}) + "\n")
    child.stdin.flush()
    child.wait(timeout=120)
    c = np.asarray(covs)
    ep = {"tag": tag, "method": a.method, "level_key": L.key, "start": a.start, "moves": a.moves, "coverage": c.tolist(),
          "auc": float(np.trapz(c, dx=1.0) / a.moves), "cov_final": float(c[-1]),
          **{f"cov{x}": float(c[x]) for x in (10, 20, 50) if x <= a.moves},
          "executed": [list(p) for p in hist], "steps": steps, "wall_s": time.time() - t_start}
    json.dump(ep, open(os.path.join(od, "episode.json"), "w"), indent=1, default=float)
    print(tag, a.method, "AUC %.3f Cov %.3f wall %.0fs" % (ep["auc"], ep["cov_final"], ep["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
