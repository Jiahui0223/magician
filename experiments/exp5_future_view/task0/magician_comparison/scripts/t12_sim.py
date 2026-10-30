"""EXP5 Task 0 / T0.12 -- indoor closed loop on the released-code-style camera GRID (magician env; simulator / GT side;
protocol v1.3 §19-§21). Same sensor (frozen MAGICIAN sensor, 456 x 256, 60 deg, 10 m), same 10 % sampling seeds, same
level-scale C_obs evaluator and the same service-record format as t06_sim / t10_sim; only the action space differs:
poses (l, w, h, theta, azim) of adapters/grid_env.py, official neighbours, official layer-0 GT segment rule.
Methods (output group name in parentheses):
  RAND-NV  (G-RAND)    seeded uniform choice among the legal poses not executed before (all legal if none); no child
  V2K4     (G-V2K4)    frozen scripts/t06_planner.py --method V2K4 (seen2scene env), unchanged
  V2K4-NV  (G-V2K4-NV) adapters/v2nv_planner.py (seen2scene env)
  MAG-G / MAG-B (G-MAG-G / G-MAG-B) adapters/magician_planner_grid.py (magician env; official grid neighbours)
Output MC/grid_protocol/closed_loop/<group>/<level>_S<start>/{svc/, episode.json, planner.jsonl}.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import mp3d_sim as S  # noqa: E402
import grid_env as GE  # noqa: E402

import numpy as np  # noqa: E402

S2E = "/bsuscratch/jiahuizhang/envs/seen2scene"
MGE = "/bsuscratch/jiahuizhang/envs/magician"
ADP = os.path.join(HERE, "..", "adapters")
T0S = os.path.join(HERE, "..", "..", "scripts")
OUT = os.path.join(S.W, "magician_comparison", "grid_protocol", "closed_loop")
GROUP = {"RAND-NV": "G-RAND", "V2K4": "G-V2K4", "V2K4-NV": "G-V2K4-NV", "MAG-G": "G-MAG-G", "MAG-B": "G-MAG-B"}


class Blocker:
    """Official line_segment_mesh_intersection on the whole-house GT mesh (z-up metres; simulator side only)."""

    def __init__(self, v, f):
        import trimesh
        self.ri = trimesh.ray.ray_triangle.RayMeshIntersector(trimesh.Trimesh(v, f, process=False))

    def blocked(self, p, q):
        d = q - p
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        loc, _, _ = self.ri.intersects_location([p], [d / n])
        return bool(len(loc)) and bool(np.any(np.linalg.norm(loc - p, axis=1) < n))


class RandNV:
    """Reference policy on the grid: seeded uniform among unexecuted legal poses (all legal if none)."""

    def __init__(self, tag):
        self.tag, self.stdin, self.stdout, self._q = tag, self, self, None

    def write(self, s):
        self._q = json.loads(s)

    def flush(self):
        pass

    def readline(self):
        q = self._q
        z = np.load(q["svc"])
        hist = {tuple(int(x) for x in r) for r in z["executed_pose_idx_history"][:-1]}
        legal = [tuple(int(x) for x in r) for r in z["legal_idx"]]
        C = [j for j, p in enumerate(legal) if p not in hist] or list(range(len(legal)))
        rng = np.random.default_rng(int(hashlib.sha256(f"exp5-T0.12|rand|{self.tag}|{q['t']}".encode()).hexdigest()[:15], 16))
        return json.dumps({"t": q["t"], "pick": int(C[int(rng.integers(len(C)))]) if C else None, "fallback": False,
                           "guard_denied": [], "refresh": None, "timing": {"decision_s": 0.0}}) + "\n"

    def wait(self, timeout=None):
        return 0


def service_record(scene_key, start, t, fr, pts, flat, pose, hist, legal, g, sen):
    """t06 service-record format with grid poses (legal centres / R / T from the grid)."""
    legal_R, legal_T, legal_C = [], [], []
    for a in legal:
        X = GE.position(g, a)
        e, az = GE.angles(g, a)
        R, T = sen.RT(X, e, az)
        legal_R.append(R[0].cpu().numpy())
        legal_T.append(T[0].cpu().numpy())
        legal_C.append(X)
    rec = {"scene_key": np.asarray(f"{scene_key}/{start}"), "start": np.asarray(start), "t": np.asarray(t),
           "zbuf": fr["zbuf"][0, ..., 0].cpu().numpy().astype(np.float32), "mask": fr["mask"][0, ..., 0].cpu().numpy().astype(bool),
           "R": fr["R"][0].cpu().numpy().astype(np.float64), "T": fr["T"][0].cpu().numpy().astype(np.float64),
           "cam_center": fr["fov"].get_camera_center()[0].cpu().numpy().astype(np.float64),
           "pose_idx": np.asarray(pose, np.int64), "sample_pixel_flat": flat.numpy().astype(np.int64),
           "sample_points": pts.cpu().numpy().astype(np.float32), "executed_pose_idx_history": np.asarray(hist, np.int64),
           "legal_idx": np.asarray(legal, dtype=np.int64).reshape(-1, 5), "legal_R": np.asarray(legal_R, np.float64).reshape(-1, 3, 3),
           "legal_T": np.asarray(legal_T, np.float64).reshape(-1, 3), "legal_center": np.asarray(legal_C, np.float64).reshape(-1, 3)}
    return S.MC.validate_service_record(rec)


def start_pose(g, P_m, blk):
    """Nearest grid position to the panorama centre (z-up m) whose segment from it is not blocked; elev 0, azim 0."""
    n = g["n"]
    idx = np.stack(np.meshgrid(*[np.arange(k) for k in n], indexing="ij"), -1).reshape(-1, 3)
    pos_m = S.rt_to_zup_m(np.stack([GE.position(g, i) for i in idx]))
    for j in np.argsort(np.linalg.norm(pos_m - P_m, axis=1), kind="stable"):
        if not blk.blocked(P_m, pos_m[j]):
            return (int(idx[j][0]), int(idx[j][1]), int(idx[j][2]), 2, 0), float(np.linalg.norm(pos_m[j] - P_m))
    raise RuntimeError("no reachable start grid position")


def legal_actions(g, pose, blk):
    p = S.rt_to_zup_m(GE.position(g, pose))
    out = []
    for q in GE.neighbours(g, pose):
        if tuple(q[:3]) == tuple(pose[:3]) or not blk.blocked(p, S.rt_to_zup_m(GE.position(g, q))):
            out.append(q)
    return out


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", required=True)
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--method", choices=list(GROUP), required=True)
    ap.add_argument("--moves", type=int, default=150)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    dev = torch.device("cuda")
    t_start = time.time()
    L = S.Level(a.scan, a.level)
    v, f = S.mesh_zup(a.scan)
    sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
    cov = S.Coverage(L, sen)
    card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
    g = GE.grid_def(card)
    blk = Blocker(v, f)
    tag = f"{L.key}_S{a.start}"
    od = os.path.join(a.out, GROUP[a.method], tag)
    os.makedirs(os.path.join(od, "svc"), exist_ok=True)
    if a.method == "RAND-NV":
        child = RandNV(tag)
    else:
        if a.method.startswith("MAG-"):
            env = {k: v_ for k, v_ in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
            cmd = [f"{MGE}/bin/python", "-I", os.path.join(ADP, "magician_planner_grid.py")]
        elif a.method == "V2K4-NV":
            env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
            cmd = [f"{S2E}/bin/python", os.path.join(ADP, "v2nv_planner.py")]
        else:
            env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
            cmd = [f"{S2E}/bin/python", os.path.join(T0S, "t06_planner.py")]
        child = subprocess.Popen(cmd + ["--method", a.method, "--card", L.key, "--tag", tag], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
        hello = json.loads(child.stdout.readline())
        assert hello["ready"]
        json.dump(hello.get("info", {}), open(os.path.join(od, "planner_info.json"), "w"), indent=1)
    s_local = next(i for i, p in enumerate(L.panos) if p["idx"] == int(a.start))
    pose, snap_m = start_pose(g, S.rt_to_zup_m(np.asarray(L.pano_rt[s_local], np.float64)), blk)
    hist, covs, steps = [], [], []
    plog = open(os.path.join(od, "planner.jsonl"), "w")
    for t in range(a.moves + 1):
        hist.append(pose)
        e, az = GE.angles(g, pose)
        R, T = sen.RT(GE.position(g, pose), e, az)
        fr = sen.render(R, T)
        pts, flat, _, _, _ = sen.sample(fr, S.derived_seed(L.key, a.start, t, "sample"))
        covs.append(cov.update(pts.cpu().numpy()))
        legal = legal_actions(g, pose, blk)
        rec = service_record(L.key, a.start, t, fr, pts, flat, pose, hist, legal, g, sen)
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
                      "refresh_s": (ans.get("refresh") or {}).get("refresh_s"), "cov": covs[-1]})
        pose = legal[pk] if pk is not None else pose
        print(tag, a.method, t, f"cov {covs[-1]:.3f}", f"dec {ans['timing']['decision_s']:.1f}s", flush=True)
    if a.method != "RAND-NV":
        child.stdin.write(json.dumps({"bye": True}) + "\n")
        child.stdin.flush()
        child.wait(timeout=120)
    c = np.asarray(covs)
    ep = {"tag": tag, "method": a.method, "group": GROUP[a.method], "level_key": L.key, "start": a.start, "moves": a.moves,
          "grid": {"n": list(g["n"]), "step_m": list(g["step_m"]), "start_pose": list(hist[0]), "start_snap_m": snap_m},
          "coverage": c.tolist(), "auc100": float(np.trapz(c[:101], dx=1.0) / 100), "auc": float(np.trapz(c, dx=1.0) / a.moves),
          "cov_final": float(c[-1]), **{f"cov{x}": float(c[x]) for x in (10, 20, 50, 100, 150) if x <= a.moves},
          "executed": [list(p) for p in hist], "steps": steps, "wall_s": time.time() - t_start}
    json.dump(ep, open(os.path.join(od, "episode.json"), "w"), indent=1, default=float)
    print(tag, a.method, "AUC100 %.3f Cov100 %.3f Cov%d %.3f wall %.0fs" % (ep["auc100"], c[min(100, a.moves)], a.moves, c[-1], ep["wall_s"]), flush=True)


if __name__ == "__main__":
    main()
