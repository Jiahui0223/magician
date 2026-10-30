"""EXP5 Task 1.1 D-BUILD (seen2scene env, 1 GPU; planner-side information only + the label file t* of t11_labels.py).

Per trajectory: replay the frozen P1 fusion over the whitelisted service records t = 0..99 (T-L5: bitwise equal to the
L3 keyframe states), and at every t
  - the coarse 8^3 grid of the P1 structure (encoder input),
  - for every legal candidate (shift = 0 for S-c trajectories, seeded otherwise): projection statistics and the label
    encoding of t* (protocol §2.3-§2.4); first view of every trajectory: T-R1 (rays = abl_score.pose_rays bitwise for
    shift 0) and the march range equal to SceneCaster's.
Output <out>/<scene>/<traj>.npz (written atomically, read-only) + a per-trajectory summary json.
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
import t5_diagnostic as TD  # noqa: E402
import t0_state_io as SIO  # noqa: E402
import abl_score as SC  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402

CFG = VF.load_cfg("data_v1.json")
KEYF = CFG["keyframes_l3"]


def check_l3(fu, p):
    z = SIO.load_state(p)
    hb = (fu.cnt > 0).cpu().numpy()
    ok = (np.array_equal(hb, z["band_mask"]) and np.array_equal(fu.empty.cpu().numpy(), z["empty_mask"])
          and int(z["n_frames"]) == fu.n_frames)
    if ok:
        fb = np.flatnonzero(hb)
        ok = (np.array_equal(fu.sum.cpu().numpy()[fb], z["band_sum"]) and np.array_equal(fu.cnt.cpu().numpy()[fb], z["band_count"]))
    return ok


def build(scene, traj, tstar_root, out_root, dev, geo=None):
    p_out = os.path.join(out_root, scene, f"{traj}.npz")
    if os.path.exists(p_out):
        return None
    tb = time.time()
    fr, dims, dom, win = geo or TD.scene_geometry(scene)
    L = np.load(os.path.join(tstar_root, scene, f"{traj}.npz"))
    tstar_all, vt, va, vR, vT, vJ = L["tstar"], L["t"], L["a"], L["R"], L["T"], L["jitter"]
    hold = bool(L["heldout"])
    rec_dir = os.path.join(VF.DATA_V1, "trajectories", scene, traj, "records")
    st_dir = os.path.join(VF.DATA_V1, "trajectories", scene, traj, "states")
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    Nv, R = tstar_all.shape
    arr = {k: np.zeros((Nv, R), dt) for k, dt in (("cls", np.uint8), ("nseg", np.uint8), ("b", np.int8), ("cat", np.uint8))}
    for k in ("tk", "dlo", "dhi", "lu", "y"):
        arr[k] = np.zeros((Nv, R), np.float16)
    coarse, l3, checks = [], {}, {}
    files = {int(f.split("_t")[-1][:-4]): os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.endswith(".npz")}
    assert sorted(files) == list(range(101)), (scene, traj)
    vi = 0
    for t in range(100):
        rec = MC.load_service_record(files[t])
        assert int(rec["t"]) == t
        fu.integrate_record(rec)
        if t in KEYF:
            l3[t] = check_l3(fu, os.path.join(st_dir, f"state_t{t:03d}.npz"))
            if not l3[t]:
                raise VF.InvalidRun(f"INVALID T-L5: P1 replay differs from L3 state {scene}/{traj} t={t}")
        sg = VF.StateGeometry(fr, dims, dom, fu, dev)
        coarse.append(G.coarse_grid(sg.struct, sg.domain).cpu().numpy())
        n = len(rec["legal_R"])
        assert np.array_equal(vt[vi:vi + n], np.full(n, t)) and np.array_equal(va[vi:vi + n], np.arange(n)), (t, vi)
        for a in range(n):
            assert np.array_equal(vR[vi], rec["legal_R"][a]) and np.array_equal(vT[vi], rec["legal_T"][a])
            jit = tuple(vJ[vi])
            rays = VF.view_rays(sg.caster, vR[vi], vT[vi], jit)
            if jit == (0.0, 0.0) and "TR1" not in checks:
                ref = SC.pose_rays(sg.caster, vR[vi], vT[vi])
                checks["TR1"] = all(torch.equal(x, y) for x, y in zip(rays, ref))
                if not checks["TR1"]:
                    raise VF.InvalidRun("INVALID T-R1: rays differ from abl_score.pose_rays")
            if vi == 0:
                tmin, tmax = sg.caster._entry_exit(rays[0][None].expand_as(rays[1]), rays[1])
                tmax = torch.minimum(tmax, rays[2])
                g_ = G.march_range(rays[0], rays[1], rays[2], sg.ext)
                checks["march_equal"] = bool(torch.equal(g_[0], tmin) and torch.equal(g_[1], tmax))
                if not checks["march_equal"]:
                    raise VF.InvalidRun("INVALID: march range differs from SceneCaster")
            pr = sg.project(rays)
            ts = torch.as_tensor(tstar_all[vi], dtype=sg.caster.dt, device=sg.caster.dev)
            lab = G.encode_labels(rays[0], rays[1], pr, ts, rays[3], fr.k_M, sg.struct, sg.domain)
            tc = rays[2]

            def rel(x):
                return torch.where(torch.isfinite(x), (x / tc).clamp(0, 1), torch.ones_like(x)).cpu().numpy()
            arr["cls"][vi] = pr["cls"].cpu().numpy()
            arr["nseg"][vi] = pr["n_seg"].clamp(max=255).cpu().numpy()
            arr["tk"][vi], arr["dlo"][vi], arr["dhi"][vi] = rel(pr["t_known"]), rel(pr["d_lo"]), rel(pr["d_hi"])
            arr["lu"][vi] = (pr["n_u"].to(tc.dtype) * G.H_STEP / tc).clamp(0, 1).cpu().numpy()
            arr["b"][vi] = lab["b"].cpu().numpy()
            arr["y"][vi] = lab["y"].cpu().numpy()
            arr["cat"][vi] = lab["cat"].cpu().numpy()
            vi += 1
        del sg
    assert vi == Nv, (vi, Nv)
    coarse = np.stack(coarse)
    os.makedirs(os.path.dirname(p_out), exist_ok=True)
    tmp = p_out + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, coarse=coarse, t=vt, a=va, R=vR, T=vT, jitter=vJ, tstar=tstar_all, heldout=np.asarray(hold),
                        k_M=np.asarray(fr.k_M), dims=np.asarray(dims), **arr)
    os.replace(tmp, p_out)
    os.chmod(p_out, 0o444)
    mu = arr["cls"] == G.CLS_UNKNOWN
    cats = {k: int((arr["cat"][mu] == v).sum()) for k, v in G.CAT.items()}
    summ = {"scene": scene, "traj": traj, "heldout": hold, "views": int(Nv), "rays": int(Nv * R),
            "cls_frac": {n: float((arr["cls"] == c).mean()) for n, c in (("unknown", 0), ("known", 1), ("empty", 2))},
            "m_unknown_cat": cats, "l3_bitwise": l3, "checks": checks, "coarse_shape": list(coarse.shape),
            "seconds": time.time() - tb, "sha256": VF.sha256_file(p_out)}
    json.dump(summ, open(p_out[:-4] + ".json", "w"), indent=1)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--trajs", default="")
    ap.add_argument("--tstar", default=os.path.join(VF.WORK, "data", "tstar"))
    ap.add_argument("--out", default=os.path.join(VF.WORK, "data", "views"))
    ap.add_argument("--wait", action="store_true", help="wait for each label file of t11_labels.py")
    a = ap.parse_args()
    VF.verify_frozen()
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    geo = TD.scene_geometry(a.scene)
    trajs = a.trajs.split(",") if a.trajs else sorted(os.listdir(os.path.join(VF.DATA_V1, "trajectories", a.scene)))
    for tj in trajs:
        while a.wait and not os.path.exists(os.path.join(a.tstar, a.scene, f"{tj}.npz")):
            time.sleep(30)
        s = build(a.scene, tj, a.tstar, a.out, dev, geo)
        if s:
            print(f"{a.scene}/{tj}: {s['views']} views, unknown {s['cls_frac']['unknown']:.3f}, "
                  f"HIT {s['m_unknown_cat']['HIT']} NOHIT {s['m_unknown_cat']['NOHIT']} KNOWN {s['m_unknown_cat']['KNOWN']} "
                  f"INF_EMPTY {s['m_unknown_cat']['INF_EMPTY']}, {s['seconds']:.0f} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
