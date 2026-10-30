"""EXP5 Task 3 / E2 pilot -- D100 training data with the FROZEN builders (seen2scene env, 1 GPU per worker).

The builders replay exactly what the planner sees at state t (records 0..t; candidate t* = labels only) and call
m_p1.P1Fusion; here m_p1.P1Fusion = p1_full.P1FusionFull (the E1 / E1.2 D100 map: the stored 10 % points bitwise +
all other valid pixels of the SAME records), everything else unchanged. Outputs go to the e2 roots (e2_paths).
  mp3d <worker> <n>          task2/scripts/t21_build.py main with C.W2 -> W2D (claims across nodes)
  out <scene> [<scene> ...]  task1/scripts/t11_build.py build() for every trajectory of the scene -> OUTD
  check-mp3d KEY TRAJ | check-out SCENE TRAJ   smoke: build one trajectory sparse (must equal the frozen shard
                             bitwise) and D100; compare the non-P1 arrays and the view lists; timings
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_paths as EP  # noqa: E402

sys.path.insert(0, os.path.join(EP.EXP5, "task2", "scripts"))
sys.path.insert(0, os.path.join(EP.EXP5, "task1", "scripts"))
import t2_common as C  # noqa: E402
import t21_build as B21  # noqa: E402
import t11_build as B11  # noqa: E402
import v2_frozen as VF  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import m_p1  # noqa: E402

sys.path.insert(0, os.path.join(EP.EXP5, "task3_full_depth_p1", "scripts"))
import p1_full as PF  # noqa: E402

SPARSE = m_p1.P1Fusion


_CHECK_L3 = B11.check_l3


def use(mode, outdoor=False):
    """d100: the D100 map; for the outdoor builder the D100 map carries a sparse twin of the same records (p1_full.
    P1FusionB2) and the frozen T-L5 check (P1 replay == stored L3 keyframe state) is applied to that twin, so the
    record-replay integrity check stays exactly as strict; inputs / labels come from the D100 map."""
    if mode == "d100":
        m_p1.P1Fusion = PF.P1FusionB2 if outdoor else PF.P1FusionFull
        B11.check_l3 = (lambda fu, p: _CHECK_L3(fu.sparse, p) and bool(torch.equal(fu.empty, fu.sparse.empty))) if outdoor else _CHECK_L3
    else:
        m_p1.P1Fusion = SPARSE
        B11.check_l3 = _CHECK_L3


def compare(a, b, p1_keys_hint=()):
    za, zb = np.load(a), np.load(b)
    ka, kb = set(za.files), set(zb.files)
    same, diff = [], []
    for k in sorted(ka & kb):
        (same if np.array_equal(za[k], zb[k]) else diff).append(k)
    return {"keys_only_a": sorted(ka - kb), "keys_only_b": sorted(kb - ka), "equal": same, "differ": diff}


def main():
    cmd = sys.argv[1]
    EP.setup_roots()
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    if cmd == "mp3d":
        use("d100")
        C.W2 = EP.W2D
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        B21.main()
    elif cmd == "out":
        use("d100", outdoor=True)
        geo = None
        for scene in sys.argv[2:]:
            for tj in sorted(f[:-4] for f in os.listdir(os.path.join(VF.WORK, "data", "views", scene)) if f.endswith(".npz")):
                tb = time.time()
                s = B11.build(scene, tj, os.path.join(VF.WORK, "data", "tstar"), EP.OUTD, dev, geo)
                print(scene, tj, "built" if s else "exists", "%.0fs" % (time.time() - tb), flush=True)
                torch.cuda.empty_cache()
    elif cmd == "check-mp3d":
        key, traj = sys.argv[2], sys.argv[3]
        job = next(j for j in json.load(open(os.path.join(EP.T2W, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
                   if j["key"] == key and j["traj"] == traj)
        res = {}
        for mode, root in (("sparse", os.path.join(EP.CHECK, "w2_sparse")), ("d100", EP.W2D)):
            if mode == "sparse":
                os.makedirs(os.path.join(root, "mp3d"), exist_ok=True)
                for x in os.listdir(os.path.join(EP.T2W, "mp3d")):
                    if x not in ("views", "views_claims") and not os.path.lexists(os.path.join(root, "mp3d", x)):
                        os.symlink(os.path.join(EP.T2W, "mp3d", x), os.path.join(root, "mp3d", x))
            use(mode)
            C.W2 = root
            tb = time.time()
            s = B21.build(job, dev)
            res[mode] = {"seconds": time.time() - tb, "summary": {k: v for k, v in (s or {}).items() if k in ("views", "cls_frac", "m_unknown_cat")}}
            torch.cuda.empty_cache()
        orig = os.path.join(EP.T2W, "mp3d", "views", key, f"{traj}.npz")
        res["sparse_vs_frozen"] = compare(orig, os.path.join(EP.CHECK, "w2_sparse", "mp3d", "views", key, f"{traj}.npz"))
        res["d100_vs_frozen"] = compare(orig, os.path.join(EP.W2D, "mp3d", "views", key, f"{traj}.npz"))
        print(json.dumps(res, indent=1, default=str))
    elif cmd == "check-out":
        scene, traj = sys.argv[2], sys.argv[3]
        res = {}
        for mode, root in (("sparse", os.path.join(EP.CHECK, "out_sparse")), ("d100", EP.OUTD)):
            use(mode, outdoor=True)
            tb = time.time()
            B11.build(scene, traj, os.path.join(VF.WORK, "data", "tstar"), root, dev, None)
            res[mode] = {"seconds": time.time() - tb}
            torch.cuda.empty_cache()
        orig = os.path.join(VF.WORK, "data", "views", scene, f"{traj}.npz")
        res["sparse_vs_frozen"] = compare(orig, os.path.join(EP.CHECK, "out_sparse", scene, f"{traj}.npz"))
        res["d100_vs_frozen"] = compare(orig, os.path.join(EP.OUTD, scene, f"{traj}.npz"))
        print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
