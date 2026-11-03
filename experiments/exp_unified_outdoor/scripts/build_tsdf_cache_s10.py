"""Unified outdoor benchmark -- S10 variant of the V3 4x partial-TSDF cache (outdoor only; seen2scene env).

= experiments/exp6_v3/shared/build_tsdf_cache.py build_outdoor() (lines 102-130) with exactly two substitutions:
  fusion   the frozen S10 P1 (experiments/exp3b_macaronspp_transfer/m_p1.P1Fusion: BAND from the simulator's 10 % sample
           points, EMPTY carving from the full depth) instead of p1_full.P1FusionFull (D100) -- the fusion the T1.1 S10
           shards were built with (experiments/exp5_future_view/task1/scripts/t11_build.py:57)
  check    the replayed coarse grid must equal the S10 shard's stored coarse grid bitwise at every state
           (work_dirs/exp5_future_view/task1/data/views/<scene>/<traj>.npz["coarse"]), STOP otherwise
Everything else is imported from the historical builder unchanged (coarse_of, save, claim; PT.aggregate_fusion / to_blob).
Output: work_dirs/exp_unified_outdoor/cache/tsdf4_s10/outdoor/<scene>/<traj>.npz (+ .json summary), the layout
seen2scene.v3.data.cache_path expects under CACHE = work_dirs/exp_unified_outdoor/cache/tsdf4_s10.

  python build_tsdf_cache_s10.py <worker index> <n workers> [--device cuda]          (formal; workers claim trajectories)
  python build_tsdf_cache_s10.py --check SCENE/TRAJ --states 3 --device cpu          (CPU feasibility / bitwise check, no output)
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import build_tsdf_cache as B  # noqa: E402  (sets the exp5 / exp3b paths; imports p1_full, t5_diagnostic, m_common)
import m_p1  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

PT, TD, MC, VF = B.PT, B.TD, B.MC, B.VF
S10 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task1", "data", "views")
W = os.path.join(REPO, "work_dirs", "exp_unified_outdoor", "cache", "tsdf4_s10")
MOVES_OUTDOOR = B.MOVES_OUTDOOR


def build_outdoor_s10(scene, tj, dev, geo=None, n_states=MOVES_OUTDOOR, save=True):
    p_out = os.path.join(W, "outdoor", scene, f"{tj}.npz")
    if save and os.path.exists(p_out):
        return None
    tb = time.time()
    shard = np.load(os.path.join(S10, scene, f"{tj}.npz"))
    fr, dims, dom, win = geo or TD.scene_geometry(scene)
    rec_dir = os.path.join(VF.DATA_V1, "trajectories", scene, tj, "records")
    files = {int(f.split("_t")[-1][:-4]): os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.endswith(".npz")}
    fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
    coarse = shard["coarse"]
    blobs, nbytes, shape, per_state = {}, 0, None, []
    for t in range(n_states):
        ts = time.time()
        rec = MC.load_service_record(files[t])
        assert int(rec["t"]) == t
        fu.integrate_record(rec)
        if not np.array_equal(B.coarse_of(fu), coarse[t]):
            raise SystemExit(f"STOP: replayed S10 fusion differs from the S10 shard coarse grid ({scene}/{tj} t={t})")
        blob, shape = PT.to_blob(PT.aggregate_fusion(fu))
        blobs[f"blob_{t}"] = blob
        nbytes += len(blob)
        per_state.append(time.time() - ts)
    meta = {"dataset": "outdoor", "scene": scene, "traj": tj, "factor": PT.FACTOR, "n_states": MOVES_OUTDOOR, "k_M": float(fr.k_M),
            "density": "s10", "fusion": "experiments/exp3b_macaronspp_transfer/m_p1.P1Fusion",
            "blocks": [{"b": 0, "origin": [float(x) for x in fr.origin_z] if hasattr(fr, "origin_z") else None,
                        "dims": [int(x) for x in dims], "shape4": [int(x) for x in shape]}]}
    summ = {"dataset": "outdoor", "density": "s10", "scene": scene, "traj": tj, "heldout": bool(shard["heldout"]), "states": n_states,
            "bytes": nbytes, "seconds": time.time() - tb, "seconds_per_state": per_state, "shape4": meta["blocks"][0]["shape4"]}
    del fu
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    if save:
        B.save(p_out, blobs, meta, summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worker", nargs="?", type=int, default=0)
    ap.add_argument("n_workers", nargs="?", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--check", default=None, help="SCENE/TRAJ: replay --states states, compare with the S10 shard, write nothing")
    ap.add_argument("--states", type=int, default=3)
    a = ap.parse_args()
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(a.device)
    if a.check:
        s, tj = a.check.split("/")
        r = build_outdoor_s10(s, tj, dev, n_states=a.states, save=False)
        print(json.dumps({k: v for k, v in r.items()}, indent=1))
        return
    B.W = W                                           # claims directory of the historical builder -> the S10 cache root
    items = [(s, f[:-4]) for s in sorted(os.listdir(S10)) for f in sorted(os.listdir(os.path.join(S10, s))) if f.endswith(".npz")]
    off = (len(items) * a.worker) // max(1, a.n_workers)
    geo = {}
    for s, tj in items[off:] + items[:off]:
        if os.path.exists(os.path.join(W, "outdoor", s, f"{tj}.npz")) or not B.claim(f"outdoor_s10_{s}_{tj}"):
            continue
        geo.setdefault(s, TD.scene_geometry(s))
        r = build_outdoor_s10(s, tj, dev, geo[s])
        if r:
            print(f"outdoor-s10 {s}/{tj} {r['states']} states {r['bytes'] / 1e6:.0f} MB {r['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
