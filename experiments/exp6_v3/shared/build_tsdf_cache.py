"""EXP6 V3 -- 4x partial-TSDF cache of every training / selection state (seen2scene env, 1 GPU per worker).

Replays the stored observation records with the FROZEN D100 fusion (experiments/exp5_future_view/task3_full_depth_p1/
scripts/p1_full.P1FusionFull: BAND from all valid pixels of the SAME records, EMPTY carving unchanged) with exactly the
planner / E2-builder state rules, and stores seen2scene.v3.partial_tsdf.aggregate() of every state:
  MP3D     E2 trajectory jobs (train + valsel): at every refresh block b = 0, 10, .. the n_win = 5 region of record b
           (task2 t21_build.region), records 0..b-1 integrated, then the states t = b..b+9 one record at a time
  outdoor  T1.1 trajectories (train6 incl. S-c): one scene grid, records 0..99 integrated one at a time
Check (STOP on failure): at every state, the V2 coarse grid of the replayed fusion equals the E2 D100 shard's stored
coarse grid bitwise, which ties the cache to the shard's ray channels / labels (same P1 history, no future record).
Output W/shared_cache/tsdf4/<mp3d|outdoor>/<scene>/<traj>.npz : blob_<state index> (zlib uint8), meta (json), per
block origin / fine dims / 4x shape, k_M; + <traj>.json summary. Workers claim trajectories across nodes.
Usage: python build_tsdf_cache.py mp3d|outdoor <worker index> <n workers>
"""
import json
import os
import sys
import time

import numpy as np
import torch

from seen2scene import PROJECT_ROOT
from seen2scene.futureview import geometry as G
from seen2scene.v3 import partial_tsdf as PT

REPO = str(PROJECT_ROOT)
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
for p in (os.path.join(EXP5, "task3_full_depth_p1", "scripts"), os.path.join(EXP5, "task2", "scripts"), os.path.join(EXP5, "task1", "scripts")):
    sys.path.insert(0, p)
import v2_frozen as VF  # noqa: E402
import t2_common as C  # noqa: E402
import t21_build as B21  # noqa: E402
import p1_full as PF  # noqa: E402
import m_common as MC  # noqa: E402

sys.path.insert(1, VF.EXP3B if hasattr(VF, "EXP3B") else os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import t5_diagnostic as TD  # noqa: E402

W = os.path.join(REPO, "work_dirs", "exp6_v3", "shared_cache", "tsdf4")
E2W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1", "e2")
W2 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task2")
REFRESH, MOVES_INDOOR, MOVES_OUTDOOR = 10, 150, 100


def coarse_of(fu):
    """V2 coarse grid of the fusion (the shard check) from the lean structure (equal to fu.structure(); tests)."""
    s = PT.structure_lean(fu)
    dom3 = fu.dom[0][:, None, None] & fu.dom[1][None, :, None] & fu.dom[2][None, None, :]
    c = G.coarse_grid(s, dom3).cpu().numpy()
    del s, dom3
    return c


def save(p_out, blobs, meta, summ):
    os.makedirs(os.path.dirname(p_out), exist_ok=True)
    tmp = p_out + f".tmp{os.getpid()}.npz"
    np.savez(tmp, meta=json.dumps(meta), **{k: np.frombuffer(v, dtype=np.uint8) for k, v in blobs.items()})
    os.replace(tmp, p_out)
    json.dump(summ, open(p_out[:-4] + ".json", "w"), indent=1)


def build_mp3d(job, dev):
    key, tj = job["key"], job["traj"]
    p_out = os.path.join(W, "mp3d", key, f"{tj}.npz")
    if os.path.exists(p_out):
        return None
    tb = time.time()
    shard = np.load(os.path.join(E2W, "w2_d100", "mp3d", "views", key, f"{tj}.npz"))
    card = C.load_card(key)
    td = os.path.join(W2, "mp3d", "traj", key, tj)
    recs = [MC.load_service_record(os.path.join(td, "svc", f"svc_t{t}.npz")) for t in range(MOVES_INDOOR)]
    blobs, meta = {}, {"dataset": "mp3d", "scene": key, "traj": tj, "factor": PT.FACTOR, "refresh": REFRESH, "n_states": MOVES_INDOOR,
                       "k_M": float(shard["k_M"]), "blocks": []}
    nbytes = 0
    for b in range(0, MOVES_INDOOR, REFRESH):
        fr, dims, dom, box = B21.region(card, recs[b]["cam_center"])
        bi = b // REFRESH
        assert np.array_equal(np.asarray(dims), shard[f"dims_b{bi}"]), (key, tj, b)
        fu = PF.P1FusionFull(fr, dims, dom, device=dev)
        for t in range(b):
            fu.integrate_record(recs[t])
        shape = None
        for t in range(b, min(b + REFRESH, MOVES_INDOOR)):
            fu.integrate_record(recs[t])
            if not np.array_equal(coarse_of(fu), shard[f"coarse_b{bi}"][t - b]):
                raise SystemExit(f"STOP: replayed D100 fusion differs from the E2 shard coarse grid ({key}/{tj} t={t})")
            grid = PT.aggregate_fusion(fu)
            blob, shape = PT.to_blob(grid)
            blobs[f"blob_{t}"] = blob
            nbytes += len(blob)
        meta["blocks"].append({"b": b, "origin": [float(x) for x in shard[f"origin_b{bi}"]], "dims": [int(x) for x in dims],
                               "shape4": [int(x) for x in shape], "box": np.asarray(box).tolist()})
        del fu
        torch.cuda.empty_cache()
    summ = {"dataset": "mp3d", "scene": key, "traj": tj, "purpose": job["purpose"], "states": MOVES_INDOOR, "bytes": nbytes,
            "seconds": time.time() - tb, "shape4": meta["blocks"][0]["shape4"]}
    save(p_out, blobs, meta, summ)
    return summ


def build_outdoor(scene, tj, dev, geo=None):
    p_out = os.path.join(W, "outdoor", scene, f"{tj}.npz")
    if os.path.exists(p_out):
        return None
    tb = time.time()
    shard = np.load(os.path.join(E2W, "out_views_d100", scene, f"{tj}.npz"))
    fr, dims, dom, win = geo or TD.scene_geometry(scene)
    rec_dir = os.path.join(VF.DATA_V1, "trajectories", scene, tj, "records")
    files = {int(f.split("_t")[-1][:-4]): os.path.join(rec_dir, f) for f in os.listdir(rec_dir) if f.endswith(".npz")}
    fu = PF.P1FusionFull(fr, dims, dom, device=dev)
    blobs, nbytes, shape = {}, 0, None
    for t in range(MOVES_OUTDOOR):
        rec = MC.load_service_record(files[t])
        assert int(rec["t"]) == t
        fu.integrate_record(rec)
        if not np.array_equal(coarse_of(fu), shard["coarse"][t]):
            raise SystemExit(f"STOP: replayed D100 fusion differs from the E2 shard coarse grid ({scene}/{tj} t={t})")
        blob, shape = PT.to_blob(PT.aggregate_fusion(fu))
        blobs[f"blob_{t}"] = blob
        nbytes += len(blob)
    meta = {"dataset": "outdoor", "scene": scene, "traj": tj, "factor": PT.FACTOR, "n_states": MOVES_OUTDOOR, "k_M": float(fr.k_M),
            "blocks": [{"b": 0, "origin": [float(x) for x in fr.origin_z] if hasattr(fr, "origin_z") else None,
                        "dims": [int(x) for x in dims], "shape4": [int(x) for x in shape]}]}
    summ = {"dataset": "outdoor", "scene": scene, "traj": tj, "heldout": bool(shard["heldout"]), "states": MOVES_OUTDOOR,
            "bytes": nbytes, "seconds": time.time() - tb, "shape4": meta["blocks"][0]["shape4"]}
    del fu
    torch.cuda.empty_cache()
    save(p_out, blobs, meta, summ)
    return summ


def claim(tag):
    d = os.path.join(W, "claims")
    os.makedirs(d, exist_ok=True)
    try:
        fd = os.open(os.path.join(d, tag + ".claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.uname()[1]}:{os.getpid()}:{time.time()}".encode())
        os.close(fd)
        return True
    except FileExistsError:
        return False


def main():
    kind, wi, wn = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    if kind == "mp3d":
        jobs = [j for j in json.load(open(os.path.join(W2, "mp3d", "manifests", "traj_jobs.json")))["jobs"] if j["purpose"] in ("train", "valsel")]
        off = (len(jobs) * wi) // max(1, wn)
        for j in jobs[off:] + jobs[:off]:
            if os.path.exists(os.path.join(W, "mp3d", j["key"], f"{j['traj']}.npz")) or not claim(f"mp3d_{j['key']}_{j['traj']}"):
                continue
            s = build_mp3d(j, dev)
            if s:
                print(f"mp3d {j['key']}/{j['traj']} {s['states']} states {s['bytes'] / 1e6:.0f} MB {s['seconds']:.0f} s", flush=True)
    else:
        root = os.path.join(E2W, "out_views_d100")
        items = [(s, f[:-4]) for s in sorted(os.listdir(root)) for f in sorted(os.listdir(os.path.join(root, s))) if f.endswith(".npz")]
        off = (len(items) * wi) // max(1, wn)
        geo = {}
        for s, tj in items[off:] + items[:off]:
            if os.path.exists(os.path.join(W, "outdoor", s, f"{tj}.npz")) or not claim(f"outdoor_{s}_{tj}"):
                continue
            geo.setdefault(s, TD.scene_geometry(s))
            r = build_outdoor(s, tj, dev, geo[s])
            if r:
                print(f"outdoor {s}/{tj} {r['states']} states {r['bytes'] / 1e6:.0f} MB {r['seconds']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
