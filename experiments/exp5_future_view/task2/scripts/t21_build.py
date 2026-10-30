"""EXP5 Task 2 / T2.1a step 4 -- V2 training shards of the MP3D trajectories (seen2scene env, 1 GPU; planner-side
information + the label file t* of t21_labels.py).

Same per-view quantities as T1.1 (task1/scripts/t11_build.py, functions imported unchanged): coarse 8^3 grid of the P1
structure, projection statistics (cls, n_seg, t_known, d_lo, d_hi, L_u) and the label encoding (b, y, cat) of t*.
The scene frame is the indoor planning region of the closed loop (Task 2, n_win = 5): at every refresh t_r = 0, 10, ..
the region box around the camera of record t_r (t2_common.region_box_m), P1 replayed over records 0..t (exactly what
the planner sees at t, the encoder input of a state may be stale by up to t - t_r moves, never across a refresh).
Output W2/mp3d/views/<level>/<traj>.npz: per-view arrays + block index; per block b: coarse_b<b> [n_t, 4, X', Y', 32],
origin_b<b>, dims_b<b>, box_b<b>.  Usage: python t21_build.py <worker index> <n workers>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
sys.path.insert(0, os.path.join(C.EXP5, "task0", "scripts"))
import v2_frozen as VF  # noqa: E402
import mp3d_frame as F  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import m_common as MC  # noqa: E402
import m_p1  # noqa: E402
from seen2scene.futureview import geometry as G  # noqa: E402

REFRESH = 10


def region(card, cam_rt):
    import m_world as MW
    box = C.region_box_m(card, cam_rt)
    fr = F.frame_of_box(card["key"], box, card["s_rt"])
    dims = MW.grid_dims(fr)
    w = MW.official_windows(fr, MW.vae_factor())
    return fr, dims, MW.domain_1d(fr, dims, w["axis_starts"]), box


def build(job, dev):
    td = os.path.join(C.W2, "mp3d", "traj", job["key"], job["traj"])
    p_ts = os.path.join(C.W2, "mp3d", "tstar", job["key"], f"{job['traj']}.npz")
    p_out = os.path.join(C.W2, "mp3d", "views", job["key"], f"{job['traj']}.npz")
    if os.path.exists(p_out) or not os.path.exists(p_ts):
        return None
    tb = time.time()
    card = C.load_card(job["key"])
    L = np.load(p_ts)
    tstar_all, vt, va, vR, vT, vJ = L["tstar"], L["t"], L["a"], L["R"], L["T"], L["jitter"]
    recs = [MC.load_service_record(os.path.join(td, "svc", f"svc_t{t}.npz")) for t in range(C.MOVES_INDOOR)]
    Nv, R = tstar_all.shape
    arr = {k: np.zeros((Nv, R), dt) for k, dt in (("cls", np.uint8), ("nseg", np.uint8), ("b", np.int8), ("cat", np.uint8))}
    for k in ("tk", "dlo", "dhi", "lu", "y"):
        arr[k] = np.zeros((Nv, R), np.float16)
    blk = np.zeros(Nv, np.int16)
    extra, vi, k_M = {}, 0, None
    for b in range(0, C.MOVES_INDOOR, REFRESH):
        fr, dims, dom, box = region(card, recs[b]["cam_center"])
        k_M = fr.k_M
        fu = m_p1.P1Fusion(fr, dims, dom, device=dev)
        for t in range(b):
            fu.integrate_record(recs[t])
        coarse = []
        for t in range(b, min(b + REFRESH, C.MOVES_INDOOR)):
            rec = recs[t]
            assert int(rec["t"]) == t
            fu.integrate_record(rec)
            sg = VF.StateGeometry(fr, dims, dom, fu, dev)
            coarse.append(G.coarse_grid(sg.struct, sg.domain).cpu().numpy())
            n = len(rec["legal_R"])
            assert np.array_equal(vt[vi:vi + n], np.full(n, t)) and np.array_equal(va[vi:vi + n], np.arange(n)), (t, vi)
            for a in range(n):
                assert np.array_equal(vR[vi], rec["legal_R"][a]) and np.array_equal(vT[vi], rec["legal_T"][a])
                rays = VF.view_rays(sg.caster, vR[vi], vT[vi], tuple(vJ[vi]))
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
                blk[vi] = b // REFRESH
                vi += 1
            del sg
        bi = b // REFRESH
        extra.update({f"coarse_b{bi}": np.stack(coarse), f"origin_b{bi}": np.asarray(fr.origin_z, np.float64),
                      f"dims_b{bi}": np.asarray(dims), f"box_b{bi}": box})
        del fu
        torch.cuda.empty_cache()
    assert vi == Nv, (vi, Nv)
    os.makedirs(os.path.dirname(p_out), exist_ok=True)
    tmp = p_out + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, t=vt, a=va, R=vR, T=vT, jitter=vJ, heldout=L["heldout"], k_M=np.asarray(k_M), block=blk,
                        n_blocks=np.asarray(len(range(0, C.MOVES_INDOOR, REFRESH))), **arr, **extra)
    os.replace(tmp, p_out)
    os.chmod(p_out, 0o444)
    mu = arr["cls"] == G.CLS_UNKNOWN
    summ = {"key": job["key"], "traj": job["traj"], "purpose": job["purpose"], "views": int(Nv),
            "cls_frac": {n: float((arr["cls"] == c).mean()) for n, c in (("unknown", 0), ("known", 1), ("empty", 2))},
            "m_unknown_cat": {k: int((arr["cat"][mu] == v).sum()) for k, v in G.CAT.items()},
            "region_dims": [list(map(int, extra[f"dims_b{i}"])) for i in range(len(range(0, C.MOVES_INDOOR, REFRESH)))],
            "seconds": time.time() - tb, "sha256": VF.sha256_file(p_out)}
    json.dump(summ, open(p_out[:-4] + ".json", "w"), indent=1)
    return summ


def claim(j):
    """Cross-node claim (O_EXCL file with host / pid / time); a claim older than STALE_S without output is re-claimable."""
    import socket
    d = os.path.join(C.W2, "mp3d", "views_claims")
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{j['key']}__{j['traj']}.claim")
    if os.path.exists(p) and time.time() - os.path.getmtime(p) > STALE_S:
        os.remove(p)
    try:
        fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
    os.close(fd)
    return True


STALE_S = 7200


def main():
    """Workers (any number, any node) loop over all jobs in an offset order, build what has labels and is unclaimed, and
    exit when every shard exists or every remaining job is labelled and claimed by another worker."""
    wi, wn = int(sys.argv[1]), int(sys.argv[2])
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device("cuda")
    jobs = json.load(open(os.path.join(C.W2, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    off = (len(jobs) * wi) // max(1, wn)
    order = jobs[off:] + jobs[:off]
    shard = lambda j: os.path.join(C.W2, "mp3d", "views", j["key"], f"{j['traj']}.npz")  # noqa: E731
    label = lambda j: os.path.join(C.W2, "mp3d", "tstar", j["key"], f"{j['traj']}.npz")  # noqa: E731
    while True:
        todo = [j for j in order if not os.path.exists(shard(j))]
        if not todo:
            break
        did = False
        for j in todo:
            if os.path.exists(shard(j)) or not os.path.exists(label(j)) or not claim(j):
                continue
            s = build(j, dev)
            did = True
            if s:
                c = s["m_unknown_cat"]
                print(f"{j['key']}/{j['traj']}: {s['views']} views, unknown {s['cls_frac']['unknown']:.3f}, HIT {c['HIT']} NOHIT {c['NOHIT']} "
                      f"KNOWN {c['KNOWN']}, {s['seconds']:.0f} s", flush=True)
            torch.cuda.empty_cache()
        if not did:
            if all(os.path.exists(label(j)) for j in todo):
                break                      # everything left is being built by other workers
            time.sleep(60)


if __name__ == "__main__":
    main()
