"""EXP5 Task 2 / T2.1a step 3 -- labels t* of the MP3D trajectories (magician env, GPU; label side, training and selection
houses only). Same renderer, rays and jitter rule as T1.1 (task1/scripts/t11_labels.Renderer, futureview.rays, imported
unchanged): first GT intersection along each of the 64 x 114 scoring rays of every legal candidate of every state
t = 0..149, in model units (t* = zbuf / dz / k_M, k_M = the level's k), inf = no hit. Jitter: seeded per
(level, trajectory, t, candidate) for training trajectories, (0, 0) for selection trajectories.
Output W2/mp3d/tstar/<level>/<traj>.npz.  Usage: python t21_labels.py <worker index> <n workers>
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402
import t21_levels as LV  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import t11_labels as TL  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

RY = TL.RY


def main():
    from pytorch3d.structures import Meshes
    wi, wn = int(sys.argv[1]), int(sys.argv[2])
    dev = torch.device("cuda")
    jobs = json.load(open(os.path.join(C.W2, "mp3d", "manifests", "traj_jobs.json")))["jobs"]
    cur, ren = None, None
    for j in jobs[wi::wn]:
        p = os.path.join(C.W2, "mp3d", "tstar", j["key"], f"{j['traj']}.npz")
        td = os.path.join(C.W2, "mp3d", "traj", j["key"], j["traj"])
        if os.path.exists(p) or not os.path.exists(os.path.join(td, "traj.json")):
            continue
        tb = time.time()
        C.assert_mesh_allowed(j["scan"], j["purpose"])
        if cur != j["scan"]:
            v, f = LV.mesh_zup(j["scan"])
            mesh = Meshes(verts=[torch.as_tensor(C.zup_m_to_rt(v), dtype=torch.float32, device=dev)],
                          faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)])
            ren, cur = TL.Renderer(mesh, dev, 0, None, 8), j["scan"]
        card = C.load_card(j["key"])
        hold = j["purpose"] == "valsel"
        T_, A_, Rs, Ts, Js = [], [], [], [], []
        for t in range(C.MOVES_INDOOR):
            z = np.load(os.path.join(td, "svc", f"svc_t{t}.npz"))
            for ai in range(len(z["legal_R"])):
                T_.append(t); A_.append(ai); Rs.append(z["legal_R"][ai]); Ts.append(z["legal_T"][ai])
                Js.append((0.0, 0.0) if hold else RY.jitter_of(j["key"], j["traj"], t, ai))
        ts = ren.tstar(Rs, Ts, Js, card["k_I"]).astype(np.float32)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + f".tmp{os.getpid()}.npz"
        np.savez(tmp, tstar=ts, t=np.asarray(T_, np.int16), a=np.asarray(A_, np.int8), R=np.asarray(Rs), T=np.asarray(Ts),
                 jitter=np.asarray(Js, np.float64), heldout=np.asarray(hold), k_M=np.asarray(card["k_I"]))
        os.replace(tmp, p)
        print(f"{j['key']}/{j['traj']}: {len(Rs)} views, hit {np.isfinite(ts).mean():.3f}, {time.time() - tb:.1f} s", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
