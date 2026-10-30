"""EXP5 Task 0.4 -- evaluation-side labels of the indoor fixed states (magician env; GT only here).

Per state (level key, start, t) and legal candidate a of record t:
  tstar[a]  : exact first GT-mesh intersection on the 64 x 114 scoring rays (EXP5 T1.1 label renderer, jitter 0;
              model units of the level frame; inf = no hit)
  gain[a]   : true one-step gain = 100 x (C_obs after executing a - C_obs(t)), level-scale coverage; the candidate's
              10 % sample uses the derived seed of step t + 1 (as if a were executed next)
C_obs(t) is reproduced by replaying the trajectory's own sample points 0..t (asserted equal to traj.json).
Outputs W/fixed_state/labels/<state>.npz + .json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mp3d_sim as S  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(S.REPO, "experiments", "exp5_future_view", "task1", "scripts"))
OUT = os.path.join(S.W, "fixed_state", "labels")


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--states", default=os.path.join(S.W, "manifests", "indoor_states.json"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    import t11_labels as TL
    from pytorch3d.structures import Meshes
    dev = torch.device("cuda")
    sts = json.load(open(a.states))["states"]
    pi, pn = (int(x) for x in a.part.split("/"))
    os.makedirs(a.out, exist_ok=True)
    cache = {}
    for st in sts[pi::pn]:
        if os.path.exists(os.path.join(a.out, f"{st['id']}.json")):
            continue
        scan, lv = st["house"], int(st["level_key"].split("_L")[1])
        if st["level_key"] not in cache:
            cache.clear()
            L = S.Level(scan, lv)
            L.build_graph()
            v, f = S.mesh_zup(scan)
            vrt = S.zup_m_to_rt(v)
            sen = S.Sensor(vrt, f, dev)
            mesh = Meshes(verts=[torch.as_tensor(vrt, dtype=torch.float32, device=dev)],
                          faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)])
            cache[st["level_key"]] = (L, sen, TL.Renderer(mesh, dev))
        L, sen, ren = cache[st["level_key"]]
        td = os.path.join(S.W, "preprocessing", "traj", st["traj"])
        tr = json.load(open(os.path.join(td, "traj.json")))
        cov = S.Coverage(L, sen)
        for i in range(st["t"] + 1):
            c = cov.update(np.load(os.path.join(td, f"svc_t{i}.npz"))["sample_points"])
        assert abs(c - tr["rows"][st["t"]]["cov"]) < 1e-12, (st["id"], c, tr["rows"][st["t"]]["cov"])
        rec = S.MC.load_service_record(os.path.join(td, f"svc_t{st['t']}.npz"))
        base = cov.covered.copy()
        gains = []
        for j in range(len(rec["legal_R"])):
            R = torch.tensor(rec["legal_R"][j][None], dtype=torch.float32, device=dev)
            T = torch.tensor(rec["legal_T"][j][None], dtype=torch.float32, device=dev)
            fr = sen.render(R, T)
            pts, _, _, _, _ = sen.sample(fr, S.derived_seed(st["level_key"], st["start"], st["t"] + 1, "sample"))
            cov.covered = base.copy()
            gains.append(100.0 * (cov.update(pts.cpu().numpy()) - c))
        cov.covered = base
        k = L.k                                   # level k = region k (height-based)
        n = len(rec["legal_R"])
        tstar = ren.tstar(rec["legal_R"], rec["legal_T"], np.zeros((n, 2)), k).astype(np.float32)
        np.savez_compressed(os.path.join(a.out, f"{st['id']}.npz"), tstar=tstar, gain=np.asarray(gains),
                            legal_idx=rec["legal_idx"])
        json.dump({"state": st, "cov_t": c, "gain": gains, "k_rt": k}, open(os.path.join(a.out, f"{st['id']}.json"), "w"),
                  indent=1, default=float)
        print(st["id"], "gains", [round(g, 2) for g in gains], flush=True)


if __name__ == "__main__":
    main()
