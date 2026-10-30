"""EXP5 Task 3 / E2 pilot -- MP3D development fixed states on the VALSEL trajectories (magician env, 1 GPU; evaluation
side: GT mesh of the 6 val selection houses, never the 5 evaluation floors or the official test houses).

States: the 9 VALSEL trajectories of Task 2 (t21_traj; service records + traj.json) at t in {0, 5, 20, 25, 50, 55, 100,
105, 140, 145}; candidate set = the Task 2 V2 choice rule (prefer unvisited, ab_common.candidate_set).
Oracle per legal candidate (the t21_traj greedy rule, the same as the C_obs evaluator's eps): gain = |GT points (the
100k area-weighted points of the level box, t21_traj.gt_points) newly within eps = EPS_MODEL * k_I of the candidate's
frame-(t+1) 10 % sample| / 100k, with the coverage replayed over the stored samples of records 0..t; the frame-(t+1)
sample uses the trajectory's own seed (derived_seed(key, start_pano, t + 1, "sample")) -> check: the executed candidate's
sample equals record t + 1 bitwise (STOP otherwise). t* of the 64 x 114 scoring rays (T1.1 renderer, jitter 0, k_I).
Output e2/valsel/manifests/states_valsel.json, e2/valsel/states/<sid>/oracle.npz. Usage: python e2_valsel_oracle.py
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
sys.path.insert(0, os.path.join(EP.EXP5, "ablation_study", "common"))
import t2_common as C  # noqa: E402
import t21_traj as TJ  # noqa: E402
import t21_levels as LV  # noqa: E402
import ab_common as A  # noqa: E402

import numpy as np  # noqa: E402

S = TJ.S
OUT = os.path.join(EP.W, "valsel")
TIMES = A.T_STATES["indoor"]


def main():
    import torch
    from scipy.spatial import cKDTree
    from pytorch3d.structures import Meshes
    import t11_labels as TL
    dev = torch.device("cuda")
    jobs = [j for j in json.load(open(os.path.join(EP.T2W, "mp3d", "manifests", "traj_jobs.json")))["jobs"] if j["purpose"] == "valsel"]
    states = []
    for job in jobs:
        tb = time.time()
        C.assert_mesh_allowed(job["scan"], job["purpose"])
        td = os.path.join(EP.T2W, "mp3d", "traj", job["key"], job["traj"])
        traj = json.load(open(os.path.join(td, "traj.json")))
        card = C.load_card(job["key"])
        v, f = LV.mesh_zup(job["scan"])
        sen = S.Sensor(S.zup_m_to_rt(v), f, dev)
        tree = cKDTree(TJ.gt_points(job["scan"], card, v, f))
        eps = S.EPS_MODEL * card["k_I"]
        covered = np.zeros(tree.n, bool)
        mesh = Meshes(verts=[torch.as_tensor(C.zup_m_to_rt(v), dtype=torch.float32, device=dev)],
                      faces=[torch.as_tensor(f, dtype=torch.int64, device=dev)])
        ren = TL.Renderer(mesh, dev, 0, None, 8)
        tag = f"{job['key']}_{job['traj']}"
        for t in range(max(TIMES) + 1):
            p = os.path.join(td, "svc", f"svc_t{t}.npz")
            rec = np.load(p)
            assert int(rec["t"]) == t
            covered |= TJ.near(tree, rec["sample_points"], eps)
            if t not in TIMES:
                continue
            hist = rec["executed_pose_idx_history"]
            sid = A.state_id("indoor", job["key"], job["start_pano"], hist)
            seed = S.derived_seed(job["key"], job["start_pano"], t + 1, "sample")
            gains, pts_all, off = [], [], [0]
            for R, T in zip(rec["legal_R"], rec["legal_T"]):
                fr = sen.render(torch.tensor(R[None], dtype=torch.float32, device=dev), torch.tensor(T[None], dtype=torch.float32, device=dev))
                pts = sen.sample(fr, seed)[0].cpu().numpy()
                gains.append(float((TJ.near(tree, pts, eps) & ~covered).sum()) / tree.n)
                pts_all.append(pts.astype(np.float32))
                off.append(off[-1] + len(pts))
            ex = traj["steps"][t]["pick"]
            nxt = np.load(os.path.join(td, "svc", f"svc_t{t + 1}.npz"))["sample_points"]
            if ex is None or not np.array_equal(pts_all[ex], nxt):
                raise SystemExit(f"STOP: executed candidate sample differs from record t+1 ({tag} t={t})")
            tstar = ren.tstar(list(rec["legal_R"]), list(rec["legal_T"]), [(0.0, 0.0)] * len(rec["legal_R"]), card["k_I"]).astype(np.float32)
            od = os.path.join(OUT, "states", "indoor", sid)
            os.makedirs(od, exist_ok=True)
            np.savez_compressed(os.path.join(od, "oracle.tmp.npz"), gain=np.asarray(gains), cov_t=np.asarray(covered.mean()),
                                remaining=np.asarray(1.0 - covered.mean()), tstar=tstar, k_M=np.asarray(card["k_I"]), eps=np.asarray(eps),
                                cand_pts=np.concatenate(pts_all), cand_off=np.asarray(off), legal_idx=rec["legal_idx"])
            os.replace(os.path.join(od, "oracle.tmp.npz"), os.path.join(od, "oracle.npz"))
            r = {k: rec[k] for k in ("legal_idx", "executed_pose_idx_history")}
            states.append({"sid": sid, "dataset": "indoor", "scene": job["key"], "start": job["start_pano"], "tag": tag, "split": "mp3d_valsel",
                           "t": t, "scan": job["scan"], "level": job["level"], "traj": job["traj"], "policy": job["policy"],
                           "cand": A.candidate_set("indoor", r), "n_legal": int(len(rec["legal_idx"])), "legal_idx": rec["legal_idx"].tolist(),
                           "sources": [{"source": "VALSEL", "svc": os.path.relpath(p, EP.REPO), "executed_pick": ex}]})
            print(tag, t, "max gain %.4f cov %.3f" % (max(gains), covered.mean()), flush=True)
        print(tag, "%.0fs" % (time.time() - tb), flush=True)
        torch.cuda.empty_cache()
    os.makedirs(os.path.join(OUT, "manifests"), exist_ok=True)
    json.dump({"dataset": "valsel", "t_states": list(TIMES), "states": states}, open(os.path.join(OUT, "manifests", "states_valsel.json"), "w"), indent=1)
    print("states", len(states))


if __name__ == "__main__":
    main()
