"""EXP6 V3 / H -- camera pose tables of the official settings grid (magician env, 1 GPU; run once, before any H run).
For every runnable scene: the frozen simulator EpisodeSim(scene, start 0) builds MAGICIAN's camera exactly as in the closed
loop; for every pose index of camera.pose_space (built from settings.json only: camera box, pose_l / w / h, n_elev,
n_azim) -> X, V = pose[:3], pose[3:] -> R, T = macarons get_camera_RT(X, V) (the function of legal_neighbours) and the
centre X. Saved to work_dirs/exp6_v3/H/pose_tables/<scene>.npz (idx [N, 5], R [N, 3, 3], T [N, 3], C [N, 3], grid sizes).
No mesh / GT / occupied-pose data enters the table. Verification: for every historical V3 service record of the scene in
the CVPR Table 1 closed-loop roots, legal_idx -> table (R, T, C) must equal legal_R / legal_T / legal_center bitwise.
Usage: python build_pose_tables.py [--scenes ...]"""
import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_t6_sim as T6S  # noqa: E402
import m_magician_replay as RP  # noqa: E402
import m_common as MC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = os.path.join(REPO, "work_dirs", "exp6_v3", "H", "pose_tables")
SCENES = ["alhambra", "bannerman", "barts", "bridge", "colosseum", "dunnottar", "eiffel", "fushimi", "liberty", "neuschwanstein", "pantheon", "pisa", "redeemer", "sestino_museum"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="*", default=SCENES)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    summary = {}
    for s in a.scenes:
        tmp = tempfile.mkdtemp(dir=os.environ.get("CLAUDE_JOB_DIR", "/tmp"))
        try:
            sim = T6S.EpisodeSim(s, 0, tmp)
            cam = sim.camera
            keys, poses = RP.pose_lookup(cam)
            idx = np.asarray([[int(c) for c in k[1:-1].split(",")] for k in keys], np.int64)
            Rs, Ts, Cs = [], [], []
            for k in keys:
                pose = cam.pose_space[k]
                X, V = pose[:3].view(1, 3), pose[3:].view(1, 2)
                R, T = RP.get_camera_RT(X, V)
                Rs.append(R[0].cpu().numpy().astype(np.float64)); Ts.append(T[0].cpu().numpy().astype(np.float64)); Cs.append(X[0].cpu().numpy().astype(np.float64))
            grid = {"pose_l": int(cam.pose_l), "pose_w": int(cam.pose_w), "pose_h": int(cam.pose_h), "pose_n_elev": int(cam.pose_n_elev), "pose_n_azim": int(cam.pose_n_azim)}
            Rs, Ts, Cs = np.stack(Rs), np.stack(Ts), np.stack(Cs)
            np.savez(os.path.join(OUT, f"{s}.npz"), idx=idx, R=Rs, T=Ts, C=Cs, grid=json.dumps(grid))
            # verification against historical service records (V3 closed loop of Table 1 / exp6_v3)
            lut = {tuple(i): j for j, i in enumerate(idx.tolist())}
            n_rec = n_rows = n_bad = 0
            roots = glob.glob(os.path.join(REPO, "work_dirs", "exp6_v3", "eval", "closed_loop", "outdoor", "T1", "V2M", "svc", f"*{s}_s*", "svc_*.npz")) + \
                glob.glob(os.path.join(REPO, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor", "V3", "V2M", "svc", f"*{s}_s*", "svc_*.npz"))
            for p in sorted(roots)[:400]:
                rec = MC.load_service_record(p)
                n_rec += 1
                for r in range(len(rec["legal_idx"])):
                    j = lut[tuple(rec["legal_idx"][r].tolist())]
                    n_rows += 1
                    ok = np.array_equal(Rs[j], rec["legal_R"][r]) and np.array_equal(Ts[j], rec["legal_T"][r]) and np.array_equal(Cs[j], rec["legal_center"][r])
                    n_bad += not ok
            summary[s] = {"n_poses": len(keys), "grid": grid, "verified_records": n_rec, "verified_rows": n_rows, "mismatch_rows": n_bad}
            print(s, summary[s], flush=True)
            del sim
            torch.cuda.empty_cache()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    json.dump(summary, open(os.path.join(OUT, "verification.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
