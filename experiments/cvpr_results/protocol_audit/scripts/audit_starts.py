"""CVPR protocol audit -- initial camera poses (read-only; magician env for LMDB access; CPU).
For each of the 15 Macarons++ scenes and start k = 0..4:
  * settings.json camera.start_positions[k] (5-D grid index [l, w, h, elev_bin, azim_bin]) = what the official tester
    (macarons/testers/magician_planning.py:834-835, official e697009 and fork 9521d69) and our frozen EpisodeSim
    (exp3b m_t6_sim.py:69-71) pass to setup_test_camera;
  * camera centre / elevation / azimuth of that index from the pose table built with MAGICIAN's own camera code
    (work_dirs/exp6_v3/H/pose_tables/<scene>.npz; elev = -90 + 180 (1 + i) / (n_elev + 1), azim = 360 i / n_azim);
  * the upstream MAGICIAN loop's first pose in our LMDB reproduction (X_cam_history[0], V_cam_history[0]);
  * the executed pose key at t = 0 of EVERY canonical CVPR Table 1 episode of every model (must equal start_positions[k]);
  * uniqueness of the five starts in the scene, and whether the start was used as the first pose of one of our training
    trajectories (data/macaronspp_belief/v1/trajectories/<scene>/*/records/*_t0.npz pose_idx).
Outputs protocol_audit/INITIAL_POSES_COMPARISON.csv and results/initial_poses_summary.json."""
import csv
import glob
import json
import os
import pickle

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy"
S2S = f"{R}/seen2scene"
DATA = f"{R}/MAGICIAN/data/Macarons++"
LMDB = f"{R}/MAGICIAN/results/scene_exploration/magician_lmdb"
EP = f"{S2S}/work_dirs/cvpr_results/table1/episodes"
PT = f"{S2S}/work_dirs/exp6_v3/H/pose_tables"
TRAJ = f"{S2S}/data/macaronspp_belief/v1/trajectories"
OUT = f"{S2S}/experiments/cvpr_results/protocol_audit"
SPLIT = {"train": ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"], "dev": ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"],
         "test": ["alhambra", "bridge", "neuschwanstein"], "other": ["sofia_church"]}
PAPER_ORDER = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein", "eiffel",
               "bridge", "sofia_church", "barts", "sestino_museum"]


def lmdb_first_poses():
    try:
        import lmdb
    except ImportError:
        return None
    env = lmdb.open(LMDB, readonly=True, lock=False)
    out = {}
    with env.begin() as txn:
        for key, val in txn.cursor():
            k = key.decode()
            if "/point_cloud" in k or k.count("/") != 1:
                continue
            d = pickle.loads(val)
            if "X_cam_history" in d:
                out[k] = (np.asarray(d["X_cam_history"], float)[0], np.asarray(d["V_cam_history"], float)[0], len(d["coverage"]))
    env.close()
    return out


FV = f"{S2S}/experiments/exp5_future_view/task1/frozen_values.json"     # frozen S-c split (train6_minus_Sc = actual training set)


def training_starts(scene):
    """(first poses of ALL 180-set trajectories of the scene, first poses of the trajectories in the TRAINING split
    train6_minus_Sc only)."""
    tr = set(json.load(open(FV))["split"]["scenes"][scene]["train_trajs"])
    s_all, s_train = set(), set()
    for d in glob.glob(os.path.join(TRAJ, scene, "*")):
        for f in glob.glob(os.path.join(d, "records", "*_t0.npz")):
            try:
                key = tuple(int(x) for x in np.load(f)["pose_idx"].reshape(-1)[:5])
            except Exception:
                continue
            s_all.add(key)
            if os.path.basename(d) in tr:
                s_train.add(key)
    return s_all, s_train


def main():
    os.makedirs(os.path.join(OUT, "results"), exist_ok=True)
    L = lmdb_first_poses()
    models = sorted(d for d in os.listdir(EP) if os.path.isdir(os.path.join(EP, d)))
    rows, summ = [], {"lmdb_available": L is not None, "models": models, "scenes": {}}
    for s in PAPER_ORDER:
        st = json.load(open(os.path.join(DATA, s, "settings.json")))
        starts = [list(map(int, x)) for x in st["camera"]["start_positions"]]
        grid = {k: st["camera"][k] for k in ("pose_n_elev", "pose_n_azim") if k in st["camera"]}
        sp = next(k for k, v in SPLIT.items() if s in v)
        pt = np.load(os.path.join(PT, f"{s}.npz")) if os.path.exists(os.path.join(PT, f"{s}.npz")) else None
        lut = {tuple(int(x) for x in r): j for j, r in enumerate(pt["idx"])} if pt is not None else {}
        g = json.loads(str(pt["grid"])) if pt is not None else {"pose_n_elev": grid.get("pose_n_elev"), "pose_n_azim": grid.get("pose_n_azim")}
        tr_all, tr = training_starts(s) if sp == "train" else (set(), set())
        summ["scenes"][s] = {"split": sp, "n_starts": len(starts), "unique": len({tuple(x) for x in starts}) == len(starts)}
        for k, idx in enumerate(starts):
            key = str(idx)
            row = {"scene": s, "split": sp, "start": k, "settings_start_idx": key}
            if pt is not None and tuple(idx) in lut:
                c = pt["C"][lut[tuple(idx)]]
                row.update({"x": round(float(c[0]), 4), "y": round(float(c[1]), 4), "z": round(float(c[2]), 4),
                            "elev_deg": round(-90 + 180 * (1 + idx[3]) / (g["pose_n_elev"] + 1), 4), "azim_deg": round(360 * idx[4] / g["pose_n_azim"], 4)})
            else:
                row.update({"x": "", "y": "", "z": "", "elev_deg": "", "azim_deg": ""})
            lk = f"{s}/{k}"
            if L is not None and lk in L:
                X0, V0, n = L[lk]
                row.update({"lmdb_x0": round(float(X0[0]), 4), "lmdb_y0": round(float(X0[1]), 4), "lmdb_z0": round(float(X0[2]), 4),
                            "lmdb_elev0": round(float(V0[0]), 4), "lmdb_azim0": round(float(V0[1]), 4), "lmdb_coverage_len": n})
                if row["x"] != "":
                    dx = float(np.max(np.abs(X0 - np.asarray([row["x"], row["y"], row["z"]]))))
                    da = abs(((float(V0[1]) - row["azim_deg"]) + 180) % 360 - 180)
                    row["lmdb_max_abs_dX"] = round(dx, 6)
                    row["lmdb_abs_d_elev"] = round(abs(float(V0[0]) - row["elev_deg"]), 6)
                    row["lmdb_abs_d_azim"] = round(da, 6)
                    row["lmdb_match"] = dx < 1e-3 and row["lmdb_abs_d_elev"] < 1e-3 and da < 1e-3
            else:
                row.update({"lmdb_x0": "", "lmdb_y0": "", "lmdb_z0": "", "lmdb_elev0": "", "lmdb_azim0": "", "lmdb_coverage_len": "", "lmdb_max_abs_dX": "",
                            "lmdb_abs_d_elev": "", "lmdb_abs_d_azim": "", "lmdb_match": "no LMDB entry"})
            have, ok, bad = 0, 0, []
            for m in models:
                p = os.path.join(EP, m, f"{s}_s{k}.json")
                if not os.path.exists(p):
                    continue
                have += 1
                e = json.load(open(p))
                k0 = e["executed_keys"][0]
                if json.loads(k0) == idx and int(e["start"]) == k:
                    ok += 1
                else:
                    bad.append(f"{m}:{k0}")
            row.update({"table1_models_with_episode": have, "table1_models_t0_match": ok, "table1_mismatches": ";".join(bad),
                        "start_unique_in_scene": summ["scenes"][s]["unique"],
                        "first_pose_of_any_of_our_180_trajectories": (tuple(idx) in tr_all) if sp == "train" else "n/a (not a training scene)",
                        "first_pose_of_a_TRAINING_split_trajectory": (tuple(idx) in tr) if sp == "train" else "n/a (not a training scene)"})
            rows.append(row)
    fields = list(rows[0].keys())
    for r in rows:
        for f in r:
            if f not in fields:
                fields.append(f)
    with open(os.path.join(OUT, "INITIAL_POSES_COMPARISON.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    tot = [r for r in rows if r["scene"] != "sofia_church"]
    summ["all14"] = {"start_rows": len(tot), "lmdb_match": sum(1 for r in tot if r.get("lmdb_match") is True),
                     "episodes_checked": sum(r["table1_models_with_episode"] for r in tot), "episodes_t0_match": sum(r["table1_models_t0_match"] for r in tot),
                     "all_starts_unique": all(v["unique"] for v in summ["scenes"].values()),
                     "train6_starts_first_pose_any_trajectory": {s: [r["start"] for r in rows if r["scene"] == s and r["first_pose_of_any_of_our_180_trajectories"] is True] for s in SPLIT["train"]},
                     "train6_starts_first_pose_training_split": {s: [r["start"] for r in rows if r["scene"] == s and r["first_pose_of_a_TRAINING_split_trajectory"] is True] for s in SPLIT["train"]}}
    json.dump(summ, open(os.path.join(OUT, "results", "initial_poses_summary.json"), "w"), indent=1)
    print(json.dumps(summ["all14"], indent=1))


if __name__ == "__main__":
    main()
