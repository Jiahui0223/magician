"""Table-1 dataset audit: assemble SCENE_REGISTRY.yaml from the audit outputs (read-only inputs).

Inputs (all in experiments/cvpr_results/table1/ unless noted):
  lmdb_summary.json           scripts/audit_lmdb.py      (magician env)
  load_test_results.json      scripts/load_test.py       (magician env, GPU)
  planner_tables.json         scripts/planner_tables.py  (seen2scene env, CPU)
  closed_loop_inventory.json  scripts/audit_closed_loops.py
  sha256_check.txt            sha256sum -c of experiments/exp4_multiworld_belief/task0/configs/SHA256SUMS
  the scene folders (settings.json, .mtl, file listing) and the frozen split / training configs (evidence strings).
Paper numbers: MAGICIAN/Magician.pdf p.12, Tables 6 (AUC) and 7 (final coverage), extracted with `pdftotext -layout
-f 12 -l 12` and transcribed below.
Usage: env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/seen2scene/bin/python build_registry.py
"""
import json
import os
import re

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"
MAN = "experiments/exp4_multiworld_belief/task0/configs/dataset_manifest_v1.json"
PROT = "experiments/exp4_multiworld_belief/task0/dataset_protocol.md"
CM = "MAGICIAN_clean_9521d69/configs/macarons/macarons_default_training_config.json"
CT = "MAGICIAN_clean_9521d69/configs/test/test_in_default_scenes_config.json"

# official name (paper Tables 6/7 order) -> local id
OFFICIAL = [("Dunnottar Castle", "dunnottar"), ("Colosseum", "colosseum"), ("Bannerman Castle", "bannerman"),
            ("Pantheon", "pantheon"), ("Christ the Redeemer", "redeemer"), ("Statue of Liberty", "liberty"),
            ("Pisa Cathedral", "pisa"), ("Fushimi Castle", "fushimi"), ("Alhambra Palace", "alhambra"),
            ("Neuschwanstein Castle", "neuschwanstein"), ("Eiffel Tower", "eiffel"), ("Manhattan Bridge", "bridge"),
            ("St. Sofia Church", "sofia_church"), ("Barts", "barts"), ("Sestino Museum", "sestino_museum")]
# Magician.pdf p.12 Table 6 (AUC) / Table 7 (final coverage): (Rand. Walk, SCONE, MACARONS, FisherRF, MAGICIAN)
PAPER_AUC = {"dunnottar": (0.149, 0.366, 0.618, 0.500, 0.745), "colosseum": (0.219, 0.589, 0.656, 0.551, 0.704),
             "bannerman": (0.192, 0.559, 0.575, 0.595, 0.761), "pantheon": (0.198, 0.465, 0.601, 0.270, 0.644),
             "redeemer": (0.439, 0.772, 0.859, 0.727, 0.793), "liberty": (0.323, 0.632, 0.711, 0.553, 0.797),
             "pisa": (0.290, 0.486, 0.678, 0.486, 0.723), "fushimi": (0.279, 0.689, 0.718, 0.565, 0.766),
             "alhambra": (0.126, 0.369, 0.567, 0.462, 0.631), "neuschwanstein": (0.184, 0.325, 0.452, 0.375, 0.608),
             "eiffel": (0.333, 0.683, 0.709, 0.616, 0.754), "bridge": (0.258, 0.632, 0.750, 0.637, 0.705),
             "sofia_church": (0.280, 0.532, 0.621, 0.608, 0.710), "barts": (0.214, 0.551, 0.660, 0.673, 0.831),
             "sestino_museum": (0.132, 0.367, 0.537, 0.571, 0.637)}
PAPER_FINAL = {"dunnottar": (0.225, 0.527, 0.820, 0.809, 0.975), "colosseum": (0.272, 0.755, 0.794, 0.757, 0.872),
               "bannerman": (0.240, 0.722, 0.834, 0.801, 0.917), "pantheon": (0.309, 0.610, 0.796, 0.444, 0.842),
               "redeemer": (0.581, 0.924, 0.967, 0.876, 0.973), "liberty": (0.443, 0.819, 0.909, 0.819, 0.947),
               "pisa": (0.353, 0.566, 0.865, 0.776, 0.941), "fushimi": (0.449, 0.844, 0.853, 0.814, 0.931),
               "alhambra": (0.162, 0.473, 0.775, 0.615, 0.852), "neuschwanstein": (0.223, 0.444, 0.551, 0.582, 0.848),
               "eiffel": (0.541, 0.856, 0.915, 0.827, 0.923), "bridge": (0.356, 0.781, 0.924, 0.877, 0.955),
               "sofia_church": (0.331, 0.619, 0.795, 0.865, 0.891), "barts": (0.240, 0.677, 0.768, 0.878, 0.996),
               "sestino_museum": (0.141, 0.430, 0.713, 0.842, 0.924)}
PAPER_AVG = {"auc": (0.241, 0.534, 0.647, 0.546, 0.721), "final": (0.324, 0.670, 0.819, 0.786, 0.919)}
METHODS = ("random_walk", "scone", "macarons", "fisherrf", "magician")

SPLIT = {"bannerman": "train", "dunnottar": "train", "fushimi": "train", "liberty": "train", "pisa": "train",
         "redeemer": "train", "barts": "dev", "eiffel": "dev", "pantheon": "dev", "colosseum": "dev",
         "sestino_museum": "dev", "alhambra": "test", "neuschwanstein": "test", "bridge": "test",
         "sofia_church": "other_unassigned"}
MAN_SET_LINE = {"alhambra": 47, "bannerman": 64, "barts": 121, "bridge": 162, "colosseum": 179, "dunnottar": 276,
                "eiffel": 556, "fushimi": 597, "liberty": 654, "neuschwanstein": 696, "pantheon": 713, "pisa": 780,
                "redeemer": 867, "sestino_museum": 914, "sofia_church": 1020}
MAN_LIST_LINES = {"train": "8-15", "dev": "16-22", "test": "23-27", "reserved": "28-30"}
PROT_LINE = {"train": 32, "dev": 33, "test": 34, "reserved": 35}
DEV3 = ["barts", "pantheon", "colosseum"]
TEST3 = ["alhambra", "bridge", "neuschwanstein"]
MACARONS_ROLE = {**{s: "train" for s in ("dunnottar", "pisa", "bannerman", "redeemer", "pantheon", "fushimi", "liberty",
                                        "colosseum")},
                 "alhambra": "val", "neuschwanstein": "val", "bridge": "test", "eiffel": "test",
                 "barts": "not in MACARONS data", "sestino_museum": "not in MACARONS data",
                 "sofia_church": "not in MACARONS data"}
MACARONS_ROLE_LINE = {"train": f"{CM}:18-27", "val": f"{CM}:28-31", "test": f"{CM}:32-35"}
INDOOR = {"barts", "sestino_museum", "sofia_church"}
TRAIN6_MODELS = {
    "V2_Original": "task1/configs/train_main.json:5-6 (data_v1.json, train6_minus_Sc) -> task1/scripts/t12_train.py:170 "
                   "VD.load_split -> v2_data.py:176-191; Train-6 from data_v1.json:5",
    "V2_Mix": "task2/configs/train_mix_v1.json:8-9 (MP3D train houses + train6_minus_Sc) -> t22_train_mix.py:68",
    "E2_D100": "task3_full_depth_p1/e2_pilot/configs/e2_d100_v1.json:8-9 (MP3D train + train6_minus_Sc, D100 shards); "
               "e2_train.py:4-7 runs the frozen t22_train_mix.py",
    "V3_T0_T1": "seen2scene/v3/trainer.py:117-118 (MP3D train/valsel + outdoor train/sc) -> seen2scene/v3/data.py:43-58 "
                "(outdoor = scene dirs of e2/out_views_d100 = Train-6; heldout flag = frozen S-c, checked)",
    "V3_1": "exp6_v3.1/configs/outdoor_train.json:6-9 (outdoor only, 0 MP3D) -> exp6_v3.1/scripts/trainer_outdoor.py:54",
    "V4": "exp6_v4/configs/outdoor_train.json:7-11 (train6_minus_Sc, D100, 0 MP3D) -> exp6_v4/scripts/train_v4.py:85",
    "V4C": "exp6_v5/configs/train_v4_continue.json:2-5 (train6_minus_Sc, 0 MP3D)",
    "V5": "exp6_v5/configs/train_v5_outdoor.json:2-5 (train6_minus_Sc, 0 MP3D) -> exp6_v5/scripts/train_v5.py:198",
    "V5_2_pilot": "exp6_v5_2/configs/train_v52.json:5-8 (train6_minus_Sc, 0 MP3D)",
    "V5_2M": "exp6_v5_2_scratch70k/configs/train_v52m.json:5-8 (train6_minus_Sc, 0 MP3D) -> train_v52m.py:180",
}
HELDOUT_OFFICIAL = {"bannerman": [0], "dunnottar": [3], "fushimi": [0], "liberty": [2], "pisa": [3], "redeemer": [0]}


class FlowDumper(yaml.SafeDumper):
    """Lists of scalars (and lists of such lists) in flow style, everything else block style."""


def _repr_list(dumper, data):
    flat = all(not isinstance(x, (dict, list)) for x in data)
    nested = all(isinstance(x, list) and all(not isinstance(y, (dict, list)) for y in x) for x in data) and data
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=bool(flat or nested))


FlowDumper.add_representer(list, _repr_list)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def r4(x):
    return None if x is None else float(round(float(x), 6))


def assets(s):
    d = os.path.join(DATA, s)
    files = sorted(os.listdir(d))
    objs = [f for f in files if f.endswith(".obj")]
    mtls = [f for f in files if f.endswith(".mtl")]
    kd = []
    for m in mtls:
        for line in open(os.path.join(d, m)):
            if line.startswith("map_Kd "):
                kd.append(line.split()[1])
    return {"obj": [{"name": f, "bytes": os.path.getsize(os.path.join(d, f))} for f in objs],
            "mtl": mtls, "textures_map_Kd": kd, "n_map_Kd": len(kd),
            "textures_present": all(os.path.exists(os.path.join(d, t)) for t in kd),
            "settings_json": os.path.exists(os.path.join(d, "settings.json")),
            "occupied_pose_pt": os.path.exists(os.path.join(d, "occupied_pose.pt")),
            "derived_dirs": [f for f in files if os.path.isdir(os.path.join(d, f))]}


def main():
    lm, lt, pt, cl = (load(x) for x in ("lmdb_summary.json", "load_test_results.json", "planner_tables.json",
                                        "closed_loop_inventory.json"))
    sha_txt = open(os.path.join(OUT, "sha256_check.txt")).read()
    diag = load("sofia_diag_result.json")
    lt = lt["results"] if lt else {}
    scenes = []
    for name, s in OFFICIAL:
        st = json.load(open(os.path.join(DATA, s, "settings.json")))
        sc, ca = st["scene"], st["camera"]
        e = {"official_name": name, "local_id": s, "type": "indoor" if s in INDOOR else "outdoor"}
        sp = SPLIT[s]
        e["split"] = sp
        hist = "reserved" if s == "sofia_church" else sp
        ev = [f"{MAN}:{MAN_SET_LINE[s]} (\"set\": \"{hist}\")", f"{MAN}:{MAN_LIST_LINES[hist]} (sets.{hist})",
              f"{PROT}:{PROT_LINE[hist]}"]
        if sp == "train":
            ev.append("experiments/exp5_future_view/task1/configs/data_v1.json:5 (scenes_train6)")
        if s in DEV3 + TEST3:
            ev.append("experiments/exp5_future_view/task2/configs/eval_outdoor_v1.json:" +
                      ("9-13 (scenes.test)" if s in TEST3 else "14-18 (scenes.dev)"))
        if s in ("barts", "eiffel", "pantheon", "colosseum", "sestino_museum"):
            ev.append("experiments/exp3b_macaronspp_transfer/m_common.py:34 (EXP3b GATE = Dev-5)")
        e["split_detail"] = {"dev": "Dev-5" + (" (Dev-3 closed-loop subset)" if s in DEV3 else
                                                                   " (not in the Dev-3 closed-loop subset)"),
                             "test": "Test-3", "train": "Train-6 (official starts partly in training data, see below)",
                             "other_unassigned": "EXP4 manifest set 'reserved' = not used"}[sp]
        e["split_evidence"] = ev
        role = MACARONS_ROLE[s]
        e["macarons_weights_role"] = {"role": role, "evidence": MACARONS_ROLE_LINE.get(role, f"{CM}:17-35 (absent)"),
                                      "note": "MAGICIAN uses MACARONS trained_macarons.pth (no own weights)"}
        e["in_magician_test_list"] = {"value": s != "sofia_church", "evidence": f"{CT}:4-19"}
        if sp == "train":
            e["used_to_train"] = {m: "yes (train6_minus_Sc)" for m in TRAIN6_MODELS}
            e["official_starts_in_training_trajectories"] = {
                "official_starts_with_trajectories": [0, 1, 2, 3], "official_start_never_used_as_start": 4,
                "held_out_by_S-c": HELDOUT_OFFICIAL[s],
                "evidence": "data/macaronspp_belief/v1/trajectories/<scene>/*/records/*_t0.npz pose_idx vs settings "
                            "start_positions (checked 2026-11-03); task1/frozen_values.json split.heldout_starts; "
                            "exp4 task0/trajectory_generation.md:36,76,78"}
        else:
            e["used_to_train"] = {m: "no" for m in TRAIN6_MODELS}
        a = assets(s)
        a["sha256_matches_frozen_manifest"] = all(f"{s}/{f}: OK" in sha_txt for f in
                                                  [x["name"] for x in a["obj"]] + a["mtl"] + ["settings.json"])
        e["assets"] = a
        e["settings"] = {
            "scene_scale_factor": 10.0,
            "scene_bbox_settings": {"x_min": sc["x_min"], "x_max": sc["x_max"]},
            "scene_bbox_runtime_min": [r4(10 * x) for x in sc["x_min"]],
            "scene_bbox_runtime_max": [r4(10 * x) for x in sc["x_max"]],
            "scene_grid_lwh": [sc["grid_l"], sc["grid_w"], sc["grid_h"]],
            "camera_bbox_settings": {"x_min": ca["x_min"], "x_max": ca["x_max"]},
            "camera_grid_l_w_h_theta_azim": [ca["pose_l"], ca["pose_w"], ca["pose_h"], ca["pose_n_theta"],
                                             ca["pose_n_azim"]],
            "n_poses": int(np.prod([ca["pose_l"], ca["pose_w"], ca["pose_h"], ca["pose_n_theta"], ca["pose_n_azim"]])),
            "contrast_factor": ca["contrast_factor"]}
        e["visibility_ratio"] = sc["visibility_ratio"]
        e["coverage_eps"] = {"value": 1.0, "rule": "2 * test_resolution 0.05 * scene_scale_factor 10",
                             "evidence": "exp3b m_t6_sim.py:72; m_magician_replay.py:51; MAGICIAN [CM]:46"}
        # starts
        L = lt.get(s, {})
        starts = []
        for k, idx in enumerate(ca["start_positions"]):
            row = {"k": k, "pose_idx": idx}
            if L.get("ok"):
                r = L["starts"][k]
                row.update({"X_runtime": [r4(x) for x in r["X"]], "elev": r4(r["elev"]), "azim": r4(r["azim"])})
                if "matches_lmdb" in r:
                    row["lmdb_X0_V0_match"] = r["matches_lmdb"]
                    row["lmdb_max_abs_dX"] = r4(r["max_abs_dX"])
            starts.append(row)
        e["official_starts"] = {
            "source": "settings.json camera.start_positions (the only start definition; MAGICIAN test loop "
                      "magician_planning.py:1457-1462 in the clean worktree iterates all 5; the test config [CT] "
                      "has no start list)",
            "episode_sim": "m_t6_sim.py:69-71 EpisodeSim(scene, k) -> setup_test_camera(settings.camera.start_positions[k])",
            "starts": starts}
        if L.get("ok"):
            e["official_starts"]["episode_sim_k0_key_equals_settings_start0"] = L["init_key_equals_settings_start0"]
            e["official_starts"]["all_5_match_lmdb_X0_V0"] = (all(x.get("lmdb_X0_V0_match") for x in starts)
                                                              if any("lmdb_X0_V0_match" in x for x in starts) else None)
        # planner tables
        P = pt[s]
        z = {"value": r4(P["z0_frozen"] if P["z0_frozen"] is not None else P["z0_rule"]),
             "rule_value": r4(P["z0_rule"]),
             "rule": "abl_common.z0_of = sum(scene box extent x 10) / 3 / 2 (MAGICIAN scene_scale / 2)"}
        if P["z0_frozen"] is not None:
            z["status"] = "frozen (task1/scripts/v2_common.py:32; asserted equal to the rule, abl_common.py:80-81)"
        else:
            z["status"] = ("derived, not frozen: not in v2_common.Z0 (Dev-3 only); the Task 2 outdoor child computes it "
                           "at start-up with the same rule (t24_v2_out_planner.py:46)" +
                           ("; Test-3 closed loops already ran with this derived value" if s in TEST3 else ""))
        e["z0"] = z
        g = {"em2_entry_present": P["em2_present"]}
        if P["em2_present"]:
            g.update({"k_M": r4(P["k_M_em2"]), "status": "frozen (work_dirs/exp3b_macaronspp_transfer/task1_protocol/"
                      "em2_windows.json)", "scene_geometry_ok": P["scene_geometry_ok"],
                      "grid_dims": P.get("geometry_dims"), "n_windows_official": P.get("geometry_n_windows")})
        else:
            g.update({"k_M": r4(P["k_M_rule"]), "status": "derived, not frozen (rule k_M = y-extent x 10 / 2.816; "
                      "the same rule reproduces the frozen k_M of barts / eiffel / sestino_museum exactly)",
                      "derived_grid_dims": [656, 443, 256], "derived_n_windows_official": 6,
                      "scene_geometry_ok": False, "scene_geometry_error": P.get("scene_geometry_error"),
                      "fails_at": "t5_diagnostic.py:111 frame_of (em2_windows.json has 14 scenes)"})
        e["planner_scene_geometry"] = g
        # MAGICIAN LMDB
        keys = [k for k in (lm["rows"] if lm else {}) if k.split("/")[0] == s]
        if keys:
            rows = [lm["rows"][k] for k in sorted(keys)]
            e["magician_lmdb"] = {"exists": True, "n_starts": len(rows), "starts": sorted(int(k.split("/")[1]) for k in keys),
                                  "X_cam_history_shape": rows[0]["X_shape"], "V_cam_history_shape": rows[0]["V_shape"],
                                  "n_coverage": sorted({r["n_coverage"] for r in rows}),
                                  "final_mean": r4(np.mean([r["cov_final_last"] for r in rows])),
                                  "auc_mean_trapz_over_100": r4(np.mean([r["auc_exp1"] for r in rows])),
                                  "source": "MAGICIAN/results/scene_exploration/magician_lmdb (2026-06-15 reproduction)"}
        else:
            e["magician_lmdb"] = {"exists": False}
        e["magician_paper"] = {"auc": dict(zip(METHODS, PAPER_AUC[s])), "final": dict(zip(METHODS, PAPER_FINAL[s])),
                               "source": "MAGICIAN/Magician.pdf p.12 Table 6 (AUC) / Table 7 (final coverage)"}
        # historical closed loops
        full, short = [], []
        for rel, sc_ in sorted((cl or {"episodes": {}})["episodes"].items()):
            if s not in sc_:
                continue
            d = sc_[s]
            item = rel.replace("/episodes", "")
            if d["starts"] == [0, 1, 2, 3, 4] and d["moves"] == [100] and "/failed/" not in rel:
                full.append(item)
            else:
                short.append(f"{item} (n={d['n']}, starts {d['starts']}, moves {d['moves']})")
        e["historical_closed_loops"] = {"full_5starts_100moves": full, "other": short}
        # load test
        if L:
            e["load_test"] = {k: (r4(L.get(k)) if isinstance(L.get(k), float) else L.get(k)) for k in
                              ("ok", "stage", "init_s", "observe0_s", "legal0_s", "n_gt_points", "eps", "vis", "cov0",
                               "n_samples0", "n_legal0", "n_valid_neighbours0", "n_collide0", "mesh_n_verts",
                               "mesh_n_faces", "texture_type", "pose_space_size", "peak_mem_gb", "error")
                              if k in L}
            if not L.get("ok"):
                tb = L.get("traceback", "")
                m = re.findall(r'File "([^"]+)", line (\d+), in (\w+)\n\s+(.+)', tb)
                e["load_test"]["traceback_frames"] = [f"{os.path.basename(f)}:{ln} {fn}: {code.strip()}" for f, ln, fn, code in m]
        sim_ok = bool(L.get("ok"))
        geo_ok = bool(P["scene_geometry_ok"])
        e["runnable"] = {"m_protocol_sim": sim_ok, "magician_planner_child": sim_ok,
                         "v2_family_planner_child": sim_ok and geo_ok,
                         "overall": sim_ok and geo_ok,
                         "note": ("" if sim_ok and geo_ok else
                                  "EpisodeSim fails in MAGICIAN load_scene (stray torch.tensor() in the multi-texture branch, "
                                  "MAGICIAN_clean_9521d69 macarons_utils.py:494; main repo :570; commented out upstream in "
                                  "eec9e1d, official HEAD e697009 macarons_utils.py:481) and em2_windows.json has no entry")}
        if s == "sofia_church" and diag:
            e["diagnostic_without_stray_line"] = {
                "what": "in-process copy of load_scene without the stray line (scripts/sofia_diag.py); NOT the frozen "
                        "protocol, no file modified",
                **{k: (r4(diag.get(k)) if isinstance(diag.get(k), float) else diag.get(k)) for k in
                   ("ok", "init_s", "n_gt_points", "eps", "vis", "texture_type", "mesh_n_faces",
                    "init_key_equals_settings_start0", "cov0", "n_samples0", "n_legal0", "n_collide0", "peak_mem_gb")},
                "starts_X_runtime": [{"k": x["k"], "pose_idx": json.loads(x["key"]), "X": [r4(v) for v in x["X"]],
                                      "elev": r4(x["elev"]), "azim": r4(x["azim"])} for x in diag.get("starts", [])]}
        scenes.append(e)
    meta = {
        "title": "Macarons++ scene registry for CVPR Table 1 (audit 2026-11-03, read-only)",
        "data_root": DATA,
        "official_scene_count": 15,
        "local_scene_dirs": 15,
        "magician_test_list_count": 14,
        "split_source": f"{MAN} (FROZEN 2026-10-14, sha256 e3cf35d7...; checked OK against SHA256SUMS) and {PROT}:28-35",
        "split_sets": {"train": sorted(s for s in SPLIT if SPLIT[s] == "train"),
                       "dev": sorted(s for s in SPLIT if SPLIT[s] == "dev"), "dev3_closed_loop": DEV3,
                       "test": TEST3, "other_unassigned": ["sofia_church"]},
        "S-c": "start-level hold-out inside each Train-6 scene (1 of 4 'A' starts + 2 of 11 'B' starts, seeded "
               "sha256('exp5-V2|holdout|<scene>')): task1/protocol.md:152-154, t11_split.py:36-43,"
               "task1/frozen_values.json 'split' (line 3); 44 / 180 trajectories; all historical models train on "
               "'train6_minus_Sc'; S-c is NOT a scene-level split",
        "global_protocol": {"scene_scale_factor": 10.0, "coverage_eps": 1.0, "gathering": "10% of z<70 pixels",
                            "moves": 100, "coverage_entries": 101,
                            "auc_ours": "trapz(cov[0..100]) / 100 (m_t7_common.py:32-35)",
                            "magician_clean_worktree": "MAGICIAN_clean_9521d69"},
        "paper_average_15": {"auc": dict(zip(METHODS, PAPER_AVG["auc"])), "final": dict(zip(METHODS, PAPER_AVG["final"]))},
        "paper_average_14_without_sofia": {
            "auc": {m: r4(np.mean([PAPER_AUC[s][i] for s in PAPER_AUC if s != "sofia_church"])) for i, m in enumerate(METHODS)},
            "final": {m: r4(np.mean([PAPER_FINAL[s][i] for s in PAPER_FINAL if s != "sofia_church"])) for i, m in enumerate(METHODS)}},
        "lmdb_average_14": ({"auc": r4(np.mean([r["auc_exp1"] for r in lm["rows"].values()])),
                             "final": r4(np.mean([r["cov_final_last"] for r in lm["rows"].values()])),
                             "n_episodes": lm["n_keys"]} if lm else None),
        "training_evidence": TRAIN6_MODELS,
        "generator": "experiments/cvpr_results/table1/scripts/build_registry.py",
    }
    with open(os.path.join(OUT, "SCENE_REGISTRY.yaml"), "w") as f:
        f.write("# Macarons++ scene registry for CVPR Table 1. Generated by scripts/build_registry.py (2026-11-03).\n"
                "# All facts are read from code / configs / data; 'derived, not frozen' marks values computed from a\n"
                "# documented rule that no frozen file stores. See DATASET_AUDIT.md for the discussion.\n")
        yaml.dump({"meta": meta, "scenes": scenes}, f, Dumper=FlowDumper, sort_keys=False, allow_unicode=True, width=200)
    print("written", os.path.join(OUT, "SCENE_REGISTRY.yaml"))


if __name__ == "__main__":
    main()
