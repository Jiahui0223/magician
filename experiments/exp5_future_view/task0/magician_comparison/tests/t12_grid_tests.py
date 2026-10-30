"""EXP5 Task 0 / T0.12 interface tests E4-E5 (protocol v1.3 §21). Magician env, one GPU.
E4  grid_env vs the OFFICIAL macarons Camera built from the same settings camera block (oLBMNvg9in8_L1, 11,520 poses):
    every pose position (runtime units) and angle equal within 1e-4; grid_env.neighbours(keep_self=True) equal to the
    official get_neighboring_poses for 200 random poses (seed 0)
E5  start snapping for the 10 T0.12 episodes: snapped grid pose, distance to the panorama centre (m), and >= 1 legal action
Output W/magician_comparison/grid_protocol/tests/grid_tests.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import magician_core as CORE  # noqa: E402
import grid_env as GE  # noqa: E402
import t12_sim as T12  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

MP, S = CORE.MP, T12.S
EPISODES = [("Z6MFQCViBuw", 0, 6), ("Z6MFQCViBuw", 0, 17), ("oLBMNvg9in8", 1, 36), ("oLBMNvg9in8", 1, 78), ("QUCTc6BB5sX", 1, 34),
            ("QUCTc6BB5sX", 1, 135), ("2azQ1b91cZZ", 0, 60), ("2azQ1b91cZZ", 0, 179), ("TbHJrupSAjP", 1, 112), ("TbHJrupSAjP", 1, 114)]


def main():
    dev = torch.device("cuda:0")
    res = {}
    card = json.load(open(os.path.join(S.W, "preprocessing", "scenes", "oLBMNvg9in8_L1.json")))
    g = GE.grid_def(card)
    sc = MP.Settings({"scene": {"x_min": [0, 0, 0], "x_max": [1, 1, 1], "grid_l": 1, "grid_w": 1, "grid_h": 1, "cell_capacity": 1,
                                "cell_resolution": 0.05, "visibility_ratio": 1.0}, "camera": GE.settings_camera(g)}, dev, 10.0).camera
    p = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
    renderer = MP.get_rgb_renderer(image_height=p.image_height, image_width=p.image_width, ambient_light_intensity=p.ambient_light_intensity,
                                   cameras=MP.FoVPerspectiveCameras(device=dev), device=dev, max_faces_per_bin=200000)
    cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=sc.pose_l, pose_w=sc.pose_w, pose_h=sc.pose_h, pose_n_elev=sc.pose_n_elev,
                    pose_n_azim=sc.pose_n_azim, n_interpolation_steps=1, zfar=p.zfar, renderer=renderer, device=dev, save_dir_path=None)
    err_pos = err_ang = 0.0
    for k, pose in cam.pose_space.items():
        idx = [int(c) for c in k[1:-1].split(",")]
        err_pos = max(err_pos, float(np.abs(GE.position(g, idx) - pose[:3].cpu().numpy()).max()))
        e, a = GE.angles(g, idx)
        err_ang = max(err_ang, abs(e - float(pose[3])), abs(a - float(pose[4])))
    rng = np.random.default_rng(0)
    keys = list(cam.pose_space)
    mism = 0
    for j in rng.choice(len(keys), 200, replace=False):
        idx = [int(c) for c in keys[j][1:-1].split(",")]
        off = sorted(tuple(int(v) for v in r) for r in cam.get_neighboring_poses(pose_idx=torch.tensor(idx, device=dev)).cpu().numpy())
        mism += off != GE.neighbours(g, idx, keep_self=True)
    res["E4"] = {"level": "oLBMNvg9in8_L1", "n_poses": len(keys), "grid_n": list(g["n"]), "max_pos_err_rt": err_pos, "max_ang_err_deg": err_ang,
                 "neighbour_mismatch_of_200": int(mism), "pass": err_pos < 1e-4 and err_ang < 1e-4 and mism == 0}
    print("E4", res["E4"], flush=True)
    e5 = []
    for scan, lv, st in EPISODES:
        L = S.Level(scan, lv)
        v, f = S.mesh_zup(scan)
        blk = T12.Blocker(v, f)
        crd = json.load(open(os.path.join(S.W, "preprocessing", "scenes", f"{L.key}.json")))
        gg = GE.grid_def(crd)
        s_local = next(i for i, q in enumerate(L.panos) if q["idx"] == st)
        pose, d = T12.start_pose(gg, S.rt_to_zup_m(np.asarray(L.pano_rt[s_local], np.float64)), blk)
        legal = T12.legal_actions(gg, pose, blk)
        e5.append({"episode": f"{L.key}_S{st}", "grid_n": list(gg["n"]), "step_m": [round(x, 3) for x in gg["step_m"]], "start_pose": list(pose),
                   "snap_m": round(d, 3), "n_legal": len(legal), "n_legal_moves": sum(q[:3] != pose[:3] for q in legal)})
        print("E5", e5[-1], flush=True)
    res["E5"] = {"episodes": e5, "pass": all(x["n_legal_moves"] > 0 for x in e5)}
    res["pass"] = res["E4"]["pass"] and res["E5"]["pass"]
    od = os.path.join(S.W, "magician_comparison", "grid_protocol", "tests")
    os.makedirs(od, exist_ok=True)
    json.dump(res, open(os.path.join(od, "grid_tests.json"), "w"), indent=1)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
