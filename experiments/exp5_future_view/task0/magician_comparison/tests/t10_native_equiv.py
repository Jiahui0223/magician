"""EXP5 Task 0 / T0.10 interface test E1 -- the MAGICIAN core (adapters/magician_core.py) reproduces the OFFICIAL
compute_magician_trajectory bitwise in NATIVE mode (official 3D pose grid, official capture, official GT-mesh test at
layer 0). Same seed and deterministic kernels for both runs:
  A: official compute_magician_trajectory, n_poses moves
  B: official capture / partial point cloud / evaluation + MagicianCore.begin_step / observe / plan with a neighbour
     callback built from the official Camera.get_neighboring_poses + get_valid_neighbors + line_segment_mesh_intersection
Pass = identical executed pose indices, X / V histories and coverage evolution, for beam (10, 10) and (1, 1).
Output W/magician_comparison/shared_protocol/tests/native_equiv.json. Magician env, one GPU.
"""
import json
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "adapters"))
import magician_core as CORE  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

MP = CORE.MP
from macarons.utility.magician_utils import line_segment_mesh_intersection  # noqa: E402

MC = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/magician_comparison"
RAW = "/bsuhome/jiahuizhang/scratch/data/mp3d_magician/mp3d_magician"
SCENE, N_POSES, SEED = "pLe4wQe7qrG", 4, 1234


def setup(params, device, run_dir):
    st = json.load(open(os.path.join(RAW, SCENE, "settings.json")))
    settings = MP.Settings(st, device, params.scene_scale_factor)
    mp = os.path.join(RAW, SCENE, f"{SCENE}.obj")
    torch.use_deterministic_algorithms(False)          # the texture-atlas loader fails under deterministic index_put
    mesh = MP.load_scene(mp, params.scene_scale_factor, device, mirror=False, mirrored_axis=None)
    torch.use_deterministic_algorithms(True, warn_only=True)
    mfc = trimesh.load(mp)
    if isinstance(mfc, trimesh.Scene):
        mfc = mfc.dump(concatenate=True)
    mfc.vertices *= params.scene_scale_factor
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    gt, cov, surf, prox = MP.setup_test_scene(params, mesh, settings, False, device, mirrored_axis=None, test_resolution=0.05)
    for sub in ("frames", "imgs"):                      # official Memory layout: <trajectory>/frames + sibling imgs
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    cam = MP.setup_test_camera(params, mesh, mfc.ray, torch.tensor(settings.camera.start_positions[0], device=device),
                               settings, device, os.path.join(run_dir, "frames"), mirrored_scene=False, mirrored_axis=None)
    return settings, mesh, mfc.ray, gt, cov, surf, prox, cam


def run_official(params, macarons, device, run_dir):
    settings, mesh, inter, gt, cov, surf, prox, cam = setup(params, device, run_dir)
    ce, Xh, Vh, *_ = MP.compute_magician_trajectory(params, macarons, cam, gt, surf, prox, cov, mesh, inter, device, settings,
                                                    test_resolution=0.05, use_perfect_depth_map=True, compute_collision=False)
    return {"coverage": [float(x) for x in ce], "X": Xh.cpu().numpy().tolist(), "V": Vh.cpu().numpy().tolist(),
            "idx": cam.cam_idx_history.cpu().numpy().astype(int).tolist()}


def run_core(params, macarons, device, run_dir):
    settings, mesh, inter, gt, cov, surf, prox, cam = setup(params, device, run_dir)
    core = CORE.MagicianCore(params, settings, macarons, cam, surf, prox, device, params.beam_width, params.beam_steps)

    def neighbours(key, layer):
        nb = cam.get_valid_neighbors(neighbor_indices=cam.get_neighboring_poses(pose_idx=key), mesh=mesh)
        cur, _ = cam.get_pose_from_idx(key)
        Xc, _, _ = cam.get_camera_parameters_from_pose(cur)
        out = []
        for row in nb:
            pose, _ = cam.get_pose_from_idx(row)
            Xn, Vn, _ = cam.get_camera_parameters_from_pose(pose)
            if layer == 0 and line_segment_mesh_intersection(Xc[0].cpu().numpy(), Xn[0].cpu().numpy(), inter):
                continue
            out.append((row, Xn, Vn))
        return out

    ce = []
    for pose_i in range(N_POSES + 1):
        core.begin_step(pose_i)
        fr = MP.load_current_frame_perfect_depth(cam, device)
        depth, mask, err, R, T = MP.apply_perfect_depth_simple(fr, device)
        fov = cam.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        part_pc, _ = cam.compute_partial_point_cloud(depth=depth, mask=(mask * err).bool(), images=fr["rgb"], fov_cameras=fov,
                                                     gathering_factor=params.gathering_factor * 2, fov_range=params.sensor_range)
        cov.fill_cells(part_pc, features=torch.zeros(len(part_pc), 1, device=device))
        core.observe(depth, mask, part_pc)
        c = gt.scene_coverage(cov, surface_epsilon=2 * 0.05 * params.scene_scale_factor)
        cc = c[0].item() if c[0] != 0. else 0.
        ce.append(cc / settings.scene.visibility_ratio)
        nxt, _ = core.plan(cam.cam_idx, neighbours)
        cam.update_camera(nxt, interpolation_step=1)
        cam.capture_image(mesh)
    return {"coverage": [float(x) for x in ce], "X": cam.X_cam_history.cpu().numpy().tolist(),
            "V": cam.V_cam_history.cpu().numpy().tolist(), "idx": cam.cam_idx_history.cpu().numpy().astype(int).tolist()}


def main():
    torch.use_deterministic_algorithms(True, warn_only=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    params = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
    for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0, WORLD_SIZE=1,
                     batch_size=1, total_batch_size=1, n_poses_in_trajectory=N_POSES).items():
        setattr(params, k, v)
    macarons = CORE.load_model(params, device)
    res = {}
    for bw, bs in ((10, 10), (1, 1)):
        params.beam_width, params.beam_steps = bw, bs
        base = os.path.join(MC, "shared_protocol", "tests", f"native_b{bw}x{bs}")
        a = run_official(params, macarons, device, os.path.join(base, "A"))
        b = run_core(params, macarons, device, os.path.join(base, "B"))
        same = a == b
        res[f"beam{bw}x{bs}"] = {"identical": same, "A": a, "B": b}
        print(f"beam {bw}x{bs}: identical = {same}", a["idx"], b["idx"], a["coverage"], b["coverage"], flush=True)
    res["pass"] = all(v["identical"] for v in res.values())
    json.dump(res, open(os.path.join(MC, "shared_protocol", "tests", "native_equiv.json"), "w"), indent=1)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
