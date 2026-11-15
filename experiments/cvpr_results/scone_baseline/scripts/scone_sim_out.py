"""EXP9-SCONE -- Macarons++ outdoor closed loop of SCONE (magician env, 1 GPU). Same harness as the Table 1 MACARONS row.

This file is experiments/exp3b_macaronspp_transfer/m_t7_macarons.py (frozen; the Table 1 MACARONS runner, adapter repairs R1-R8)
copied line for line, with exactly these differences (everything else -- EpisodeSim, start poses, legal list, GT depth sensor,
shared 10 % sample, seeds, surface recompute every 20 loops, coverage, episode format -- is the frozen code path):
  D1 weights: the official ShapeNet-pretrained SCONE modules (scone_common.load_scone_model: pretrained_scone_occ.pth +
     pretrained_scone_vis.pth from the MACARONS Google Drive, checked bitwise equal to the scone part of pretrained_macarons.pth)
     instead of load_pretrained_macarons + trained_macarons.pth (MACARONS self-supervised fine-tuning).
  D2 method name "scone" (episode tag scone_<scene>_s<k>_r<seed>, planner seeds derived from "scone").
  D3 the NBV step is scone_common.scone_greedy (the same official calls in the same order as m_t7_macarons.py:111-123).
  D4 diagnostics only (never read by the decision): occupancy-field statistics per step, optional snapshots of the occupancy
     field (--diag-steps) under --diag-dir, and an end-of-episode information-flow check (every point of the planner surface map
     is one of the observed 10 % depth samples of this trajectory).
Usage: $MAGICIAN_PYTHON scone_sim_out.py --scene S --start K --seed 0 --out DIR [--moves N] [--diag-dir D --diag-steps 0 5 10]
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
EXP3B = os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer")
sys.path.insert(0, EXP3B)
sys.path.insert(0, HERE)
import m_magician_replay as RP  # noqa: E402
import m_common as MC  # noqa: E402
import m_t6_common as T6  # noqa: E402
import m_t7_common as T7  # noqa: E402
import m_t6_magician as MG  # noqa: E402  (test_params: identical params setup; WEIGHTS = clean worktree weights dir)
import m_t6_sim as SIM  # noqa: E402
import scone_common as SC  # noqa: E402

np, torch, MP, P = RP.np, RP.torch, RP.MP, RP.P
from macarons.utility.macarons_utils import (compute_scene_occupancy_probability_field,  # noqa: E402
                                             fill_surface_scene, get_curriculum_sampling_distances,
                                             load_pretrained_macarons, predict_coverage_gain_for_single_camera)

METHOD = "scone"


def surface_from_observations(surface_scene, full_pc):
    """D4: every point stored in the planner surface map must be an observed 10 % depth sample (bitwise)."""
    pts = [surface_scene.cells[c].cell_pts for c in surface_scene.cells if len(surface_scene.cells[c].cell_pts) > 0]
    if not pts:
        return {"n_surface": 0, "n_full_pc": int(len(full_pc)), "all_observed": True}
    S = torch.cat(pts).float().cpu().numpy()
    F = full_pc.float().cpu().numpy()
    seen = {r.tobytes() for r in F}
    n_bad = sum(1 for r in S if r.tobytes() not in seen)
    return {"n_surface": int(len(S)), "n_full_pc": int(len(F)), "n_not_observed": int(n_bad), "all_observed": n_bad == 0}


def run(a):
    t_start = time.time()
    s, k, seed, moves = a.scene, a.start, a.seed, a.moves
    tag = T6.episode_tag(METHOD, s, k, seed)
    ep_path = os.path.join(a.out, "episodes", f"ep_{tag}.json")
    assert not os.path.exists(ep_path), ep_path
    os.makedirs(os.path.dirname(ep_path), exist_ok=True)
    dev = torch.device("cuda")
    params = MG.test_params(RP.load_params(os.path.join(P.MAGICIAN_CLEAN_ROOT, "configs", "macarons",
                                                        "macarons_default_training_config.json")), (1, 1))
    macarons, model_info = SC.load_scone_model(load_pretrained_macarons, MG.WEIGHTS, dev)            # D1
    frames_root = os.path.join(a.out, "frames", tag)
    sim = SIM.EpisodeSim(s, k, frames_root, keep_planner_scenes=True)
    torch.manual_seed(T6.planner_seed(METHOD, s, k, seed))                              # R8
    sim.proxy_scene.initialize_proxy_points()
    camera, surface_scene, proxy_scene, covered = sim.camera, sim.surface_scene, sim.proxy_scene, sim.covered
    torch.cuda.reset_peak_memory_stats(dev)
    macarons.eval()
    curriculum_distances = get_curriculum_sampling_distances(params, surface_scene, proxy_scene)  # noqa: F841 (as scene.py)
    full_pc = torch.zeros(0, 3, device=dev)
    cov, steps = [], []
    if a.diag_dir:
        os.makedirs(a.diag_dir, exist_ok=True)
    for t in range(moves + 1):
        t_iter = time.time()
        camera.fov_camera_0 = camera.fov_camera
        recomputed = False
        if t > 0 and t % params.recompute_surface_every_n_loop == 0:          # scene.py:475-481 (native)
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)
            recomputed = True
        current_frame = MP.load_current_frame_perfect_depth(camera, dev)
        depth, mask, error_mask, R, T = MP.apply_perfect_depth_simple(current_frame, dev)
        fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        X_cam = fov_camera.get_camera_center()
        torch.manual_seed(MC.derived_seed(s, k, t, "sample"))                         # R8
        part_pc, part_pc_features = camera.compute_partial_point_cloud(
            depth=depth, mask=(mask * error_mask).bool(), images=current_frame['rgb'], fov_cameras=fov_camera,
            gathering_factor=params.gathering_factor * 2, fov_range=params.sensor_range)   # R5
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(proxy_scene.proxy_points, return_mask=True,
                                                                    fov_camera=None, fov_range=params.sensor_range)
        sgn_dists = None
        if fov_proxy_mask.any():
            sgn_dists = camera.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth, mask=mask,
                                                                 fov_camera=None)
        zeros = torch.zeros(len(part_pc), 1, device=dev)
        torch.manual_seed(MC.derived_seed(s, k, t, "cap"))                            # R8
        covered.fill_cells(part_pc, features=zeros)
        torch.manual_seed(T6.planner_step_seed(METHOD, s, k, seed, t))                # R8
        t_map = time.time()
        surface_scene.fill_cells(part_pc, features=zeros)
        full_pc = torch.vstack((full_pc, part_pc))
        if fov_proxy_mask.any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(fov_proxy_mask)
            proxy_scene.fill_cells(fov_proxy_points, features=fov_proxy_indices.view(-1, 1))
            proxy_scene.update_proxy_view_states(camera, fov_proxy_mask, signed_distances=sgn_dists,
                                                 distance_to_surface=None, X_cam=X_cam)
            proxy_scene.update_proxy_supervision_occ(fov_proxy_mask, sgn_dists, tol=params.carving_tolerance)
            proxy_scene.update_proxy_out_of_field(fov_proxy_mask)
        surface_scene.set_all_features_to_value(value=1.)
        map_s = time.time() - t_map
        c = sim.gt_scene.scene_coverage(covered, surface_epsilon=sim.eps)              # R4 (evaluation side)
        cov.append((c[0].item() if c[0] != 0. else 0.) / sim.vis)
        legal, _ = sim.legal()                                                        # R1, R2
        keys = [T6.key_of(x[0]) for x in legal]
        step = {"t": t, "pose_key": sim.key(), "legal_keys": keys, "n_legal": len(keys), "n_samples": int(len(part_pc)),
                "part_pc_sha": MG.sha_t(part_pc), "surface_recomputed": recomputed}
        if t == moves:
            steps.append(step)
            break
        torch.cuda.synchronize()
        t_dec = time.time()
        cands = []
        for key in keys:
            neighbor_idx = camera.get_idx_from_key(key)
            neighbor_pose, _ = camera.get_pose_from_idx(neighbor_idx)
            X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
            cands.append((X_neighbor, fov_neighbor))
        pick, gains, X_world, occ_probs = SC.scone_greedy(params, macarons.scone, camera, surface_scene, proxy_scene, cands,
                                                          dev, compute_scene_occupancy_probability_field,
                                                          predict_coverage_gain_for_single_camera)       # D3
        torch.cuda.synchronize()
        dec_s = time.time() - t_dec
        step.update({"pred_gains": gains, "pick": pick, "action_key": keys[pick] if pick is not None else None,
                     "hover": pick is None, "default_used": bool(keys) and all(g is None for g in gains),
                     "occ": SC.occ_stats(X_world, occ_probs, proxy_scene),                              # D4
                     "map_update_s": map_s, "decision_s": dec_s, "planning_s": map_s + dec_s,
                     "iter_s": time.time() - t_iter})
        if a.diag_dir and t in a.diag_steps:                                                            # D4
            np.savez_compressed(os.path.join(a.diag_dir, f"{tag}_occ_t{t}.npz"), X=X_world.float().cpu().numpy(),
                                n_pred=int(X_world.shape[0] - (proxy_scene.out_of_field > 0.).sum().item()),
                                occ=occ_probs.float().cpu().numpy().reshape(-1), cam=X_cam.float().cpu().numpy(),
                                legal_keys=np.asarray(keys), gains=np.asarray([np.nan if g is None else g for g in gains]),
                                pick=-1 if pick is None else pick)
        steps.append(step)
        sim.move(None if pick is None else keys[pick])
    flow = surface_from_observations(surface_scene, full_pc)                                             # D4
    out = {"method": METHOD, "scene": s, "start": k, "seed": seed, "tag": tag, "kind": "scone", "moves": moves,
           "model": model_info, "steps": steps, "coverage": cov,
           "auc": T7.auc(cov, moves), "cov_final": cov[-1], "cov0": cov[0],
           "cov_at": {str(x): cov[x] for x in (10, 20, 50, 80, 100) if x <= moves},
           "executed_keys": [x["pose_key"] for x in steps], "n_hover": sum(1 for x in steps[:-1] if x["hover"]),
           "n_default_used": sum(1 for x in steps[:-1] if x["default_used"]),
           "planning_s_total": float(sum(x["planning_s"] for x in steps[:-1])), "info_flow": flow,
           "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30, "seconds_total": time.time() - t_start,
           "runner": os.path.abspath(__file__), "runner_sha256": SC.sha256_file(os.path.abspath(__file__)),
           "common_sha256": SC.sha256_file(os.path.join(HERE, "scone_common.py"))}
    json.dump(out, open(ep_path + ".tmp", "w"), indent=1)
    os.replace(ep_path + ".tmp", ep_path)
    os.chmod(ep_path, 0o444)
    shutil.rmtree(frames_root)
    print(f"SCONE episode {tag}: done, AUC {out['auc']:.4f} final {cov[-1]:.4f}, {out['seconds_total']:.0f} s, "
          f"info_flow {flow}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--moves", type=int, default=T7.T7_MOVES)
    ap.add_argument("--out", required=True)
    ap.add_argument("--diag-dir", default=None)
    ap.add_argument("--diag-steps", type=int, nargs="*", default=[])
    run(ap.parse_args())


if __name__ == "__main__":
    main()
