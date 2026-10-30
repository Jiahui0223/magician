"""EXP5 Task 0 / T0.10 -- MAGICIAN planner core, a line-by-line re-statement of the loop body of the OFFICIAL
compute_magician_trajectory (macarons/testers/magician_planning.py, commit e697009) that calls only official functions.
Nothing of the official repository is modified; the only re-implemented logic is the orchestration and the hook that
lets a caller supply the candidate poses of each beam layer (the official code reads its native 3D pose grid).

Per decision step (official order):
  begin_step(pose_i): fov_camera_0 = fov_camera; every recompute_surface_every_n_loop steps fill_surface_scene(full_pc)
  observe(depth, mask, part_pc): surface fill, full_pc, proxy fill / view states / carving / out-of-field
  plan(neighbours): occupancy field (SCONE, occ > 0.5 -> Imagined Gaussians, scale 0.7154 / 2), novelty from every past
                    camera, beam search (width x steps) with the official gain (novelty x min(1, (d / (scene_scale / 2))^2)),
                    layer 0 candidates as given (environment-legal), deeper layers filtered by the official
                    occupancy-point segment test; returns the first pose of the best beam + diagnostics.
neighbours(pose_key, layer) -> list of (pose_key, X (1, 3), V (1, 2)) AFTER the caller's validity rule
(official get_valid_neighbors = prefer poses not visited by the real trajectory).
Run with the magician env and the official clone on sys.path (MAGICIAN_ROOT).
"""
import os
import sys
import time

MAGICIAN_ROOT = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/exp5_future_view/task0/magician_comparison/external/MAGICIAN"
if MAGICIAN_ROOT not in sys.path:
    sys.path.insert(0, MAGICIAN_ROOT)

import torch  # noqa: E402
from macarons.testers import magician_planning as MP  # noqa: E402
from macarons.utility.magician_utils import line_segment_intersects_point_cloud_region  # noqa: E402

GAUSSIAN_SCALE = 0.7154 / 2      # official constant (magician_planning.py)
VIS_TOL = 1.0                    # official depth_tolerance of check_point_visibility_from_depth


def load_model(params, device):
    """Official setup_test model part: pretrained MACARONS + trained_macarons.pth (ddp state dict)."""
    macarons = MP.load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path, device=device,
                                           learn_pose=params.learn_pose)
    tw = torch.load(os.path.join(MP.weights_dir, "trained_macarons.pth"), map_location=device, weights_only=False)
    macarons.load_state_dict(tw["model_state_dict"], ddp=True)
    macarons.eval()
    return macarons


def make_planner_scenes(params, settings, device):
    """surface_scene and proxy_scene exactly as the official setup_test_scene builds them (no GT / covered scene)."""
    Scene = MP.Scene
    surface_scene = Scene(x_min=settings.scene.x_min, x_max=settings.scene.x_max, grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w, grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity, cell_resolution=None,
                          n_proxy_points=params.n_proxy_points, device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=1, mirrored_scene=False, mirrored_axis=None)
    proxy_scene = Scene(x_min=settings.scene.x_min, x_max=settings.scene.x_max, grid_l=settings.scene.grid_l,
                        grid_w=settings.scene.grid_w, grid_h=settings.scene.grid_h,
                        cell_capacity=params.proxy_cell_capacity, cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points, device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1, mirrored_scene=False, score_threshold=params.score_threshold, mirrored_axis=None)
    proxy_scene.initialize_proxy_points()
    return surface_scene, proxy_scene


class MagicianCore:
    def __init__(self, params, settings, macarons, camera, surface_scene, proxy_scene, device, beam_width, beam_steps):
        self.params, self.settings, self.macarons, self.camera = params, settings, macarons, camera
        self.surface_scene, self.proxy_scene, self.dev = surface_scene, proxy_scene, device
        self.beam_width, self.beam_steps = int(beam_width), int(beam_steps)
        ext = settings.scene.x_max - settings.scene.x_min
        self.scene_scale = float((ext[0] + ext[1] + ext[2]) / 3.0)          # official scene_scale
        self.full_pc = torch.zeros(0, 3, device=device)
        self.K = None
        self.pose_i = 0

    # ------------------------------------------------------------------------------------------------ official order
    def begin_step(self, pose_i):
        self.pose_i = pose_i
        self.camera.fov_camera_0 = self.camera.fov_camera
        if pose_i > 0 and pose_i % self.params.recompute_surface_every_n_loop == 0:
            MP.fill_surface_scene(self.surface_scene, self.full_pc, random_sampling_max_size=self.params.n_gt_surface_points,
                                  min_n_points_per_cell_fill=3, progressive_fill=self.params.progressive_fill,
                                  max_n_points_per_fill=self.params.max_points_per_progressive_fill)

    def observe(self, depth, mask, part_pc):
        """depth / mask (1, H, W, 1) as apply_perfect_depth_simple returns them; part_pc = the 10 % sample (N, 3)."""
        cam, ps, p = self.camera, self.proxy_scene, self.params
        fov_proxy_points, fov_proxy_mask = cam.get_points_in_fov(ps.proxy_points, return_mask=True, fov_camera=None,
                                                                 fov_range=p.sensor_range)
        sgn = None
        if fov_proxy_mask.any():
            sgn = cam.get_signed_distance_to_depth_maps(pts=fov_proxy_points, depth_maps=depth, mask=mask, fov_camera=None)
        X_cam = cam.fov_camera.get_camera_center()
        self.surface_scene.fill_cells(part_pc, features=torch.zeros(len(part_pc), 1, device=self.dev))
        self.full_pc = torch.vstack((self.full_pc, part_pc))
        if fov_proxy_mask.any():
            idx = ps.get_proxy_indices_from_mask(fov_proxy_mask)
            ps.fill_cells(fov_proxy_points, features=idx.view(-1, 1))
            ps.update_proxy_view_states(cam, fov_proxy_mask, signed_distances=sgn, distance_to_surface=None, X_cam=X_cam)
            ps.update_proxy_supervision_occ(fov_proxy_mask, sgn, tol=p.carving_tolerance)
            ps.update_proxy_out_of_field(fov_proxy_mask)
        self.surface_scene.set_all_features_to_value(value=1.)

    def _gs(self, fov):
        fov.K = self.K
        return MP.convert_camera_from_pytorch3d_to_gs(fov, height=self.camera.image_height, width=self.camera.image_width,
                                                      device=self.dev)[0]

    def _render(self, means, opac, scales, rots, colors, fov):
        d, img = MP.render_gaussian_depth(gaussian_means=means, gaussian_opacities=opac, gaussian_scales=scales,
                                          gaussian_rotations=rots, gaussian_colors=colors, gs_camera=self._gs(fov),
                                          device=self.dev, bg_color=torch.tensor([1.0, 1.0, 1.0], device=self.dev),
                                          kernel_size=0.01)
        return d[0], img

    def plan(self, current_key, neighbours):
        """Returns (next pose key or None, diagnostics)."""
        cam, dev = self.camera, self.dev
        t0 = time.time()
        with torch.no_grad():
            X_world, _, occ = MP.compute_scene_occupancy_probability_field(self.params, self.macarons.scone, cam,
                                                                          self.surface_scene, self.proxy_scene, dev)
        keep = occ.squeeze() > 0.5
        Xg = X_world[keep]
        n = Xg.shape[0]
        opac = occ[keep]
        scales = torch.ones(n, 3, device=dev) * GAUSSIAN_SCALE
        rots = torch.tensor([[1, 0, 0, 0]], device=dev, dtype=torch.float32).repeat(n, 1)
        t_occ = time.time()
        if self.K is None:                      # official: at pose_i == 0, from the first camera of the history
            R0, T0 = MP.get_camera_RT(cam.X_cam_history[0].view(1, 3), cam.V_cam_history[0].view(1, 2))
            self.K = MP.FoVPerspectiveCameras(R=R0, T=T0, zfar=cam.zfar, device=dev).get_projection_transform(
            ).get_matrix().transpose(-1, -2)
        novelty = torch.zeros(n, device=dev)
        colors0 = MP.update_gaussian_colors_from_novelty(torch.zeros(n, device=dev))
        for i in range(len(cam.X_cam_history)):
            R, T = MP.get_camera_RT(cam.X_cam_history[i].view(1, 3), cam.V_cam_history[i].view(1, 2))
            fov = MP.FoVPerspectiveCameras(R=R, T=T, zfar=cam.zfar, device=dev)
            with torch.no_grad():
                dm, _ = self._render(Xg, opac, scales, rots, colors0, fov)
                novelty[cam.check_point_visibility_from_depth(Xg, fov, dm, depth_tolerance=VIS_TOL)] = 1.0
        t_nov = time.time()
        beams = [{"trajectory": [], "novelty": novelty.clone(), "total": 0.0, "key": current_key, "X": None, "first": None}]
        expanded, depth_thr, layer0 = 0, self.scene_scale / 2.0, {}
        for layer in range(self.beam_steps):
            cands = []
            for beam in beams:
                for key, Xn, Vn in neighbours(beam["key"], layer):
                    # official: layer 0 = GT segment test (here: the environment-legal list given by `neighbours`);
                    # deeper layers = segment test against the predicted-occupancy points
                    if layer > 0 and line_segment_intersects_point_cloud_region(Xg, beam["X"][0], Xn[0]):
                        continue
                    R, T = MP.get_camera_RT(Xn, Vn)
                    fov = MP.FoVPerspectiveCameras(R=R, T=T, zfar=cam.zfar, device=dev)
                    with torch.no_grad():
                        dm, img = self._render(Xg, opac, scales, rots, MP.update_gaussian_colors_from_novelty(beam["novelty"]), fov)
                        vis = cam.check_point_visibility_from_depth(Xg, fov, dm, depth_tolerance=VIS_TOL)
                        valid = dm > 0
                        gray = img.mean(dim=0)
                        gain = float((gray * ((dm / depth_thr) ** 2).clamp_max(1.0) * valid.float()).sum().item()) if valid.any() else 0.0
                    nv = beam["novelty"].clone()
                    nv[vis] = 1.0
                    expanded += 1
                    if layer == 0:
                        layer0[key] = gain
                    cands.append({"trajectory": beam["trajectory"] + [key], "novelty": nv, "total": beam["total"] + gain,
                                  "key": key, "X": Xn, "first": gain if layer == 0 else beam["first"]})
            if not cands:
                break
            cands.sort(key=lambda x: x["total"], reverse=True)       # official: stable sort on total_coverage_gain
            beams = cands[:self.beam_width]
        best = beams[0] if beams and beams[0]["trajectory"] else None
        diag = {"n_gaussians": int(n), "n_history": int(len(cam.X_cam_history)), "expanded": expanded,
                "layers": layer + 1, "best_total": best["total"] if best else None,
                "best_first_gain": best["first"] if best else None, "layer0_gains": layer0,
                "occ_s": t_occ - t0, "novelty_s": t_nov - t_occ, "beam_s": time.time() - t_nov}
        return (best["trajectory"][0] if best else None), diag
