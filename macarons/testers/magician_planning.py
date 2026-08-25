import os
import sys
import gc
import shutil
from ..utility.macarons_utils import *
from ..utility.utils import count_parameters
from ..utility.gaussian_utils import CamerasWrapper, convert_camera_from_pytorch3d_to_gs
from ..utility.magician_utils import *
import trimesh
import lmdb

# ==================== RaDe-GS Integration ====================
RADE_GS_PATH = os.path.join(os.path.dirname(__file__), "../../RaDe-GS")
if RADE_GS_PATH not in sys.path:
    sys.path.insert(0, RADE_GS_PATH)


def write_ply_bin(path, xyz, rgb, max_points=400000):
    """Binary little-endian PLY, xyz float32 + rgb uint8. Used by the exp4 gain probe to
    dump the imagined cloud tagged real/phantom, so the audit is inspectable in 3D and not
    only as a screen-space map. Deterministically subsampled so a dump stays a few MB."""
    import numpy as _np
    xyz = _np.asarray(xyz, _np.float32).reshape(-1, 3)
    rgb = _np.asarray(rgb, _np.uint8).reshape(-1, 3)
    if len(xyz) > max_points:
        idx = _np.random.RandomState(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    rec = _np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\n")
        f.write(f"element vertex {len(rec)}\n".encode())
        f.write(b"property float x\nproperty float y\nproperty float z\n"
                b"property uchar red\nproperty uchar green\nproperty uchar blue\n"
                b"end_header\n")
        f.write(rec.tobytes())


class SimpleGaussianModel:
    def __init__(self, means, opacities, scales, rotations, colors, device):
        """
        Args:
            means: (N, 3) locations
            opacities: (N, 1) opacity[0, 1]
            scales: (N, 3) 
            rotations: (N, 4) 
            colors: (N, 3) 
        """
        self.device = device
        self._xyz = means.to(device)
        self._opacity = self.inverse_sigmoid(opacities.to(device))  # logit
        self._scaling = torch.log(scales.to(device))  # log
        self._rotation = rotations.to(device)
        self._colors_precomp = colors.to(device)  

        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self.max_radii2D = torch.zeros(means.shape[0], device=device)

    @staticmethod
    def inverse_sigmoid(x, eps=1e-6):
        """ logit: logit(x) = log(x / (1-x))"""
        x = torch.clamp(x, eps, 1 - eps)
        return torch.log(x / (1 - x))

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        # none
        return torch.zeros(self._xyz.shape[0], 1, 3, device=self.device)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def get_opacity_with_3D_filter(self):
        return self.get_opacity

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_rotation(self):
        return self._rotation

    @property
    def get_scaling_n_opacity_with_3D_filter(self):
        return self.get_scaling, self.get_opacity

    @property
    def get_colors_precomp(self):
        return self._colors_precomp


def render_gaussian_depth(gaussian_means, gaussian_opacities, gaussian_scales,
                          gaussian_rotations, gaussian_colors, gs_camera, device,
                          bg_color=None, kernel_size=0.1):
    """

    Args:
        gaussian_means: (N, 3) 
        gaussian_opacities: (N, 1)
        gaussian_scales: (N, 3) 
        gaussian_rotations: (N, 4) 
        gaussian_colors: (N, 3) 
        gs_camera: GSCamera 
        device: torch device
        bg_color: 
        kernel_size: Mip-Splatting kernel size

    Returns:
        rendered_depth: (1, H, W) median depth
        rendered_image: (3, H, W) RGB image
    """
    if bg_color is None:
        bg_color = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

    gaussians = SimpleGaussianModel(
        means=gaussian_means,
        opacities=gaussian_opacities,
        scales=gaussian_scales,
        rotations=gaussian_rotations,
        colors=gaussian_colors,
        device=device
    )
    import math
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    tanfovx = math.tan(gs_camera.FoVx * 0.5)
    tanfovy = math.tan(gs_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(gs_camera.image_height),
        image_width=int(gs_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=gs_camera.world_view_transform,
        projmatrix=gs_camera.full_proj_transform,
        sh_degree=0,
        campos=gs_camera.camera_center,
        prefiltered=False,
        require_coord=False,
        require_depth=True,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=False, device=device)
    scales, opacity = gaussians.get_scaling_n_opacity_with_3D_filter
    rotations = gaussians.get_rotation
    colors_precomp = gaussians.get_colors_precomp

    with torch.no_grad():
        rendered_image, radii, _, _, rendered_expected_depth, rendered_median_depth, rendered_alpha, rendered_normal = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )

    return rendered_median_depth, rendered_image


def update_gaussian_colors_from_novelty(novelty_values):
    """
    use novelty_values update Gaussian colors

    Args:
        novelty_values: 
            0 = unknown → white [1,1,1]
            1 = known → black [0,0,0]
    """
    inverted_values = 1.0 - novelty_values
    colors = inverted_values.unsqueeze(1).repeat(1, 3)
    return colors

# ==================== End RaDe-GS Integration ====================

def load_current_frame_perfect_depth(camera, device):
    current_frame_nb = camera.n_frames_captured - 1
    frame_path = os.path.join(camera.save_dir_path, str(current_frame_nb) + '.pt')

    
    frame_dict = torch.load(frame_path, map_location=device)
    
    return {
        'rgb': frame_dict['rgb'],           # (1, H, W, 3)
        'zbuf': frame_dict['zbuf'],         # (1, H, W, 1) 
        'mask': frame_dict['mask'],         # (1, H, W, 1)
        'R': frame_dict['R'],               # (1, 3, 3)
        'T': frame_dict['T'],               # (1, 3)
        'zfar': camera.zfar
    }

def apply_perfect_depth_simple(frame_data, device, use_error_mask=True):
    images = frame_data['rgb']
    zbuf = frame_data['zbuf'] 
    mask = frame_data['mask'].bool()
    R = frame_data['R']
    T = frame_data['T']
    
    # GT zbuf
    depth = torch.clamp(zbuf, min=0.5, max=750.0) 
    
    if use_error_mask:
        error_mask = mask
    else:
        error_mask = torch.ones_like(mask)
    
    return depth, mask, error_mask, R, T

dir_path = os.path.abspath(os.path.dirname(__file__))
# data_path = os.path.join(dir_path, "../../../../../../datasets/rgb")
data_path = os.path.join(dir_path, "../../data/scenes")
results_dir = os.path.join(dir_path, "../../results/scene_exploration")
weights_dir = os.path.join(dir_path, "../../weights/macarons")
configs_dir = os.path.join(dir_path, "../../configs/macarons")

def setup_test(params, model_path, device, verbose=True):
    # Create dataloader
    _, _, test_dataloader = get_dataloader(train_scenes=params.train_scenes,
                                           val_scenes=params.val_scenes,
                                           test_scenes=params.test_scenes,
                                           batch_size=1,
                                           ddp=False, jz=False,
                                           world_size=None, ddp_rank=None,
                                           data_path=params.data_path)
    print("\nThe following scenes will be used to test the model:")
    for batch, elem in enumerate(test_dataloader):
        print(elem['scene_name'][0])

    # Create model
    macarons = load_pretrained_macarons(pretrained_model_path=params.pretrained_model_path,
                                        device=device, learn_pose=params.learn_pose)


    trained_weights = torch.load(model_path, map_location=device, weights_only=False)
    macarons.load_state_dict(trained_weights["model_state_dict"], ddp=True)  # todo: replace by params.ddp
    depth_losses = np.array(trained_weights["depth_losses"])
    depth_losses_per_epoch = (depth_losses[::2] + depth_losses[1::2]) / 2
    # depth_losses_per_epoch = depth_losses
    print("\nModel name:", model_path)
    print("\nThe model has", (count_parameters(macarons.depth) + count_parameters(macarons.scone)) / 1e6,
          "trainable parameters.")
    print("It has been trained for", trained_weights["epoch"], "epochs.")
    print("The loss was:", depth_losses_per_epoch[-1], depth_losses_per_epoch[-1] * 3 / 4)
    print(params.n_alpha, "additional frames are used for depth prediction.")

    # Creating memory
    print("\nUsing memory folders", params.memory_dir_name)
    scene_memory_paths = []
    for scene_name in params.test_scenes:
        scene_path = os.path.join(test_dataloader.dataset.data_path, scene_name)
        scene_memory_path = os.path.join(scene_path, params.memory_dir_name)
        scene_memory_paths.append(scene_memory_path)
    memory = Memory(scene_memory_paths=scene_memory_paths, n_trajectories=params.n_memory_trajectories,
                    current_epoch=0, verbose=verbose)

    return test_dataloader, macarons, memory


def setup_test_scene(params,
                     mesh,
                     settings,
                     mirrored_scene,
                     device,
                     mirrored_axis=None,
                     surface_scene_feature_dim=1,
                     test_resolution=0.05,
                     covered_scene_feature_dim=1):
    """
    Setup the different scene objects used for prediction and performance evaluation.

    :param params:
    :param mesh:
    :param settings:
    :param device:
    :param is_master:
    :return:
    """

    # Initialize gt_scene: we use this scene to store gt surface points to evaluate the performance of the model.
    # This scene is not used for supervision during training, since the model is self-supervised from RGB data
    # captured in real-time.
    gt_scene = Scene(x_min=settings.scene.x_min,
                     x_max=settings.scene.x_max,
                     grid_l=settings.scene.grid_l,
                     grid_w=settings.scene.grid_w,
                     grid_h=settings.scene.grid_h,
                     cell_capacity=params.surface_cell_capacity,
                     cell_resolution=test_resolution * params.scene_scale_factor,
                     n_proxy_points=params.n_proxy_points,
                     device=device,
                     view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                     feature_dim=3,
                     mirrored_scene=mirrored_scene,
                     mirrored_axis=mirrored_axis)  # We use colors as features

    covered_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=test_resolution * params.scene_scale_factor,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=covered_scene_feature_dim,
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)  # We use colors as features

    # We fill gt_scene with points sampled on the surface of the ground truth mesh
    gt_surface, gt_normals, gt_surface_colors = get_scene_gt_surface(gt_scene=gt_scene,
                                                         verts=mesh.verts_list()[0],
                                                         faces=mesh.faces_list()[0],
                                                         n_surface_points=params.n_gt_surface_points,
                                                         return_colors=True,
                                                         mesh=mesh)
    gt_scene.fill_cells(gt_surface, features=gt_surface_colors)

    # Initialize surface_scene: we store in this scene the surface points computed by the depth model from RGB images
    surface_scene = Scene(x_min=settings.scene.x_min,
                          x_max=settings.scene.x_max,
                          grid_l=settings.scene.grid_l,
                          grid_w=settings.scene.grid_w,
                          grid_h=settings.scene.grid_h,
                          cell_capacity=params.surface_cell_capacity,
                          cell_resolution=None,
                          n_proxy_points=params.n_proxy_points,
                          device=device,
                          view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                          feature_dim=surface_scene_feature_dim,  # We use visibility history as features
                          mirrored_scene=mirrored_scene,
                          mirrored_axis=mirrored_axis)

    # Initialize proxy_scene: we store in this scene the proxy points
    proxy_scene = Scene(x_min=settings.scene.x_min,
                        x_max=settings.scene.x_max,
                        grid_l=settings.scene.grid_l,
                        grid_w=settings.scene.grid_w,
                        grid_h=settings.scene.grid_h,
                        cell_capacity=params.proxy_cell_capacity,
                        cell_resolution=params.proxy_cell_resolution,
                        n_proxy_points=params.n_proxy_points,
                        device=device,
                        view_state_n_elev=params.view_state_n_elev, view_state_n_azim=params.view_state_n_azim,
                        feature_dim=1,  # We use proxy points indices as features
                        mirrored_scene=mirrored_scene,
                        score_threshold=params.score_threshold,
                        mirrored_axis=mirrored_axis)
    proxy_scene.initialize_proxy_points()

    return gt_scene, covered_scene, surface_scene, proxy_scene


def setup_test_camera(params,
                      mesh, intersector, start_cam_idx,
                      settings,
                      occupied_pose_data,
                      device,
                      training_frames_path,
                      mirrored_scene=False,
                      mirrored_axis=None):
    """
    Setup the camera used for prediction.

    :param params:
    :param mesh:
    :param start_cam_idx:
    :param settings:
    :param occupied_pose_data:
    :param device:
    :param training_frames_path:
    :return:
    """
    # Default camera to initialize the renderer
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=200000
                                )

    # Initialize camera
    camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                    pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                    pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                    n_interpolation_steps=params.n_interpolation_steps, zfar=params.zfar,
                    renderer=renderer,
                    device=device,
                    contrast_factor=settings.camera.contrast_factor,
                    gathering_factor=params.gathering_factor,
                    occupied_pose_data=occupied_pose_data,
                    save_dir_path=training_frames_path,
                    mirrored_scene=mirrored_scene,
                    mirrored_axis=mirrored_axis)  # Change or remove this path during inference or test


    # Select a random, valid camera pose as starting pose
    camera.initialize_camera(start_cam_idx=start_cam_idx)

    # Capture initial image
    camera.capture_image(mesh)

    return camera


def compute_magician_trajectory(params, macarons, camera, gt_scene, surface_scene,
                           proxy_scene, covered_scene, mesh, intersector, device, settings,
                           test_resolution=0.05, use_perfect_depth_map=False,
                           compute_collision=False):

    macarons.eval()

    # --- exp3 read-only hallucination dump hook (env-gated; no effect on planning) ---
    _dump_dir = os.environ.get("MAGICIAN_DUMP_DIR")
    _dump_steps = set()
    if _dump_dir:
        os.makedirs(_dump_dir, exist_ok=True)
        _dump_steps = set(int(x) for x in os.environ.get("MAGICIAN_DUMP_STEPS", "0,5,10,20,40,70,100").split(",") if x.strip())

    # --- exp4 read-only gain-attribution probe (env-gated; no effect on planning) ---
    # For the viewpoint the planner CHOOSES each step, decompose its coverage-gain into
    # phantom (imagined points far from GT surface) vs real, to test whether hallucination
    # distorts the planner's view-selection. MAGICIAN_GAIN_PROBE = output csv path.
    _probe_path = os.environ.get("MAGICIAN_GAIN_PROBE")
    _probe_eps_frac = float(os.environ.get("MAGICIAN_GAIN_PROBE_EPS", "0.03"))
    _probe_rows = []
    _probe_tree = None
    _probe_eps = None
    # Optional: also save the chosen-view rendered gain maps (total / phantom) at selected steps,
    # so the gain decomposition can be viewed as images (MAGICIAN_PROBE_IMG_DIR = output dir).
    _probe_img_dir = os.environ.get("MAGICIAN_PROBE_IMG_DIR")
    _probe_img_steps = set()
    if _probe_img_dir:
        os.makedirs(_probe_img_dir, exist_ok=True)
        _probe_img_steps = set(int(x) for x in os.environ.get(
            "MAGICIAN_PROBE_IMG_STEPS", "0,1,2,3,4,5,6,7,8,9,10").split(",") if x.strip())

    # --- exp5 Oracle counterfactual intervention (env-gated; no effect when MAGICIAN_ORACLE_DROP unset) ---
    # GO/NO-GO upper bound: use GT to tag phantom imagined points (dist to GT surface > eps) and
    # multiply their opacity by (1-lambda) in the beam-search CANDIDATE gain render, so hallucinated
    # structure no longer inflates any candidate's coverage-gain. lambda=1 => hard drop (main).
    # Only the candidate gain-scoring render is modified (the planning objective); novelty bookkeeping
    # stays at baseline. This is an Oracle (uses GT) => ceiling for our learned method, not deployable.
    _oracle_lam = float(os.environ.get("MAGICIAN_ORACLE_DROP", "0") or "0")
    _oracle_on = _oracle_lam > 0.0
    _oracle_tree = None
    _oracle_eps = None

    # --- exp6 verify-steering (env-gated; no effect when MAGICIAN_VERIFY_MODE unset) ---
    # STEERING (not substitution): the map stays hallucinated; we ADD a verify-gain term to the beam
    # value that pulls the drone toward structure it should go CONFIRM, so real observation then
    # deletes phantoms / covers missed structure. Uses GT to define verify targets => Oracle ceiling.
    #   oracle-fp   : verify = imagined points far from GT surface (confident phantoms, occ>0.5)
    #   oracle-both : + FN beacons (missed GT surface) injected into the VERIFY render only (ch0=0 =>
    #                 they steer but do NOT fake coverage; collision set is untouched => no fake walls)
    # beam value = total_coverage_gain + lambda * total_verify_gain.  Coverage render is left byte-
    # identical to baseline (verify uses a SEPARATE render), so any effect is purely the steering term.
    #   support     : DEPLOYABLE (no GT) — verify = imagined point far from ACCUMULATED OBSERVED surface
    #                 ("predicted-occupied but observation-unsupported"). No FN beacons (no GT to know misses).
    _verify_mode = os.environ.get("MAGICIAN_VERIFY_MODE")   # None | 'oracle-fp' | 'oracle-both' | 'support'
    _verify_on = _verify_mode in ("oracle-fp", "oracle-both", "support")
    _verify_lam = float(os.environ.get("MAGICIAN_VERIFY_LAMBDA", "1.0") or "1.0")
    _verify_tree = None
    _verify_gt_np = None
    _verify_eps = None
    _verify_beacon_cap = int(os.environ.get("MAGICIAN_VERIFY_BEACON_CAP", "20000"))
    _verify_beacon_opac = float(os.environ.get("MAGICIAN_VERIFY_BEACON_OPAC", "0.15"))
    _verify_trace_path = os.environ.get("MAGICIAN_VERIFY_TRACE")
    _verify_trace = []
    if _verify_on:
        print(f"[VERIFY] mode={_verify_mode} lambda={_verify_lam} beacon_cap={_verify_beacon_cap}")

    # --- exp7 self-correcting coverage (env-gated; no effect when MAGICIAN_SC_MODE unset) ---
    # Deployable (no GT). Goal = efficient coverage; verification EMERGES instead of being a bolt-on
    # verify-gain term (exp6). Per step: tag imagined points the drone's OWN observations have seen
    # through (carve_ratio = proxy_n_behind_depth/proxy_n_inside_fov low, after >=K views) as suspect,
    # then (1) fade them out of the candidate gain render via render_opacities (no fake gain) and
    # (2) drop them from the imagined-point collision set (no fake walls). Unobserved predictions keep
    # full gain => exploration preserved (this maturity gate is what exp5's fix-FP lacked).
    # SIGNAL NOTE (smoke-verified): a carve_ratio threshold is INERT here — the pipeline's own
    # carving already excludes ratio<0.95 points from occ scoring (macarons_utils occ_mask), so
    # surviving imagined points all have ratio>=0.95. The surviving phantoms are the never-caught
    # ones; the deployable tag for them is "MATURE yet UNSUPPORTED": looked toward >=K times but
    # still no observed surface within eps of the prediction. The maturity gate is what separates
    # this from exp6's failed 'support' (never-viewed regions stay immature => not flagged =>
    # exploration preserved).
    _sc_on = os.environ.get("MAGICIAN_SC_MODE") is not None
    _sc_signal = os.environ.get("MAGICIAN_SC_SIGNAL", "unsupported")  # 'unsupported' | 'ratio' (ablation)
    _sc_k = int(os.environ.get("MAGICIAN_SC_K", "2"))            # maturity gate: min #views in FOV
    _sc_tau = float(os.environ.get("MAGICIAN_SC_TAU", "0.6"))    # ratio signal only: carve_ratio < tau
    _sc_lam = float(os.environ.get("MAGICIAN_SC_LAMBDA", "1.0")) # gain-render opacity discount
    _sc_collide = os.environ.get("MAGICIAN_SC_COLLIDE", "1") not in ("0", "false")
    _sc_trace_path = os.environ.get("MAGICIAN_SC_TRACE")
    _sc_trace = []
    _sc_proxy_tree = None   # exact-match cKDTree over proxy_scene.proxy_points (built once)

    # --- stage_3 Phase 0: observation-side Beta posterior (env-gated; inert when unset) ---
    # SCCov replaces a belief with a binary verdict using two hand-set constants (K, eps).
    # Those two constants are a hard-threshold approximation of one posterior, so write the
    # posterior instead. Beta is conjugate to Bernoulli => evidence ADDS:
    #     alpha = c*p_net + n_surface
    #     beta  = c*(1-p_net) + n_free + w_occ*n_occluded
    # with c the prior pseudo-count ("the network's opinion is worth c measurements").
    # The three counters come free from carving (macarons_utils.update_proxy_supervision_occ).
    #
    # WHY OCCLUSION IS NEGATIVE EVIDENCE, not zero evidence: the quantity the planner needs
    # is not "is there matter here" but "is this a VISIBLE SURFACE point worth flying to".
    # A point that is repeatedly inside the frustum yet never on the surface and never carved
    # is always strictly behind observed geometry, i.e. not on the visible surface, while the
    # gain render treats it as if it were. Counting occlusion as beta is also what makes the
    # maturity gate EMERGENT: never-looked points have zero evidence, so their posterior is
    # the prior and they keep full gain (exploration preserved by construction), whereas
    # looked-at-yet-never-confirmed points decay smoothly. With w_occ=0 the two cases become
    # indistinguishable and the method degenerates to base -- kept as a negative control.
    #
    # NOTE on n_free: the pipeline's own occ_mask (macarons_utils:1601) already drops points
    # with carve ratio < 0.95 BEFORE scoring, so n_free is near-zero on the surviving imagined
    # set. It is included for completeness but n_occluded is the load-bearing channel here.
    # This is the same reason the exp7 carve-ratio ablation was provably inert.
    _beta_on = os.environ.get("MAGICIAN_BETA") is not None
    _beta_c = float(os.environ.get("MAGICIAN_BETA_C", "5.0"))         # prior pseudo-count; ->inf == base
    _beta_ws = float(os.environ.get("MAGICIAN_BETA_WS", "1.0"))       # weight of n_surface  (alpha)
    _beta_wf = float(os.environ.get("MAGICIAN_BETA_WF", "1.0"))       # weight of n_free     (beta)
    _beta_wocc = float(os.environ.get("MAGICIAN_BETA_WOCC", "0.25"))  # weight of n_occluded (beta); 0 == control
    _beta_kappa = float(os.environ.get("MAGICIAN_BETA_KAPPA", "0.0")) # lower credible bound mu - kappa*sigma
    _beta_taucol = float(os.environ.get("MAGICIAN_BETA_TAUCOL", "0.5"))  # collision LCB threshold
    _beta_gain = os.environ.get("MAGICIAN_BETA_GAIN", "1") not in ("0", "false")
    _beta_collide = os.environ.get("MAGICIAN_BETA_COLLIDE", "1") not in ("0", "false")
    _beta_trace_path = os.environ.get("MAGICIAN_BETA_TRACE")
    _beta_trace = []
    _beta_dump_dir = os.environ.get("MAGICIAN_BETA_DUMP")   # D-A diagnostic: per-step counter dump
    _beta_dump_steps = set(int(s) for s in os.environ.get(
        "MAGICIAN_BETA_DUMP_STEPS", "0,5,10,20,40,70,99").split(",") if s.strip())
    if _beta_dump_dir:
        os.makedirs(_beta_dump_dir, exist_ok=True)

    # --- stage_2 A1: no-imagination control (env-gated; no effect when unset) ---
    # Replace the imagined set with the ACCUMULATED OBSERVED surface (full_pc): the planner
    # believes only what it has actually seen — no extrapolation. Tests whether imagination
    # is load-bearing (expected: novelty≈0 everywhere → no gain signal → collapse).
    _noimag = os.environ.get("MAGICIAN_NOIMAG") == "1"
    if _noimag:
        print("[NOIMAG] imagination OFF: imagined set <- observed surface only")

    # --- stage_2 D1: depth-noise robustness (env-gated; no effect when unset) ---
    # Multiplicative Gaussian noise on the depth map (sigma as a fraction of depth), applied
    # before back-projection AND carving — dirties both the observed pc (support test input)
    # and the carve counters. Tests whether sccov survives non-perfect depth.
    _depth_noise = float(os.environ.get("MAGICIAN_DEPTH_NOISE", "0") or "0")
    if _depth_noise > 0:
        print(f"[DEPTH-NOISE] multiplicative sigma={_depth_noise}")
    if _sc_on and os.environ.get("MAGICIAN_ORACLE_MODE"):
        print("[SCCOV] WARNING: MAGICIAN_ORACLE_MODE swaps the imagined set -> proxy identity breaks; disabling SC")
        _sc_on = False
    if _sc_on:
        print(f"[SCCOV] on: K={_sc_k} tau={_sc_tau} lambda={_sc_lam} collide={int(_sc_collide)}")
    if _beta_on and os.environ.get("MAGICIAN_ORACLE_MODE"):
        print("[BETA] WARNING: MAGICIAN_ORACLE_MODE swaps the imagined set -> proxy identity breaks; disabling BETA")
        _beta_on = False
    if _beta_on and _sc_on:
        print("[BETA] WARNING: SCCov is also on; the two levers would compose. Disabling SCCov.")
        _sc_on = False
    if _beta_on:
        print(f"[BETA] on: c={_beta_c} ws={_beta_ws} wf={_beta_wf} wocc={_beta_wocc} "
              f"kappa={_beta_kappa} taucol={_beta_taucol} gain={int(_beta_gain)} collide={int(_beta_collide)}")

    # compute scene_scales
    scene_bbox_x = settings.scene.x_max[0] - settings.scene.x_min[0]
    scene_bbox_y = settings.scene.x_max[1] - settings.scene.x_min[1]
    scene_bbox_z = settings.scene.x_max[2] - settings.scene.x_min[2]
    scene_scale = (scene_bbox_x + scene_bbox_y + scene_bbox_z) / 3.0
    print(f"Scene scale computed: {scene_scale:.2f} (bbox: x={scene_bbox_x:.2f}, y={scene_bbox_y:.2f}, z={scene_bbox_z:.2f})")

    full_pc = torch.zeros(0, 3, device=device)
    full_pc_colors = torch.zeros(0, 3, device=device)
    full_pc_idx = torch.zeros(0, 1, device=device)
    coverage_evolution = []
    pose_i = 0
    
    def process_current_frame():
        current_frame = load_current_frame_perfect_depth(camera, device)
        depth, mask, error_mask, R, T = apply_perfect_depth_simple(current_frame, device)
        if _depth_noise > 0:  # stage_2 D1: corrupt depth before back-projection and carving
            depth = depth * (1.0 + _depth_noise * torch.randn_like(depth))
        
        fov_camera = camera.get_fov_camera_from_RT(R_cam=R, T_cam=T)
        X_cam = fov_camera.get_camera_center() 
        
        part_pc, part_pc_features = camera.compute_partial_point_cloud(
            depth=depth, mask=(mask * error_mask).bool(), images=current_frame['rgb'],
            fov_cameras=fov_camera,
            gathering_factor=params.gathering_factor * 2,
            fov_range=params.sensor_range
        )
        
        fov_proxy_points, fov_proxy_mask = camera.get_points_in_fov(
            proxy_scene.proxy_points, return_mask=True,
            fov_camera=None, fov_range=params.sensor_range
        )
        
        sgn_dists = None
        if fov_proxy_mask.any():
            sgn_dists = camera.get_signed_distance_to_depth_maps(
                pts=fov_proxy_points, depth_maps=depth,
                mask=mask, fov_camera=None
            )
        
        return {
            'part_pc': part_pc,
            'part_pc_features': part_pc_features,
            'fov_proxy_points': fov_proxy_points,
            'fov_proxy_mask': fov_proxy_mask,
            'sgn_dists': sgn_dists,
            'X_cam': X_cam,  
            'current_frame': current_frame
        }
            

    _max_steps_env = os.environ.get("MAGICIAN_MAX_STEPS")  # smoke-test cap; no effect when unset
    _max_steps = int(_max_steps_env) if _max_steps_env else None
    while pose_i <= params.n_poses_in_trajectory:
        if _max_steps is not None and pose_i >= _max_steps:
            print(f"[MAGICIAN_MAX_STEPS] stopping at pose {pose_i}")
            break
        if pose_i % 10 == 0:
            print("Processing pose", str(pose_i) + "...")
        
        camera.fov_camera_0 = camera.fov_camera

        if pose_i > 0 and pose_i % params.recompute_surface_every_n_loop == 0:
            print("Recomputing surface...")
            fill_surface_scene(surface_scene, full_pc,
                               random_sampling_max_size=params.n_gt_surface_points,
                               min_n_points_per_cell_fill=3,
                               progressive_fill=params.progressive_fill,
                               max_n_points_per_fill=params.max_points_per_progressive_fill)

        frame_data = process_current_frame()

        # Unpdate scene
        part_pc_features = torch.zeros(len(frame_data['part_pc']), 1, device=device)
        covered_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        surface_scene.fill_cells(frame_data['part_pc'], features=part_pc_features)
        full_pc = torch.vstack((full_pc, frame_data['part_pc']))
        full_pc_colors = torch.vstack((full_pc_colors, frame_data['part_pc_features']))
        part_pc_idx = torch.full((frame_data['part_pc'].shape[0], 1), pose_i, device=device)
        full_pc_idx = torch.vstack((full_pc_idx, part_pc_idx))

        if frame_data['fov_proxy_mask'].any():
            fov_proxy_indices = proxy_scene.get_proxy_indices_from_mask(frame_data['fov_proxy_mask'])
            proxy_scene.fill_cells(frame_data['fov_proxy_points'], 
                                 features=fov_proxy_indices.view(-1, 1))
            
            proxy_scene.update_proxy_view_states(
                camera, frame_data['fov_proxy_mask'],
                signed_distances=frame_data['sgn_dists'],
                distance_to_surface=None, 
                X_cam=frame_data['X_cam']  
            )
            
            proxy_scene.update_proxy_supervision_occ(
                frame_data['fov_proxy_mask'], frame_data['sgn_dists'], 
                tol=params.carving_tolerance
            )
            proxy_scene.update_proxy_out_of_field(frame_data['fov_proxy_mask'])

        surface_scene.set_all_features_to_value(value=1.)

        # Compute coverage gain for evaulation
        current_coverage = gt_scene.scene_coverage(
            covered_scene, surface_epsilon=2 * test_resolution * params.scene_scale_factor
        )
        if pose_i % 5 == 0:
            print("==========current coverage:", current_coverage)
        current_cov = current_coverage[0].item() if current_coverage[0] != 0. else 0.
        coverage_evolution.append(current_cov / settings.scene.visibility_ratio)

        # Occupancy field prediction
        with torch.no_grad():
            X_world, view_harmonics, occ_probs = compute_scene_occupancy_probability_field(
                params, macarons.scone, camera, surface_scene, proxy_scene, device
            )
        # We only keep the points with occupancy value larger than 0.5
        filtered_X_world = X_world[occ_probs.squeeze() > 0.5]
        n_points = filtered_X_world.shape[0]
        gaussian_means = filtered_X_world  # (N, 3)
        occ_values = occ_probs[occ_probs.squeeze() > 0.5]

        # --- stage_2 A1 (MAGICIAN_NOIMAG=1): believe only the observed surface ---
        if _noimag:
            _obs_pc = full_pc
            if _obs_pc.shape[0] > 120000:
                _sel = np.random.RandomState(0).choice(_obs_pc.shape[0], 120000, replace=False)
                _obs_pc = _obs_pc[torch.tensor(_sel, dtype=torch.long, device=device)]
            filtered_X_world = _obs_pc.detach().clone()
            n_points = filtered_X_world.shape[0]
            gaussian_means = filtered_X_world
            occ_values = torch.ones(n_points, 1, device=device)
            if pose_i % 10 == 0:
                print(f"[NOIMAG] step {pose_i}: imagined set = {n_points} observed points")

        # --- exp5 Oracle DECOMPOSITION (env-gated MAGICIAN_ORACLE_MODE; no effect when unset) ---
        # Swap the imagined point set using GT, to separate the two error types of the occupancy net:
        #   fp   = drop false positives  (imagined-occupied in EMPTY GT space)      -> keep only TP
        #   fn   = add false negatives   (real GT surface the net MISSED, occ<0.5)  -> imagined + missed
        #   both = use the GT surface itself (perfect occupancy)                    -> TP + FN, no FP
        # eps = 3% of scene diagonal (same phantom metric as exp3/4). Uses GT => Oracle upper bound.
        _oracle_mode = os.environ.get("MAGICIAN_ORACLE_MODE")
        if _oracle_mode in ("fp", "fn", "both"):
            from scipy.spatial import cKDTree as _mode_ckt
            if _oracle_tree is None:
                _ogt = gt_scene.return_entire_pt_cloud(return_features=False).detach().cpu().numpy()
                if len(_ogt) > 120000:
                    _ogt = _ogt[np.random.RandomState(0).choice(len(_ogt), 120000, replace=False)]
                _oracle_gt_np = _ogt
                _oracle_tree = _mode_ckt(_ogt)
                _oracle_eps = _probe_eps_frac * float(np.linalg.norm(_ogt.max(0) - _ogt.min(0)))
            _img_np = filtered_X_world.detach().cpu().numpy()
            _occ_np = occ_values.detach().cpu().numpy().reshape(-1, 1)
            if _oracle_mode == "both":
                new_np = _oracle_gt_np
                new_occ = np.ones((len(new_np), 1), dtype="float32")
            elif _oracle_mode == "fp":
                keep = (_oracle_tree.query(_img_np, k=1)[0] <= _oracle_eps) if len(_img_np) else np.zeros(0, bool)
                new_np, new_occ = _img_np[keep], _occ_np[keep]
            else:  # fn : keep all imagined (incl FP) + add missed real structure
                if len(_img_np):
                    _it = _mode_ckt(_img_np if len(_img_np) <= 120000 else
                                    _img_np[np.random.RandomState(1).choice(len(_img_np), 120000, replace=False)])
                    missed = _oracle_gt_np[_it.query(_oracle_gt_np, k=1)[0] > _oracle_eps]
                    new_np = np.vstack([_img_np, missed]); new_occ = np.vstack([_occ_np, np.ones((len(missed), 1), "float32")])
                else:
                    new_np = _oracle_gt_np; new_occ = np.ones((len(_oracle_gt_np), 1), "float32")
            filtered_X_world = torch.tensor(new_np, dtype=torch.float32, device=device)
            occ_values = torch.tensor(new_occ, dtype=torch.float32, device=device).view(-1, 1)
            n_points = filtered_X_world.shape[0]
            gaussian_means = filtered_X_world
            if pose_i % 10 == 0:
                print(f"[ORACLE_MODE={_oracle_mode}] step {pose_i}: n_points -> {n_points}")

        # Convert occupancy field to Imagined Gaussians
        gaussian_opacities = occ_values    # (N, 1) 
        gaussian_scales = torch.ones(n_points, 3, device=device) * (0.7154/2)  
        gaussian_rotations = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(n_points, 1) 
        novelty_values = torch.zeros(n_points, device=device)  # (N,)
        gaussian_colors = update_gaussian_colors_from_novelty(novelty_values)  # (N, 3)

        # --- exp7 Step 0: diagnose suspect (seen-through) imagined points from own observations ---
        # Imagined points are exact rows of the persistent proxy cloud (sampled once), so an
        # exact-match KDTree lookup recovers their stable proxy index; per-point carve counters
        # (updated for free every frame in update_proxy_supervision_occ) then give
        # carve_ratio = n_behind_depth/n_inside_fov. suspect = mature (>=K views) AND seen through
        # (ratio < tau). Unobserved/immature points are NOT touched (exploration preserved).
        _sc_suspect = None
        if _sc_on and n_points > 0:
            from scipy.spatial import cKDTree as _sc_ckt
            if _sc_proxy_tree is None:
                _sc_proxy_tree = _sc_ckt(proxy_scene.proxy_points.detach().cpu().numpy())
            _sc_d, _sc_idx = _sc_proxy_tree.query(filtered_X_world.detach().cpu().numpy(), k=1)
            if pose_i == 0:
                assert float(_sc_d.max() if len(_sc_d) else 0.0) < 1e-3, \
                    f"[SCCOV] proxy identity broken: max exact-match dist {_sc_d.max():.4f}"
            _pidx = torch.tensor(_sc_idx, dtype=torch.long, device=device)
            _n_in = proxy_scene.proxy_n_inside_fov[_pidx].view(-1)
            _n_behind = proxy_scene.proxy_n_behind_depth[_pidx].view(-1)
            _sc_ratio = _n_behind / _n_in.clamp(min=1.0)
            if _sc_signal == "ratio":
                # ablation: provably inert (surviving imagined points all have ratio>=0.95)
                _sc_suspect = (_n_in >= _sc_k) & (_sc_ratio < _sc_tau)
            else:
                # 'unsupported': mature (>=K views toward it) yet no observed surface within eps
                # of the predicted point. eps from the a-priori scene bbox (GT-independent),
                # same recipe as the exp6 support mode.
                _sc_eps = _probe_eps_frac * float(np.linalg.norm(
                    [float(scene_bbox_x), float(scene_bbox_y), float(scene_bbox_z)]))
                _obs_np = full_pc.detach().cpu().numpy()
                if len(_obs_np) > 120000:
                    _obs_np = _obs_np[np.random.RandomState(0).choice(len(_obs_np), 120000, replace=False)]
                if len(_obs_np):
                    _d_obs = _sc_ckt(_obs_np).query(filtered_X_world.detach().cpu().numpy(), k=1)[0]
                    _unsup = torch.tensor((_d_obs > _sc_eps), dtype=torch.bool, device=device)
                else:
                    _unsup = torch.ones(n_points, dtype=torch.bool, device=device)
                _sc_suspect = (_n_in >= _sc_k) & _unsup
            _n_suspect = int(_sc_suspect.sum().item())
            _n_mature = int((_n_in >= _sc_k).sum().item())
            if _sc_trace_path is not None:
                _sc_trace.append((pose_i, n_points, _n_mature, _n_suspect))
            if pose_i % 10 == 0:
                print(f"[SCCOV] step {pose_i}: n_imagined={n_points} n_mature={_n_mature} n_suspect={_n_suspect}")

        # --- stage_3 Phase 0: Beta posterior over the imagined set (and the D-A dump) ---
        # Runs when MAGICIAN_BETA is on, OR when only MAGICIAN_BETA_DUMP is set (read-only
        # diagnostic on an otherwise untouched baseline run).
        _beta_mu = None
        _beta_lcb = None
        _beta_dump_payload = None
        if (_beta_on or _beta_dump_dir) and n_points > 0:
            from scipy.spatial import cKDTree as _b_ckt
            if _sc_proxy_tree is None:
                _sc_proxy_tree = _b_ckt(proxy_scene.proxy_points.detach().cpu().numpy())
            _b_d, _b_i = _sc_proxy_tree.query(filtered_X_world.detach().cpu().numpy(), k=1)
            if pose_i == 0:
                assert float(_b_d.max() if len(_b_d) else 0.0) < 1e-3, \
                    f"[BETA] proxy identity broken: max exact-match dist {_b_d.max():.4f}"
            _bidx = torch.tensor(_b_i, dtype=torch.long, device=device)
            _n_in_b = proxy_scene.proxy_n_inside_fov[_bidx].view(-1)
            _n_surf = proxy_scene.proxy_n_surface[_bidx].view(-1)
            _n_free = proxy_scene.proxy_n_free[_bidx].view(-1)
            _n_occ = proxy_scene.proxy_n_occluded[_bidx].view(-1)

            _p_net = occ_values.view(-1).clamp(0.0, 1.0)   # GELU output, not a probability: clamp
            _b_alpha = _beta_c * _p_net + _beta_ws * _n_surf
            _b_beta = _beta_c * (1.0 - _p_net) + _beta_wf * _n_free + _beta_wocc * _n_occ
            _b_S = (_b_alpha + _b_beta).clamp(min=1e-6)
            _beta_mu = _b_alpha / _b_S
            _b_sigma = torch.sqrt((_b_alpha * _b_beta) / (_b_S * _b_S * (_b_S + 1.0)))
            _beta_lcb = (_beta_mu - _beta_kappa * _b_sigma).clamp(0.0, 1.0)

            if _beta_trace_path is not None:
                _beta_trace.append((pose_i, n_points,
                                    float(_n_in_b.mean()), float(_n_surf.mean()),
                                    float(_n_free.mean()), float(_n_occ.mean()),
                                    float(_p_net.mean()), float(_beta_mu.mean()),
                                    float(_b_sigma.mean()),
                                    int((_beta_lcb <= _beta_taucol).sum())))
            if pose_i % 10 == 0:
                print(f"[BETA] step {pose_i}: n_imagined={n_points} "
                      f"mean n_surf={float(_n_surf.mean()):.2f} n_free={float(_n_free.mean()):.2f} "
                      f"n_occ={float(_n_occ.mean()):.2f} | p={float(_p_net.mean()):.3f} "
                      f"-> mu={float(_beta_mu.mean()):.3f} sigma={float(_b_sigma.mean()):.3f} "
                      f"below_tau={int((_beta_lcb <= _beta_taucol).sum())}")

            # D-A diagnostic dump: everything needed to score the signal offline against GT.
            if _beta_dump_dir and pose_i in _beta_dump_steps:
                import numpy as _np
                _d = dict(
                    step=_np.int32(pose_i),
                    imagined=filtered_X_world.detach().cpu().numpy().astype(_np.float32),
                    pidx=_b_i.astype(_np.int32),
                    occ=occ_values.view(-1).detach().cpu().numpy().astype(_np.float32),  # RAW (unclamped)
                    n_inside_fov=_n_in_b.detach().cpu().numpy().astype(_np.int16),
                    n_surface=_n_surf.detach().cpu().numpy().astype(_np.int16),
                    n_free=_n_free.detach().cpu().numpy().astype(_np.int16),
                    n_occluded=_n_occ.detach().cpu().numpy().astype(_np.int16),
                )
                if pose_i == min(_beta_dump_steps):   # GT surface is static: store it once
                    _d["gt_surface"] = gt_scene.return_entire_pt_cloud(
                        return_features=False).detach().cpu().numpy().astype(_np.float32)
                    _d["bbox"] = _np.array([float(scene_bbox_x), float(scene_bbox_y),
                                            float(scene_bbox_z)], dtype=_np.float32)
                # Held until novelty_values is computed further down: only NOVEL imagined
                # points produce coverage gain, so novelty is the population that matters
                # (exp4: phantoms are ~10% of imagined points but 33-36% of the gain, i.e.
                # they are strongly enriched among the novel ones).
                _beta_dump_payload = _d

        # --- exp6 verify-steering: build this step's verify render substrate (env-gated) ---
        _verify_active = _verify_on and n_points > 0
        if _verify_active:
            from scipy.spatial import cKDTree as _v_ckt
            _vimg = filtered_X_world.detach().cpu().numpy()
            _use_gt = _verify_mode in ("oracle-fp", "oracle-both")
            if _use_gt:
                # Oracle: tag imagined points against the GT surface (static -> cache the tree)
                if _verify_tree is None:
                    _vg = gt_scene.return_entire_pt_cloud(return_features=False).detach().cpu().numpy()
                    if len(_vg) > 120000:
                        _vg = _vg[np.random.RandomState(0).choice(len(_vg), 120000, replace=False)]
                    _verify_gt_np = _vg
                    _verify_tree = _v_ckt(_vg)
                    _verify_eps = _probe_eps_frac * float(np.linalg.norm(_vg.max(0) - _vg.min(0)))
                _ref_tree, _ref_eps = _verify_tree, _verify_eps
            else:
                # support (DEPLOYABLE, no GT): tag against the ACCUMULATED OBSERVED surface (grows -> rebuild).
                # eps from the a-priori scene bbox (GT-independent). full_pc = accumulated back-projected obs.
                _obs = full_pc.detach().cpu().numpy()
                if len(_obs) > 120000:
                    _obs = _obs[np.random.RandomState(0).choice(len(_obs), 120000, replace=False)]
                _ref_eps = _probe_eps_frac * float(np.linalg.norm(
                    [float(scene_bbox_x), float(scene_bbox_y), float(scene_bbox_z)]))
                _ref_tree = _v_ckt(_obs) if len(_obs) else None
            # verify score on imagined points: far from reference surface = "worth verifying"
            if _ref_tree is not None and len(_vimg):
                _vfp = (_ref_tree.query(_vimg, k=1)[0] > _ref_eps).astype("float32")  # (N,)
            else:
                _vfp = np.ones(len(_vimg), dtype="float32")  # nothing observed yet => all unverified
            _verify_fp_t = torch.tensor(_vfp, device=device)
            _n_phantom = int(_vfp.sum())
            # oracle-both only: inject FN beacons = missed GT surface (support has no GT => no beacons)
            if _verify_mode == "oracle-both":
                _it = _v_ckt(_vimg if len(_vimg) <= 120000 else
                             _vimg[np.random.RandomState(1).choice(len(_vimg), 120000, replace=False)])
                _missed = _verify_gt_np[_it.query(_verify_gt_np, k=1)[0] > _verify_eps]
                if len(_missed) > _verify_beacon_cap:
                    _missed = _missed[np.random.RandomState(2).choice(len(_missed), _verify_beacon_cap, replace=False)]
                _n_beacon = len(_missed)
            else:
                _missed = np.zeros((0, 3), dtype="float32"); _n_beacon = 0
            # augmented substrate used ONLY for the verify render (collision/coverage sets untouched)
            if _n_beacon > 0:
                _beacon_t = torch.tensor(_missed, dtype=torch.float32, device=device)
                _vm_means = torch.vstack([filtered_X_world, _beacon_t])
                _vm_opac = torch.vstack([gaussian_opacities,
                                         torch.full((_n_beacon, 1), _verify_beacon_opac, device=device)])
            else:
                _vm_means = filtered_X_world
                _vm_opac = gaussian_opacities
            _vm_scales = torch.ones(_vm_means.shape[0], 3, device=device) * (0.7154 / 2)
            _vm_rot = torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(_vm_means.shape[0], 1)
            _verify_score_full = torch.cat([_verify_fp_t, torch.ones(_n_beacon, device=device)])  # (N+B,)
            if _verify_trace_path is not None:
                _verify_trace.append((pose_i, _n_phantom, n_points, _n_beacon))
            if pose_i % 10 == 0:
                print(f"[VERIFY={_verify_mode}] step {pose_i}: n_imagined={n_points} "
                      f"n_phantom={_n_phantom} n_beacon={_n_beacon}")

        # --- exp5 Oracle: per-step phantom mask -> discounted opacities for the candidate gain render ---
        render_opacities = gaussian_opacities
        if _oracle_on and n_points > 0:
            from scipy.spatial import cKDTree as _oc_ckt
            _ow = filtered_X_world.detach().cpu().numpy()
            if _oracle_tree is None:
                _og = gt_scene.return_entire_pt_cloud(return_features=False).detach().cpu().numpy()
                _oracle_tree = _oc_ckt(_og)
                _oracle_eps = _probe_eps_frac * float(np.linalg.norm(_og.max(0) - _og.min(0)))
            _od, _ = _oracle_tree.query(_ow, k=1)
            _ophantom = torch.tensor((_od > _oracle_eps).astype("float32"), device=device).view(-1, 1)
            render_opacities = gaussian_opacities * (1.0 - _oracle_lam * _ophantom)

        # --- exp7 Step 1 (gain lever): fade confirmed-suspect phantoms out of the candidate gain
        # render (same consumption point as the exp5 oracle hook) => no more fake coverage gain.
        # --- exp7 Step 2 (collision lever): drop them from the imagined-point collision set
        # => fake walls open up, shorter paths. Empty set is safe (helper returns False).
        _sc_collision_pts = filtered_X_world
        if _sc_on and _sc_suspect is not None:
            if _sc_lam > 0:
                render_opacities = render_opacities * (1.0 - _sc_lam * _sc_suspect.float().view(-1, 1))
            if _sc_collide:
                _sc_collision_pts = filtered_X_world[~_sc_suspect]

        # --- stage_3 Phase 0: the same two levers, now CONTINUOUS, driven by the posterior ---
        # gain lever: opacity <- LCB of the posterior (NOT opacity * LCB: multiplying gives
        # p^2 at c->inf instead of p, breaking the exact base reduction; setting is also the
        # honest semantic — the splat's opacity IS the belief). Downstream SimpleGaussianModel
        # clamps opacity to [1e-6, 1-1e-6], so at c->inf, kappa=0 this is pointwise identical
        # to base even for the raw-GELU points with occ > 1 (both saturate at the same clamp).
        # This restores the expectation-over-occupancy gain that already exists in the training
        # code (macarons_utils.predict_coverage_gain_for_single_camera) and that the MAGICIAN
        # planner replaced with a hard >0.5 cut.
        # collision lever: an imagined point is an obstacle only if its lower credible bound
        # clears tau, i.e. risk-aware collision instead of a boolean suspect mask.
        if _beta_on and _beta_mu is not None:
            if _beta_gain:
                render_opacities = _beta_lcb.view(-1, 1)
            if _beta_collide:
                _beta_keep = _beta_lcb > _beta_taucol
                _sc_collision_pts = filtered_X_world[_beta_keep]

        # --- exp3 dump (env-gated, read-only): occ field + imagined gaussians + GT surface + camera ---
        if _dump_dir and pose_i in _dump_steps:
            import numpy as _np
            _gt_pts = gt_scene.return_entire_pt_cloud(return_features=False)
            _np.savez_compressed(
                os.path.join(_dump_dir, f"step_{pose_i:03d}.npz"),
                X_world=X_world.detach().cpu().numpy(),
                occ_probs=occ_probs.detach().cpu().numpy().reshape(-1),
                imagined=filtered_X_world.detach().cpu().numpy(),
                occ_imagined=occ_values.detach().cpu().numpy().reshape(-1),
                X_cam=camera.X_cam_history[-1].detach().cpu().numpy().reshape(3),
                X_cam_history=_np.stack([x.detach().cpu().numpy().reshape(3) for x in camera.X_cam_history]),
                gt_surface=_gt_pts.detach().cpu().numpy(),
            )
            print(f"[DUMP] step {pose_i}: imagined={filtered_X_world.shape[0]} gt_surf={_gt_pts.shape[0]} -> {_dump_dir}")

        if pose_i == 0:
            sample_X_cam = camera.X_cam_history[0].view(1, 3)
            sample_V_cam = camera.V_cam_history[0].view(1, 2)
            R_sample, T_sample = get_camera_RT(sample_X_cam, sample_V_cam)
            sample_camera = FoVPerspectiveCameras(R=R_sample, T=T_sample, zfar=camera.zfar, device=device)
            K_matrix = sample_camera.get_projection_transform().get_matrix().transpose(-1, -2)

        # 1. initialize all novelty_values to 0
        novelty_values = torch.zeros(n_points, device=device)

        # 2. revisit all previous cameras
        history_length = len(camera.X_cam_history)

        for cam_idx in range(history_length):
            current_X_cam = camera.X_cam_history[cam_idx]
            current_V_cam = camera.V_cam_history[cam_idx]
            X_cam = current_X_cam.view(1, 3)
            V_cam = current_V_cam.view(1, 2)
            R_cam, T_cam = get_camera_RT(X_cam, V_cam)
            current_fov_camera = FoVPerspectiveCameras(R=R_cam, T=T_cam, zfar=camera.zfar, device=device)
            current_fov_camera.K = K_matrix  

            gs_cameras = convert_camera_from_pytorch3d_to_gs(
                current_fov_camera,
                height=camera.image_height,
                width=camera.image_width,
                device=device
            )
            gs_camera = gs_cameras[0]

            with torch.no_grad():
                rendered_depth, _ = render_gaussian_depth(
                    gaussian_means=gaussian_means,
                    gaussian_opacities=gaussian_opacities,
                    gaussian_scales=gaussian_scales,
                    gaussian_rotations=gaussian_rotations,
                    gaussian_colors=gaussian_colors,
                    gs_camera=gs_camera,
                    device=device,
                    bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                    kernel_size=0.01
                )
                current_depth_map = rendered_depth[0]

                current_visible_mask = camera.check_point_visibility_from_depth(
                    filtered_X_world, current_fov_camera, current_depth_map, depth_tolerance=1.0
                )
                # update the novelty along the visited cameras
                novelty_values[current_visible_mask] = 1.0

        print(f"historical: {novelty_values.sum().item()}/{n_points}")

        # stage_3 D-A: flush the dump now that novelty is known (novelty 1 = already seen).
        if _beta_dump_payload is not None:
            import numpy as _np
            _beta_dump_payload["novelty"] = novelty_values.detach().cpu().numpy().astype(_np.int8)
            _np.savez_compressed(os.path.join(_beta_dump_dir, f"beta_{pose_i:03d}.npz"),
                                 **_beta_dump_payload)
            print(f"[BETA-DUMP] step {pose_i}: {n_points} imagined "
                  f"({int((novelty_values == 0).sum())} novel) -> {_beta_dump_dir}")
            _beta_dump_payload = None

        # 3. Beam Search 
        remaining_steps = params.n_poses_in_trajectory + 1 - history_length
        print(f"Beam search remain: {remaining_steps} steps")

        # initialize beam search
        initial_pose_idx = camera.cam_idx
        beams = [{
            'trajectory': [],
            'novelty_values': novelty_values.clone(),
            'score': novelty_values.sum().item(),
            'total_coverage_gain': 0.0,
            'total_value': 0.0,  # exp6: coverage_gain + lambda*verify_gain (sort key when verify on)
            'current_pose_idx': initial_pose_idx
        }]

        # settings for beam search
        beam_width = params.beam_width
        for bs_i in range(params.beam_steps):
            print(f"Beam search step {bs_i + 1}/{params.beam_steps}")

            all_candidates = []

            # extend to every beams
            for beam in beams:
                neighbor_indices = camera.get_neighboring_poses(pose_idx=beam['current_pose_idx'])
                valid_neighbors = camera.get_valid_neighbors(neighbor_indices=neighbor_indices, mesh=mesh)

                rendering_candidate = []
                idx_candidate = []

                current_pose, _ = camera.get_pose_from_idx(beam['current_pose_idx'])
                X_current, _, _ = camera.get_camera_parameters_from_pose(current_pose)
                current_loc = X_current[0].cpu().numpy()

                for row in valid_neighbors:
                    neighbor_pose, _ = camera.get_pose_from_idx(row)
                    X_neighbor, V_neighbor, fov_neighbor = camera.get_camera_parameters_from_pose(neighbor_pose)
                    target_loc = X_neighbor[0].cpu().numpy()

                    if bs_i == 0:
                        if line_segment_mesh_intersection(current_loc, target_loc, intersector):
                            continue
                    else:
                        # we use occupancy points to check for future collisions.
                        # exp7: _sc_collision_pts == filtered_X_world unless SC mode drops suspects.
                        if line_segment_intersects_point_cloud_region(_sc_collision_pts, X_current[0], X_neighbor[0]):
                            continue

                    rendering_candidate.append(fov_neighbor)
                    idx_candidate.append(row)

                if len(rendering_candidate) == 0:
                    continue

                # rendering for every pose
                for j, pose_idx in enumerate(idx_candidate):
                    fov_camera = rendering_candidate[j]
                    fov_camera.K = K_matrix

                    gs_cameras = convert_camera_from_pytorch3d_to_gs(
                        fov_camera,
                        height=camera.image_height,
                        width=camera.image_width,
                        device=device
                    )
                    gs_camera = gs_cameras[0]

                    # update colors
                    current_novelty= beam['novelty_values']
                    gaussian_colors = update_gaussian_colors_from_novelty(current_novelty)

                    with torch.no_grad():
                        rendered_depth, rendered_image = render_gaussian_depth(
                            gaussian_means=gaussian_means,
                            gaussian_opacities=render_opacities,  # exp5 Oracle: phantom discounted (baseline if unset)
                            gaussian_scales=gaussian_scales,
                            gaussian_rotations=gaussian_rotations,
                            gaussian_colors=gaussian_colors,
                            gs_camera=gs_camera,
                            device=device,
                            bg_color=torch.tensor([1.0, 1.0, 1.0], device=device),
                            kernel_size=0.01
                        )
                        depth_map = rendered_depth[0]

                        # compute visible mask
                        visible_mask = camera.check_point_visibility_from_depth(
                            filtered_X_world, fov_camera, depth_map, depth_tolerance=1.0
                        )

                        # white pixels: unseen points（novelty_values=0）
                        valid_depth_mask = depth_map > 0
                        rgb_image = rendered_image  # shape: [3, H, W]
                        grayscale = rgb_image.mean(dim=0)  # [H, W]

               
                        depth_threshold = scene_scale / 2.0  

                        if valid_depth_mask.any():
                            nb_observed_pts_per_pixel = (depth_map / depth_threshold) ** 2
                            depth_weight = nb_observed_pts_per_pixel.clamp_max(1.0)

                            # compute coverage gain by using novelty map and depth weights map
                            coverage_gain = (grayscale * depth_weight * valid_depth_mask.float()).sum().item()
                        else:
                            coverage_gain = 0.0

                        new_novelty = current_novelty.clone()
                        new_novelty[visible_mask] = 1.0

                        new_total_coverage_gain = beam['total_coverage_gain'] + coverage_gain

                        # --- exp6 verify-steering: separate render carries verify score in ch0 ---
                        # (coverage render above is left untouched => coverage_gain == baseline)
                        verify_gain = 0.0
                        if _verify_active:
                            _vcol = torch.zeros(_vm_means.shape[0], 3, device=device)
                            _vcol[:, 0] = _verify_score_full  # ch0 = per-point verify score
                            with torch.no_grad():
                                _vrd, _vri = render_gaussian_depth(
                                    gaussian_means=_vm_means, gaussian_opacities=_vm_opac,
                                    gaussian_scales=_vm_scales, gaussian_rotations=_vm_rot,
                                    gaussian_colors=_vcol, gs_camera=gs_camera, device=device,
                                    bg_color=torch.tensor([0., 0., 0.], device=device), kernel_size=0.01)
                            _vdm = _vrd[0]
                            _vmask = (_vdm > 0).float()
                            _vdw = ((_vdm / (scene_scale / 2.0)) ** 2).clamp_max(1.0)
                            verify_gain = float((_vri[0] * _vdw * _vmask).sum().item())
                        new_total_value = beam['total_value'] + coverage_gain + _verify_lam * verify_gain

                        all_candidates.append({
                            'trajectory': beam['trajectory'] + [pose_idx],
                            'novelty_values': new_novelty,
                            'coverage_gain': coverage_gain,  # single step
                            'total_coverage_gain': new_total_coverage_gain,
                            'total_value': new_total_value,
                            'verify_gain': verify_gain,
                            'current_pose_idx': pose_idx
                        })

            if len(all_candidates) == 0:
                print("No valid candidates found!")
                break

            # coverage gains based on rgb imgs (exp6: add verify-steering term when enabled)
            _sort_key = 'total_value' if _verify_on else 'total_coverage_gain'
            all_candidates.sort(key=lambda x: x[_sort_key], reverse=True)
            beams = all_candidates[:beam_width]
            # print(f"Step {bs_i + 1}: Best total_coverage_gain = {beams[0]['total_coverage_gain']:.2f}, Score = {beams[0]['score']}/{n_points}, Current step gain = {beams[0].get('coverage_gain', 0):.2f}")
            print(f"Top {min(beam_width, len(all_candidates))} beams selected from {len(all_candidates)} candidates")

        if len(beams) > 0 and len(beams[0]['trajectory']) > 0:
            best_beam = beams[0]
            best_trajectory = best_beam['trajectory']
        else:
            # Dead-end: every neighbor was collision-blocked at the first beam step (typically
            # phantom walls boxing the drone in), so the initial beam's empty trajectory survived
            # and best_trajectory[0] would crash (pre-existing baseline bug). Hover at the current
            # pose instead: observations keep accumulating and can carve the blockers open.
            print("No valid trajectory found! Hovering at current pose.")
            best_trajectory = [initial_pose_idx]

        # move one step
        next_idx = best_trajectory[0]
        print(f"move one step: pose_idx = {next_idx}")

        # --- exp4 gain-attribution probe (env-gated, read-only): decompose the chosen
        #     viewpoint's coverage-gain into phantom (far from GT surface) vs real ---
        if _probe_path and n_points > 0:
            from scipy.spatial import cKDTree as _cKDTree
            _imgn = filtered_X_world.detach().cpu().numpy()
            if _probe_tree is None:
                _gp = gt_scene.return_entire_pt_cloud(return_features=False).detach().cpu().numpy()
                _probe_tree = _cKDTree(_gp)
                _probe_eps = _probe_eps_frac * float(np.linalg.norm(_gp.max(0) - _gp.min(0)))
            _pd, _ = _probe_tree.query(_imgn, k=1)
            _is_phantom = torch.tensor((_pd > _probe_eps).astype("float32"), device=device)  # (N,)
            # build the chosen next pose camera and render the imagined gaussians from it
            _pose_c, _ = camera.get_pose_from_idx(next_idx)
            _Xn, _Vn, _fovn = camera.get_camera_parameters_from_pose(_pose_c)
            _fovn.K = K_matrix
            _gscam = convert_camera_from_pytorch3d_to_gs(_fovn, height=camera.image_height,
                                                         width=camera.image_width, device=device)[0]
            # ch0 = (1-novelty) = the gain signal the planner sums; ch1 = phantom part of it
            _col = torch.zeros(n_points, 3, device=device)
            _col[:, 0] = (1.0 - novelty_values)
            _col[:, 1] = (1.0 - novelty_values) * _is_phantom
            with torch.no_grad():
                _rd, _ri = render_gaussian_depth(gaussian_means, gaussian_opacities, gaussian_scales,
                                                 gaussian_rotations, _col, _gscam, device,
                                                 bg_color=torch.tensor([0., 0., 0.], device=device), kernel_size=0.01)
            _dm = _rd[0]
            _vmask = (_dm > 0).float()
            _dw = ((_dm / (scene_scale / 2.0)) ** 2).clamp_max(1.0)
            _total_map = (_ri[0] * _dw * _vmask)
            _phan_map = (_ri[1] * _dw * _vmask)
            _total = float(_total_map.sum().item())
            _phan = float(_phan_map.sum().item())
            _probe_rows.append((pose_i, _total, _phan, int(n_points), int(_is_phantom.sum().item())))
            # save the per-pixel gain maps so the decomposition is viewable as images
            if _probe_img_dir and pose_i in _probe_img_steps:
                import numpy as _np2
                # The CHOSEN POSE goes in the file. GS rendering is non-deterministic, so a
                # re-run's trajectory does not match any stored LMDB; without the pose here
                # there is no way to render the matching ground-truth view afterwards.
                _np2.savez_compressed(
                    os.path.join(_probe_img_dir, f"probeimg_{pose_i:03d}.npz"),
                    total=_total_map.detach().cpu().numpy(),
                    phantom=_phan_map.detach().cpu().numpy(),
                    depth=_dm.detach().cpu().numpy(),
                    X_cam=_Xn.detach().cpu().numpy().reshape(3),
                    V_cam=_Vn.detach().cpu().numpy().reshape(2),
                    pose_idx=_np2.asarray(next_idx.detach().cpu().numpy()
                                          if torch.is_tensor(next_idx) else next_idx),
                    novelty=novelty_values.detach().cpu().numpy().astype(_np2.int8),
                    is_phantom=_is_phantom.detach().cpu().numpy().astype(_np2.int8))
                # the imagined cloud in 3D, coloured by (real|phantom) x (novel|already seen).
                # novel = contributes to the gain score; that is the population the figure is about.
                _nov = (novelty_values.detach().cpu().numpy() == 0)
                _phn = _is_phantom.detach().cpu().numpy() > 0.5
                _col_ply = _np2.zeros((len(_imgn), 3), _np2.uint8)
                _col_ply[~_phn & _nov] = (60, 220, 60)     # real, drives gain
                _col_ply[_phn & _nov] = (230, 45, 45)      # PHANTOM, drives gain  <-- the claim
                _col_ply[~_phn & ~_nov] = (35, 90, 35)     # real, already covered
                _col_ply[_phn & ~_nov] = (110, 40, 40)     # phantom, already covered
                write_ply_bin(os.path.join(_probe_img_dir, f"imagined_{pose_i:03d}.ply"),
                              _imgn, _col_ply)
                if pose_i == min(_probe_img_steps):        # scene context, dumped once
                    _gtp = gt_scene.return_entire_pt_cloud(return_features=False).detach().cpu().numpy()
                    write_ply_bin(os.path.join(_probe_img_dir, "gt_surface.ply"),
                                  _gtp, _np2.full((len(_gtp), 3), 170, _np2.uint8),
                                  max_points=300000)
                print(f"[GAIN-PROBE] step {pose_i}: dumped maps + pose + ply "
                      f"({int(_nov.sum())} novel, {int((_phn & _nov).sum())} of them phantom)")
            print(f"[GAIN-PROBE] step {pose_i}: total={_total:.1f} phantom={_phan:.1f} "
                  f"frac={(_phan / _total if _total > 0 else 0):.3f}")

        interpolation_step = 1
        for i in range(camera.n_interpolation_steps):
            camera.update_camera(next_idx, interpolation_step=interpolation_step)
            camera.capture_image(mesh)
            interpolation_step += 1

        pose_i += 1

    print("Coverage Evolution:", coverage_evolution)

    # --- exp4: write the gain-attribution probe csv ---
    if _probe_path and _probe_rows:
        os.makedirs(os.path.dirname(_probe_path), exist_ok=True)
        with open(_probe_path, "w") as _f:
            _f.write("step,total_gain,phantom_gain,n_imagined,n_phantom\n")
            for _r in _probe_rows:
                _f.write(f"{_r[0]},{_r[1]:.4f},{_r[2]:.4f},{_r[3]},{_r[4]}\n")
        print(f"[GAIN-PROBE] wrote {len(_probe_rows)} rows -> {_probe_path}")

    # --- exp6: write per-step verify trace (phantom count over steps) ---
    if _verify_trace_path and _verify_trace:
        os.makedirs(os.path.dirname(_verify_trace_path), exist_ok=True)
        with open(_verify_trace_path, "w") as _f:
            _f.write("step,n_phantom,n_imagined,n_beacon\n")
            for _r in _verify_trace:
                _f.write(f"{_r[0]},{_r[1]},{_r[2]},{_r[3]}\n")
        print(f"[VERIFY] wrote {len(_verify_trace)} rows -> {_verify_trace_path}")

    # --- exp7: write per-step self-correct trace (suspect count over steps) ---
    if _sc_trace_path and _sc_trace:
        os.makedirs(os.path.dirname(_sc_trace_path), exist_ok=True)
        with open(_sc_trace_path, "w") as _f:
            _f.write("step,n_imagined,n_mature,n_suspect\n")
            for _r in _sc_trace:
                _f.write(f"{_r[0]},{_r[1]},{_r[2]},{_r[3]}\n")
        print(f"[SCCOV] wrote {len(_sc_trace)} rows -> {_sc_trace_path}")

    # --- stage_3 Phase 0: write per-step Beta posterior trace ---
    if _beta_trace_path and _beta_trace:
        _d = os.path.dirname(_beta_trace_path)
        if _d:
            os.makedirs(_d, exist_ok=True)
        with open(_beta_trace_path, "w") as _f:
            _f.write("step,n_imagined,mean_n_inside_fov,mean_n_surface,mean_n_free,"
                     "mean_n_occluded,mean_p_net,mean_mu,mean_sigma,n_below_tau\n")
            for _r in _beta_trace:
                _f.write(",".join(str(_v) for _v in _r) + "\n")
        print(f"[BETA] wrote {len(_beta_trace)} rows -> {_beta_trace_path}")

    return coverage_evolution, camera.X_cam_history, camera.V_cam_history, full_pc, full_pc_colors, full_pc_idx
        
def run_magician_test(params_name,
             model_name,
             results_json_name,
             numGPU,
             test_scenes,
             test_resolution=0.05,
             use_perfect_depth_map=False,
             compute_collision=False,
             load_json=False,
             dataset_path=None,
             test_params=None):

    params_path = os.path.join(configs_dir, params_name)
    weights_path = os.path.join(weights_dir, model_name)
    results_json_path = os.path.join(results_dir, results_json_name)

    params = load_params(params_path)
    params.test_scenes = test_scenes
    # exp5 (env-gated, no effect when unset): override the scene list so one config can target any
    # scene (paired with MAGICIAN_ONLY_START / MAGICIAN_LMDB_DIR_NAME for isolated per-scene runs).
    _scenes_env = os.environ.get("MAGICIAN_TEST_SCENES")
    if _scenes_env:
        params.test_scenes = [s.strip() for s in _scenes_env.split(",") if s.strip()]
        print(f"[MAGICIAN_TEST_SCENES] overriding test scenes -> {params.test_scenes}")
    params.jitter_probability = 0.
    params.symmetry_probability = 0.
    params.anomaly_detection = False
    params.memory_dir_name = "test_memory_" + str(numGPU)

    params.jz = False
    params.numGPU = numGPU
    params.WORLD_SIZE = 1
    params.batch_size = 1
    params.total_batch_size = 1

    if dataset_path is None:
        params.data_path = data_path
    else:
        params.data_path = dataset_path

    # Setup device
    device = setup_device(params, None)

    # Setup model and dataloader
    dataloader, macarons, memory = setup_test(params, weights_path, device)

    params.beam_width = test_params.beam_width
    params.beam_steps = test_params.beam_steps

    lmdb_dir = os.path.join(results_dir, os.environ.get("MAGICIAN_LMDB_DIR_NAME") or test_params.lmdb_dir_name)
    os.makedirs(lmdb_dir, exist_ok=True)
    print(f"\nLMDB database directory: {lmdb_dir}")

    for i in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[i]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]
        occupied_pose_datas = [scene_dict['occupied_pose']]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)
            occupied_pose_data = occupied_pose_datas[i_scene]
            print("\nScene name:", scene_name)
            print("-------------------------------------")

            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)
            # segmented_mesh_path = os.path.join(scene_path, 'segmented.obj')

            mirrored_scene = False
            mirrored_axis = None

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=mirrored_scene, mirrored_axis=mirrored_axis)
           
            mesh_for_check = trimesh.load(mesh_path)

            if isinstance(mesh_for_check, trimesh.Scene):
                mesh_for_check = mesh_for_check.dump(concatenate=True)
            mesh_for_check.vertices *= params.scene_scale_factor

            intersector = mesh_for_check.ray

            print("Mesh Vertices shape:", mesh.verts_list()[0].shape)
            print("Min Vert:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax Vert:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Use memory info to set frames and poses path
            scene_memory_path = os.path.join(scene_path, params.memory_dir_name)

            torch.cuda.empty_cache()

            for start_cam_idx_i in range(len(settings.camera.start_positions)):
                # exp3: optionally restrict to specific start indices (env-gated; default runs all)
                _only = os.environ.get("MAGICIAN_ONLY_START")
                if _only and str(start_cam_idx_i) not in [s.strip() for s in _only.split(",")]:
                    continue
                start_cam_idx = settings.camera.start_positions[start_cam_idx_i]
                print("\n" + "="*60)
                print(f"Start cam index {start_cam_idx_i} for {scene_name}: {start_cam_idx}")
                print("="*60)

                # Each start_cam_idx_i gets its own trajectory number
                trajectory_nb = start_cam_idx_i
                training_frames_path = memory.get_trajectory_frames_path(scene_memory_path, trajectory_nb)
                print(f"Using trajectory folder: {training_frames_path}")

                # Setup the Scene and Camera objects
                gt_scene, covered_scene, surface_scene, proxy_scene = None, None, None, None
                gc.collect()
                torch.cuda.empty_cache()
                gt_scene, covered_scene, surface_scene, proxy_scene = setup_test_scene(params,
                                                                                       mesh,
                                                                                       settings,
                                                                                       mirrored_scene,
                                                                                       device,
                                                                                       mirrored_axis=mirrored_axis,
                                                                                       test_resolution=test_resolution)

                # clear_folder(training_frames_path)
                camera = setup_test_camera(params, mesh, intersector, start_cam_idx, settings, occupied_pose_data,
                                           device, training_frames_path,
                                           mirrored_scene=mirrored_scene, mirrored_axis=mirrored_axis)
                print(camera.X_cam_history[0], camera.V_cam_history[0])

                coverage_evolution, X_cam_history, V_cam_history, full_pc, full_pc_colors, full_pc_idx = compute_magician_trajectory(params, macarons,
                                                                                      camera,
                                                                                      gt_scene, surface_scene,
                                                                                      proxy_scene, covered_scene,
                                                                                      mesh,
                                                                                      intersector,
                                                                                      device,
                                                                                      settings,
                                                                                      test_resolution=test_resolution,
                                                                                      use_perfect_depth_map=use_perfect_depth_map,
                                                                                      compute_collision=compute_collision)
                

                # Open LMDB, save data, then close
                print(f"\n=== Saving trajectory data to LMDB ===")
                lmdb_env = lmdb.open(lmdb_dir, map_size=30 * 1024 * 1024 * 1024)

                # Save trajectory data to LMDB
                lmdb_key = f"{scene_name}/{start_cam_idx_i}"
                trajectory_data = {
                    'coverage': coverage_evolution,
                    'X_cam_history': X_cam_history.cpu().numpy(),
                    'V_cam_history': V_cam_history.cpu().numpy(),
                    'points': full_pc.cpu().numpy(),
                    'points_color': full_pc_colors.cpu().numpy()
                }
                save_to_lmdb(lmdb_env, lmdb_key, trajectory_data)

                # Close LMDB
                lmdb_env.close()
                print(f"Closed LMDB database for {scene_name}/{start_cam_idx_i}\n")

                # Cleanup: Keep only imgs folder, delete frames/depths/occupancy folders
                # cleanup_trajectory_folders(training_frames_path, keep_folders=['imgs'])
                # print(f"Finished processing trajectory {start_cam_idx_i}\n")

    print("All trajectories computed.")
