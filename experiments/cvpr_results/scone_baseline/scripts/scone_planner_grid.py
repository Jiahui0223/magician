"""EXP9-SCONE -- SCONE planner child for the MP3D-150 indoor closed loop (magician env; planner side only).

Same child protocol, settings, camera bookkeeping, observation update, seeds scheme and LeakGuard as the indoor MAGICIAN baseline
(experiments/exp5_future_view/task0/magician_comparison/adapters/magician_planner_grid.py, MAG-B of EXP7), imported unchanged;
only the decision differs:
  MAG-B : SconeOcc occupancy -> imagined Gaussians -> novelty x distance gain -> beam search 10 x 10 (trained_macarons.pth)
  SCONE : SconeOcc occupancy field -> SconeVis coverage gain of every candidate -> greedy argmax (official ShapeNet-pretrained
          SCONE weights, scone_common.load_scone_model), the official MACARONS-repo calls of testers/scene.py compute_trajectory.
Candidates = the simulator legal list of the service record (official grid neighbours + GT single-segment rule, simulator side)
after the official get_valid_neighbors rule "poses not executed before first (all if none)" (= MAG-B layer 0 and RAND-NV).
Observation per step = the whitelisted service record (zbuf, mask, R, T, 10 % sample points, legal list, executed history);
GT mesh / coverage / evaluator files are blocked by the guard (installed after the weights are loaded, before the first record).
Seeds: torch / numpy seeded with sha256("cvpr-SCONE|<tag>") before the scene setup (proxy points) and with
sha256("cvpr-SCONE|<tag>|<t>") at the start of every step.
JSON lines: {"t", "svc"} -> {"pick", "scores", "fallback", "timing", "diag", "guard_denied"} (t12 / t23 protocol).
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
ADP = os.path.join(REPO, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters")
sys.path.insert(0, ADP)
sys.path.insert(0, HERE)
import magician_planner as MPL  # noqa: E402  (Guard, seed_of, MC.load_service_record)
import magician_planner_grid as MPG  # noqa: E402  (GridNeighbours)
import grid_env as GE  # noqa: E402
import station_env as SE  # noqa: E402
import scone_common as SC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CORE, MP = MPL.CORE, MPL.MP


class Planner:
    def __init__(self, card_key, tag, dev, diag_dir=None, diag_steps=()):
        self.tag, self.dev, self.diag_dir, self.diag_steps = tag, dev, diag_dir, set(diag_steps)
        card = json.load(open(os.path.join(MPL.W, "preprocessing", "scenes", f"{card_key}.json")))
        grid = GE.grid_def(card)
        sd = {"scene": SE.settings_dict(card)["scene"], "camera": GE.settings_camera(grid)}
        p = self.params = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
        for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0,
                         WORLD_SIZE=1, batch_size=1, total_batch_size=1).items():
            setattr(p, k, v)
        self.settings = MP.Settings(sd, dev, p.scene_scale_factor)
        self.nb = MPG.GridNeighbours(grid, dev)
        s = MPL.seed_of(f"cvpr-SCONE|{tag}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        self.macarons, self.model_info = SC.load_scone_model(MP.load_pretrained_macarons, MP.weights_dir, dev)
        surf, prox = CORE.make_planner_scenes(p, self.settings, dev)
        renderer = MP.get_rgb_renderer(image_height=p.image_height, image_width=p.image_width,
                                       ambient_light_intensity=p.ambient_light_intensity,
                                       cameras=MP.FoVPerspectiveCameras(device=dev), device=dev, max_faces_per_bin=200000)
        sc = self.settings.camera            # a 1 x 1 x 1 camera grid only holds the camera state (as MAG-B)
        self.cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=1, pose_w=1, pose_h=1, pose_n_elev=sc.pose_n_elev,
                             pose_n_azim=sc.pose_n_azim, n_interpolation_steps=p.n_interpolation_steps, zfar=p.zfar,
                             renderer=renderer, device=dev, contrast_factor=sc.contrast_factor,
                             gathering_factor=p.gathering_factor, save_dir_path=None)
        # MagicianCore is used ONLY for begin_step / observe (the official surface / proxy update); its plan() is never called
        self.core = CORE.MagicianCore(p, self.settings, self.macarons, self.cam, surf, prox, dev, 1, 1)
        self.info = {"method": "SCONE", "settings": {"scene": sd["scene"], "camera": sd["camera"]}, "grid_n": list(grid["n"]),
                     "grid_step_m": list(grid["step_m"]), "n_proxy_points": int(p.n_proxy_points),
                     "distance_factor_th": p.distance_factor_th, "min_occ_for_proxy_points": p.min_occ_for_proxy_points,
                     "seq_len": p.seq_len, "sensor_range": p.sensor_range, "model": self.model_info,
                     "planner": os.path.abspath(__file__), "planner_sha256": SC.sha256_file(os.path.abspath(__file__)),
                     "common_sha256": SC.sha256_file(os.path.join(HERE, "scone_common.py")),
                     "magician_root": CORE.MAGICIAN_ROOT}

    def step(self, t, rec):
        t0 = time.time()
        s = MPL.seed_of(f"cvpr-SCONE|{self.tag}|{t}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        key = tuple(int(v) for v in rec["pose_idx"])
        X, V = self.nb.XV(key)
        R, T = MP.get_camera_RT(X, V)
        assert torch.allclose(R.cpu().double(), torch.as_tensor(rec["R"]).view(1, 3, 3), atol=1e-5), "pose / record mismatch"
        cam = self.cam                                    # official Camera.update_camera bookkeeping for the executed pose
        cam.X_cam, cam.V_cam = X, V
        cam.X_cam_history = torch.vstack((cam.X_cam_history, X))
        cam.V_cam_history = torch.vstack((cam.V_cam_history, V))
        cam.fov_camera = MP.FoVPerspectiveCameras(R=R, T=T, zfar=cam.zfar, device=self.dev)
        self.nb.visited.add(key)
        self.nb.set_legal(rec["legal_idx"])
        self.core.begin_step(t)                           # fov_camera_0 = fov_camera; surface recompute every 20 loops
        zb = torch.as_tensor(np.asarray(rec["zbuf"], np.float32), device=self.dev).view(1, cam.image_height, cam.image_width, 1)
        depth = torch.clamp(zb, min=0.5, max=750.0)                    # official apply_perfect_depth_simple
        mask = torch.as_tensor(np.asarray(rec["mask"]), device=self.dev).view(1, cam.image_height, cam.image_width, 1).bool()
        self.core.observe(depth, mask, torch.as_tensor(np.asarray(rec["sample_points"], np.float32), device=self.dev))
        t_obs = time.time()
        legal = self.nb.legal
        cand_keys = self.nb._valid(list(legal))           # official get_valid_neighbors: unexecuted poses first
        cands = []
        for k_ in cand_keys:
            Xn, Vn = self.nb.XV(k_)
            Rn, Tn = MP.get_camera_RT(Xn, Vn)
            cands.append((Xn, MP.FoVPerspectiveCameras(R=Rn, T=Tn, zfar=cam.zfar, device=self.dev)))
        j, gains, X_world, occ_probs = SC.scone_greedy(self.params, self.macarons.scone, cam, self.core.surface_scene,
                                                       self.core.proxy_scene, cands, self.dev,
                                                       MP.compute_scene_occupancy_probability_field,
                                                       MP.predict_coverage_gain_for_single_camera)
        pick = legal.index(cand_keys[j]) if j is not None else None
        g_of = dict(zip(cand_keys, gains))
        scores = [g_of.get(k_) for k_ in legal]
        diag = {"occ": SC.occ_stats(X_world, occ_probs, self.core.proxy_scene), "n_candidates": len(cand_keys),
                "n_legal": len(legal), "n_legal_unvisited": sum(1 for k_ in legal if k_ not in self.nb.visited),
                "n_surface_pc": int(len(self.core.full_pc))}
        if self.diag_dir and t in self.diag_steps:
            np.savez_compressed(os.path.join(self.diag_dir, f"{self.tag}_occ_t{t}.npz"), X=X_world.float().cpu().numpy(),
                                n_pred=int(X_world.shape[0] - (self.core.proxy_scene.out_of_field > 0.).sum().item()),
                                occ=occ_probs.float().cpu().numpy().reshape(-1), cam=X.float().cpu().numpy(),
                                legal_idx=np.asarray(legal), gains=np.asarray([np.nan if g is None else g for g in scores]),
                                pick=-1 if pick is None else pick)
        return {"t": t, "pick": pick, "scores": scores, "fallback": False, "refresh": None, "diag": diag,
                "timing": {"decision_s": time.time() - t0, "observe_s": t_obs - t0, "scone_s": time.time() - t_obs},
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--diag-dir", default=None)
    ap.add_argument("--diag-steps", type=int, nargs="*", default=[])
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")              # protocol on a private fd; library prints go to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    if a.diag_dir:
        os.makedirs(a.diag_dir, exist_ok=True)
    pl = Planner(a.card, a.tag, dev, a.diag_dir, a.diag_steps)
    g = MPL.Guard().install()
    proto.write(json.dumps({"ready": True, "info": pl.info}) + "\n")
    proto.flush()
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        ans = pl.step(int(q["t"]), MPL.MC.load_service_record(q["svc"]))
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
