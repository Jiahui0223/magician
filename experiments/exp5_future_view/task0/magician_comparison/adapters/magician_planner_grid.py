"""EXP5 Task 0 / T0.12 -- MAGICIAN planner child on the released-code-style camera grid (magician env; planner side).
Same protocol, model, core, seeds and LeakGuard as adapters/magician_planner.py (T0.10); only the pose / candidate
space differs (protocol v1.3 §20):
  layer 0  : the environment-legal list of the service record (official neighbours + official GT segment rule)
  layer >0 : official get_neighboring_poses of the beam pose (grid_env.neighbours, keep_self=True, checked equal to the
             official Camera in test E5), then the official occupancy-point segment test inside the core
  validity : official get_valid_neighbors = poses not executed by the real trajectory first (all if none)
  settings : scene block as T0.10 (1.40 m horizontal cells, 1 vertical); camera block = grid_env.settings_camera
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import magician_planner as MPL  # noqa: E402  (T0.10 planner: Guard, seeds, fallback, beam table)
import grid_env as GE  # noqa: E402
import station_env as SE  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CORE, MP = MPL.CORE, MPL.MP


class GridNeighbours:
    def __init__(self, g, device):
        self.g, self.dev, self.visited, self.legal = g, device, set(), []

    def XV(self, key):
        X = torch.tensor(GE.position(self.g, key).reshape(1, 3), dtype=torch.float32, device=self.dev)
        e, a = GE.angles(self.g, key)
        return X, torch.tensor([[e, a]], dtype=torch.float32, device=self.dev)

    def set_legal(self, legal_idx):
        self.legal = [tuple(int(v) for v in r) for r in legal_idx]

    def _valid(self, keys):
        new = [k for k in keys if k not in self.visited]
        return new if new else keys

    def __call__(self, key, layer):
        keys = list(self.legal) if layer == 0 else GE.neighbours(self.g, key, keep_self=True)
        return [(k, *self.XV(k)) for k in self._valid(keys)]


class Planner(MPL.Planner):
    def __init__(self, group, card_key, tag, dev):
        self.g, self.tag, self.dev = group, tag, dev
        card = json.load(open(os.path.join(MPL.W, "preprocessing", "scenes", f"{card_key}.json")))
        grid = GE.grid_def(card)
        sd = {"scene": SE.settings_dict(card)["scene"], "camera": GE.settings_camera(grid)}
        p = self.params = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
        for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0,
                         WORLD_SIZE=1, batch_size=1, total_batch_size=1).items():
            setattr(p, k, v)
        p.beam_width, p.beam_steps = MPL.BEAM[group]
        self.settings = MP.Settings(sd, dev, p.scene_scale_factor)
        self.nb = GridNeighbours(grid, dev)
        s = MPL.seed_of(f"exp5-T0.12|MAG|{tag}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        macarons = CORE.load_model(p, dev)
        surf, prox = CORE.make_planner_scenes(p, self.settings, dev)
        renderer = MP.get_rgb_renderer(image_height=p.image_height, image_width=p.image_width, ambient_light_intensity=p.ambient_light_intensity,
                                       cameras=MP.FoVPerspectiveCameras(device=dev), device=dev, max_faces_per_bin=200000)
        sc = self.settings.camera            # a 1 x 1 x 1 camera grid only holds the camera state (no 10^5-pose table)
        self.cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=1, pose_w=1, pose_h=1, pose_n_elev=sc.pose_n_elev,
                             pose_n_azim=sc.pose_n_azim, n_interpolation_steps=p.n_interpolation_steps, zfar=p.zfar, renderer=renderer,
                             device=dev, contrast_factor=sc.contrast_factor, gathering_factor=p.gathering_factor, save_dir_path=None)
        self.core = CORE.MagicianCore(p, self.settings, macarons, self.cam, surf, prox, dev, *MPL.BEAM[group])
        self.info = {"settings": {"scene": sd["scene"], "camera": sd["camera"]}, "grid_n": list(grid["n"]), "grid_step_m": list(grid["step_m"]),
                     "beam": MPL.BEAM[group], "n_proxy_points": int(p.n_proxy_points), "scene_scale": self.core.scene_scale}

    def step(self, t, rec):
        t0 = time.time()
        s = MPL.seed_of(f"exp5-T0.12|MAG|{self.tag}|{t}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        key = tuple(int(v) for v in rec["pose_idx"])
        X, V = self.nb.XV(key)
        R, T = MP.get_camera_RT(X, V)
        assert torch.allclose(R.cpu().double(), torch.as_tensor(rec["R"]).view(1, 3, 3), atol=1e-5), "pose / record mismatch"
        cam = self.cam
        cam.X_cam, cam.V_cam = X, V
        cam.X_cam_history = torch.vstack((cam.X_cam_history, X))
        cam.V_cam_history = torch.vstack((cam.V_cam_history, V))
        cam.fov_camera = MP.FoVPerspectiveCameras(R=R, T=T, zfar=cam.zfar, device=self.dev)
        self.nb.visited.add(key)
        self.nb.set_legal(rec["legal_idx"])
        self.core.begin_step(t)
        zb = torch.as_tensor(np.asarray(rec["zbuf"], np.float32), device=self.dev).view(1, cam.image_height, cam.image_width, 1)
        depth = torch.clamp(zb, min=0.5, max=750.0)
        mask = torch.as_tensor(np.asarray(rec["mask"]), device=self.dev).view(1, cam.image_height, cam.image_width, 1).bool()
        self.core.observe(depth, mask, torch.as_tensor(np.asarray(rec["sample_points"], np.float32), device=self.dev))
        t_obs = time.time()
        nxt, diag = self.core.plan(key, self.nb)
        legal = self.nb.legal
        scores = [diag["layer0_gains"].get(k) for k in legal]
        if nxt is None:
            pick, fb = (MPL.fallback(self.tag, t, len(legal)), True) if legal else (None, False)
        else:
            pick, fb = legal.index(nxt), False
        diag = {k: v for k, v in diag.items() if k != "layer0_gains"}
        return {"t": t, "pick": pick, "scores": scores, "fallback": fb, "refresh": None, "diag": diag,
                "timing": {"decision_s": time.time() - t0, "observe_s": t_obs - t0, "occ_s": diag["occ_s"],
                           "novelty_s": diag["novelty_s"], "beam_s": diag["beam_s"]},
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(MPL.BEAM), required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    pl = Planner(a.method, a.card, a.tag, dev)
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
