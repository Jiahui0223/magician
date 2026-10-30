"""EXP5 Task 0 / T0.10 -- MAGICIAN planner child for the Task 0 indoor closed loop (magician env; planner side).
JSON lines on stdin / stdout, same protocol as t06_planner.py:
  {"t": t, "svc": path} -> {"pick": i, "scores": [...], "fallback": bool, "timing": {...}, "diag": {...}, "guard_denied": [...]}
Groups (magician_comparison protocol v1.2 §11):
  MAG-G : official MAGICIAN with beam width 1, beam steps 1 (single-step; EXP3b magician_greedy convention)
  MAG-B : official MAGICIAN with its released beam search, width 10, steps 10
The official model / representation / gain run unchanged through adapters/magician_core.py (bitwise equal to the
official loop in native mode, test E1); candidates come from adapters/station_env.py. Observations = the whitelisted
service record (zbuf, mask, R, T, 10 % sample points, legal list); no GT, coverage or evaluation file is readable
(LeakGuard installed after the model weights are loaded, before the first record).
Seeds (common random numbers for MAG-G and MAG-B): torch / numpy seeded with sha256("exp5-T0.10|MAG|<tag>") before
the scene setup (proxy points) and with sha256("exp5-T0.10|MAG|<tag>|<t>") at the start of every step. No pick -> seeded uniform fallback (never expected).
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import magician_core as CORE  # noqa: E402
import station_env as SE  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(REPO, "work_dirs", "exp5_future_view", "task0")
sys.path.insert(0, os.path.join(REPO, "experiments", "exp3b_macaronspp_transfer"))
import m_common as MC  # noqa: E402
import t5_diagnostic as TD  # noqa: E402

MP = CORE.MP
BEAM = {"MAG-G": (1, 1), "MAG-B": (10, 10)}


class Guard(TD.LeakGuard):
    @staticmethod
    def forbidden(p):
        if TD.LeakGuard.forbidden(p):
            return True
        a = os.path.abspath(p) if isinstance(p, str) else ""
        return any(x in a for x in ("/matterport3d/", "/mp3d_magician/", "/preprocessing/coverage/", "/preprocessing/mesh_cache/",
                                    "/labels/", "/eval3d/", "/metric_replay/")) or a.endswith(("traj.json", ".house", "episode.json"))


def seed_of(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:15], 16)


def fallback(tag, t, n):
    rng = np.random.default_rng(seed_of(f"exp5-T0.10|fallback|{tag}|{t}"))
    return int(rng.integers(n))


class Planner:
    def __init__(self, group, card_key, tag, dev):
        self.g, self.tag, self.dev = group, tag, dev
        card = json.load(open(os.path.join(W, "preprocessing", "scenes", f"{card_key}.json")))
        self.params = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
        for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0,
                         WORLD_SIZE=1, batch_size=1, total_batch_size=1).items():
            setattr(self.params, k, v)
        self.params.beam_width, self.params.beam_steps = BEAM[group]
        self.settings = MP.Settings(SE.settings_dict(card), dev, self.params.scene_scale_factor)
        self.nb = SE.StationNeighbours(card, dev)
        s = seed_of(f"exp5-T0.10|MAG|{tag}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        macarons = CORE.load_model(self.params, dev)
        surf, prox = CORE.make_planner_scenes(self.params, self.settings, dev)
        p = self.params
        fov0 = MP.FoVPerspectiveCameras(device=dev)
        renderer = MP.get_rgb_renderer(image_height=p.image_height, image_width=p.image_width,
                                       ambient_light_intensity=p.ambient_light_intensity, cameras=fov0, device=dev,
                                       max_faces_per_bin=200000)
        sc = self.settings.camera
        self.cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=sc.pose_l, pose_w=sc.pose_w, pose_h=sc.pose_h,
                             pose_n_elev=sc.pose_n_elev, pose_n_azim=sc.pose_n_azim, n_interpolation_steps=p.n_interpolation_steps,
                             zfar=p.zfar, renderer=renderer, device=dev, contrast_factor=sc.contrast_factor,
                             gathering_factor=p.gathering_factor, save_dir_path=None)
        self.core = CORE.MagicianCore(p, self.settings, macarons, self.cam, surf, prox, dev, *BEAM[group])
        self.info = {"settings": SE.settings_dict(card), "beam": BEAM[group], "n_proxy_points": int(p.n_proxy_points),
                     "scene_scale": self.core.scene_scale, "image": [p.image_height, p.image_width]}

    def step(self, t, rec):
        t0 = time.time()
        s = seed_of(f"exp5-T0.10|MAG|{self.tag}|{t}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        pi = [int(x) for x in rec["pose_idx"]]
        key = (pi[0], pi[3], pi[4])
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
        self.core.begin_step(t)
        zb = torch.as_tensor(np.asarray(rec["zbuf"], np.float32), device=self.dev).view(1, cam.image_height, cam.image_width, 1)
        depth = torch.clamp(zb, min=0.5, max=750.0)                    # official apply_perfect_depth_simple
        mask = torch.as_tensor(np.asarray(rec["mask"]), device=self.dev).view(1, cam.image_height, cam.image_width, 1).bool()
        part_pc = torch.as_tensor(np.asarray(rec["sample_points"], np.float32), device=self.dev)
        self.core.observe(depth, mask, part_pc)
        t_obs = time.time()
        nxt, diag = self.core.plan(key, self.nb)
        legal = self.nb.legal
        scores = [diag["layer0_gains"].get(k) for k in legal]
        if nxt is None:
            pick, fb = (fallback(self.tag, t, len(legal)), True) if legal else (None, False)
        else:
            pick, fb = legal.index(nxt), False
        diag = {k: v for k, v in diag.items() if k != "layer0_gains"}
        return {"t": t, "pick": pick, "scores": scores, "fallback": fb, "refresh": None, "diag": diag,
                "timing": {"decision_s": time.time() - t0, "observe_s": t_obs - t0, "occ_s": diag["occ_s"],
                           "novelty_s": diag["novelty_s"], "beam_s": diag["beam_s"]},
                "peak_mem_gb": torch.cuda.max_memory_allocated(self.dev) / 2 ** 30}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(BEAM), required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")              # protocol on a private fd; library prints go to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    pl = Planner(a.method, a.card, a.tag, dev)
    g = Guard().install()
    proto.write(json.dumps({"ready": True, "info": pl.info}) + "\n")
    proto.flush()
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("bye"):
            break
        rec = MC.load_service_record(q["svc"])
        ans = pl.step(int(q["t"]), rec)
        ans["guard_denied"] = list(g.denied)
        proto.write(json.dumps(ans) + "\n")
        proto.flush()
    g.uninstall()


if __name__ == "__main__":
    main()
