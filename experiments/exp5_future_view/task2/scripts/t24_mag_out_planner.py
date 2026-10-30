"""EXP5 Task 2 -- outdoor (Macarons++) MAGICIAN planner child, groups MAG-G (beam 1 x 1) and MAG-B (beam 10 x 10)
(magician env; planner side). Official MAGICIAN e697009 through the adapter core that is bitwise equal to the official
loop (task0/magician_comparison/adapters/magician_core.py, test E1), with the scene's OFFICIAL settings.json (scene and
camera blocks unchanged):
  layer 0  : the environment's legal list of the service record (= official get_neighboring_poses -> get_valid_neighbors
             -> GT segment test, computed by the shared M-protocol simulator)
  layer >0 : official get_neighboring_poses of the beam pose (grid_env.neighbours on the settings grid, equal to the
             official Camera, test E4), official prefer-unvisited validity, official occupancy-point segment test (core)
  seeds    : sha256("exp5-T2|MAG|<tag>|<t>"), shared by MAG-G and MAG-B
Protocol: the JSON lines of m_t6_sim.PlannerClient ({t, svc} -> {t, pick, fallback, decision_s, scores};
{cmd: end} -> {bye}); --resume-t T re-observes the records 0..T-1 first.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MCA = os.path.join(HERE, "..", "..", "task0", "magician_comparison", "adapters")
sys.path.insert(0, MCA)
import magician_planner as MPL  # noqa: E402
import magician_planner_grid as MPG  # noqa: E402
import grid_env as GE  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

CORE, MP = MPL.CORE, MPL.MP
DATA = "/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++"


def grid_of(settings):
    c = settings["camera"]
    return {"x_min": 10.0 * np.asarray(c["x_min"], np.float64), "x_max": 10.0 * np.asarray(c["x_max"], np.float64),
            "n": (int(c["pose_l"]), int(c["pose_w"]), int(c["pose_h"])), "n_theta": int(c["pose_n_theta"]), "n_azim": int(c["pose_n_azim"])}


class Planner(MPG.Planner):
    def __init__(self, group, scene, tag, dev):
        self.g, self.tag, self.dev = group, tag, dev
        sd = json.load(open(os.path.join(DATA, scene, "settings.json")))
        grid = grid_of(sd)
        p = self.params = MP.load_params(os.path.join(MP.configs_dir, "macarons_default_training_config.json"))
        for k, v in dict(jitter_probability=0., symmetry_probability=0., anomaly_detection=False, jz=False, numGPU=0,
                         WORLD_SIZE=1, batch_size=1, total_batch_size=1).items():
            setattr(p, k, v)
        p.beam_width, p.beam_steps = MPL.BEAM[group]
        self.settings = MP.Settings(sd, dev, p.scene_scale_factor)
        self.nb = MPG.GridNeighbours(grid, dev)
        s = MPL.seed_of(f"exp5-T2|MAG|{tag}")
        torch.manual_seed(s)
        np.random.seed(s % 2 ** 32)
        macarons = CORE.load_model(p, dev)
        surf, prox = CORE.make_planner_scenes(p, self.settings, dev)
        renderer = MP.get_rgb_renderer(image_height=p.image_height, image_width=p.image_width, ambient_light_intensity=p.ambient_light_intensity,
                                       cameras=MP.FoVPerspectiveCameras(device=dev), device=dev, max_faces_per_bin=200000)
        sc = self.settings.camera
        self.cam = MP.Camera(x_min=sc.x_min, x_max=sc.x_max, pose_l=1, pose_w=1, pose_h=1, pose_n_elev=sc.pose_n_elev,
                             pose_n_azim=sc.pose_n_azim, n_interpolation_steps=p.n_interpolation_steps, zfar=p.zfar, renderer=renderer,
                             device=dev, contrast_factor=sc.contrast_factor, gathering_factor=p.gathering_factor, save_dir_path=None)
        self.core = CORE.MagicianCore(p, self.settings, macarons, self.cam, surf, prox, dev, *MPL.BEAM[group])
        self.info = {"settings": "official settings.json", "grid_n": list(grid["n"]), "beam": MPL.BEAM[group],
                     "scene_scale": self.core.scene_scale, "seed_key": f"exp5-T2|MAG|{tag}"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(MPL.BEAM), required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--svc-dir", required=True)
    ap.add_argument("--resume-t", type=int, default=0)
    a = ap.parse_args()
    proto = os.fdopen(os.dup(1), "w")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    dev = torch.device("cuda:0")
    torch.cuda.set_device(dev)
    tag = f"{a.scene}_s{a.start}"
    seed0 = MPL.seed_of

    def seed_t2(key):                                 # Task 2 seed keys (the grid planner formats "exp5-T0.12|MAG|...")
        return seed0(key.replace("exp5-T0.12|MAG|", "exp5-T2|MAG|"))
    MPL.seed_of = seed_t2
    pl = Planner(a.method, a.scene, tag, dev)
    svc = os.path.abspath(a.svc_dir)
    for t in range(a.resume_t):
        pl.step(t, MPL.MC.load_service_record(os.path.join(svc, f"svc_{a.scene}_s{a.start}_t{t}.npz")))
    g = MPL.Guard().install()
    proto.write(json.dumps({"ready": True, "info": pl.info, "resume_t": a.resume_t}) + "\n")
    proto.flush()
    tot = 0.0
    for line in sys.stdin:
        q = json.loads(line)
        if q.get("cmd") == "end":
            break
        t = int(q["t"])
        ans = pl.step(t, MPL.MC.load_service_record(q["svc"]))
        tot += ans["timing"]["decision_s"]
        proto.write(json.dumps({"t": t, "pick": ans["pick"], "fallback": ans["fallback"], "decision_s": ans["timing"]["decision_s"],
                                "scores": ans["scores"], "diag": ans["diag"], "guard_denied": list(g.denied)}, default=float) + "\n")
        proto.flush()
    g.uninstall()
    proto.write(json.dumps({"bye": True, "decision_s_total": tot, "guard_denied": list(g.denied),
                            "peak_mem_gb": torch.cuda.max_memory_allocated(dev) / 2 ** 30}) + "\n")
    proto.flush()


if __name__ == "__main__":
    main()
