"""EXP7 / UAVScenes -- closed loop on a UAV-v0 tile (magician env; simulator / GT side). The frozen Macarons++ outdoor chain run
unchanged: task2 t24_out_sim.py (Task 1.4 v2_sim / EXP3b EpisodeSim: official settings grid, official neighbours ->
get_valid_neighbors -> GT segment test, hover at dead ends, seeded 10 % observation, official MAGICIAN coverage / visibility_ratio,
100 moves) with the data root MACARONS_PLUSPLUS_ROOT = the UAV tile root (must be set by the caller before Python starts).
  --method OURS   planner child uav_d_out_planner.py (V3-T0@40K + D1, D_CONFIG = configs/d_config_uav.yaml), group V2M
  --method RANDOM frozen EXP3b m_t7_sim.py --method random --seed 0 (uniform over the simulator's legal list), as CVPR Table 1
  --method MAG-B  planner child uav_mag_planner.py (task2 t24_mag_out_planner.py with the settings folder redirected), group MAG-B
Usage: MACARONS_PLUSPLUS_ROOT=<uav root> python uav_sim_out.py --method OURS|RANDOM|MAG-B --scene uav_<...> --start K --root DIR"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments")
sys.path.insert(0, HERE)
import uav_hooks as UH  # noqa: E402

assert os.path.abspath(os.environ.get("MACARONS_PLUSPLUS_ROOT", "")) == UH.UAV_ROOT, "set MACARONS_PLUSPLUS_ROOT to the UAV root"
CK = os.path.join(R, "work_dirs", "exp6_v3", "task0_vae", "checkpoints", "final.pt")


def main():
    i = sys.argv.index("--method")
    method = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    if method == "RANDOM":
        a = sys.argv[1:]
        sc, st, root = a[a.index("--scene") + 1], a[a.index("--start") + 1], a[a.index("--root") + 1]
        cmd = [sys.executable, os.path.join(X, "exp3b_macaronspp_transfer", "m_t7_sim.py"), "--method", "random", "--scene", sc,
               "--start", st, "--seed", "0", "--out", root]
        sys.exit(subprocess.call(cmd))
    assert method in ("OURS", "MAG-B")
    sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
    import t24_out_sim as T24
    LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
    if method == "MAG-B":
        class MagPlanner(T6S.PlannerClient):
            def __init__(self, a, resume_t, err_path):
                tg = LC.tag(a.exp, a.scene, a.start, a.rep)
                cmd = [LC.MAGICIAN_PYTHON, "-I", os.path.join(HERE, "uav_mag_planner.py"), "--method", "MAG-B", "--scene", a.scene,
                       "--start", str(a.start), "--svc-dir", LC.dirs(a.root, tg)["svc"], "--resume-t", str(resume_t)]
                env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
                env["MACARONS_PLUSPLUS_ROOT"] = UH.UAV_ROOT
                if a.planner_gpu is not None:
                    env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
                self.err = open(err_path, "a")
                self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
                self.hello = self._read()
                if not self.hello.get("ready"):
                    raise RuntimeError(f"planner did not start: {self.hello}")
        VS.Planner = MagPlanner
        sys.argv[1:1] = ["--group", "MAG-B"] if "--group" not in sys.argv else []
        T24.main()
        return
    os.environ["V3_CKPT"] = CK
    os.environ["D_POLICY"] = "d1"

    class Planner(T6S.PlannerClient):
        def __init__(self, a, resume_t, err_path):
            g = CFG["groups"][a.exp]
            cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "uav_d_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
                   "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
                   "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
            env = T6.seen2scene_env(os.environ)
            env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "T3_P1_MODE": "b2", "V3_CKPT": CK, "D_POLICY": "d1",
                        "D_CONFIG": UH.D_CONFIG_UAV, "MACARONS_PLUSPLUS_ROOT": UH.UAV_ROOT})
            if a.planner_gpu is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
            self.err = open(err_path, "a")
            self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
            self.hello = self._read()
            if not self.hello.get("ready"):
                raise RuntimeError(f"planner did not start: {self.hello}")
    VS.Planner = Planner
    sys.argv[1:1] = ["--group", "V2M"] if "--group" not in sys.argv else []
    T24.main()


if __name__ == "__main__":
    main()
