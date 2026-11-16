"""Experiment G3 -- Macarons++ closed loop (magician env; protocol G3-v1): task2 t24_out_sim.py unchanged (frozen M protocol), planner child
g3_out_planner.py (G3_MODE off | raw | depth). Same construction as sim_out_g.py / cvpr_results table3 sim_out_kd.py.
Usage: python sim_out_g3.py --v3-ckpt PATH --g3-mode off|raw|depth --group V2M --scene S --start K [--moves N] --root DIR"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
D_DIR = os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead")


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "g3_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = "b2"
        env["V3_CKPT"] = os.environ["V3_CKPT"]
        env["KD_NOISE_OFFSET"] = "0"
        env["D_POLICY"] = "d1"
        env["D_CONFIG"] = os.path.join(D_DIR, "configs", "d_config.yaml")
        env["G3_MODE"] = os.environ["G3_MODE"]
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    for flag, env in (("--v3-ckpt", "V3_CKPT"), ("--g3-mode", "G3_MODE")):
        i = sys.argv.index(flag)
        os.environ[env] = os.path.abspath(sys.argv[i + 1]) if env == "V3_CKPT" else sys.argv[i + 1]
        del sys.argv[i:i + 2]
    assert os.environ["G3_MODE"] in ("off", "raw", "depth"), os.environ["G3_MODE"]
    VS.Planner = Planner
    T24.main()
