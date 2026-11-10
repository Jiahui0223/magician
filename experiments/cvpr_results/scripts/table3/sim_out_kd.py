"""CVPR Table 3 (K x planning depth) -- Macarons++ closed loop (magician env): task2 t24_out_sim.py unchanged (frozen M protocol),
planner child kd_out_planner.py. Same structure as exp6_v3 .../experiment_D_h3_lookahead/scripts/sim_out_d.py.
Usage: python sim_out_kd.py --v3-ckpt PATH --k 1|4 --depth 1|2 [--noise 0|1|2|3] --group V2M --scene S --start K --root DIR
(--noise j: K = 1 only, use noise map k = j of the K = 4 set; default 0)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))                      # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
D_DIR = os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead")


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "kd_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = "b2"
        env["V3_CKPT"] = os.environ["V3_CKPT"]
        env["KD_K"] = os.environ["KD_K"]
        env["KD_NOISE_OFFSET"] = os.environ.get("KD_NOISE_OFFSET", "0")
        if os.environ["KD_DEPTH"] == "2":       # frozen D-v1 policy d1, frozen config file
            env["D_POLICY"] = "d1"
            env["D_CONFIG"] = os.path.join(D_DIR, "configs", "d_config.yaml")
        else:                                    # KD-v1 one-step policy
            env["D_POLICY"] = "depth1"
            env["D_CONFIG"] = os.path.join(HERE, "kd_config.yaml")
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    os.environ["KD_NOISE_OFFSET"] = "0"
    if "--noise" in sys.argv:
        i = sys.argv.index("--noise")
        os.environ["KD_NOISE_OFFSET"] = sys.argv[i + 1]
        del sys.argv[i:i + 2]
    for flag, env in (("--v3-ckpt", "V3_CKPT"), ("--k", "KD_K"), ("--depth", "KD_DEPTH")):
        i = sys.argv.index(flag)
        os.environ[env] = os.path.abspath(sys.argv[i + 1]) if env == "V3_CKPT" else sys.argv[i + 1]
        del sys.argv[i:i + 2]
    assert os.environ["KD_K"] in ("1", "4") and os.environ["KD_DEPTH"] in ("1", "2") and os.environ["KD_NOISE_OFFSET"] in ("0", "1", "2", "3")
    assert os.environ["KD_NOISE_OFFSET"] == "0" or os.environ["KD_K"] == "1"
    VS.Planner = Planner
    T24.main()
