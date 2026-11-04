"""EXP6 V3 / H -- Macarons++ closed loop (magician env): task2/scripts/t24_out_sim.py unchanged (frozen M-protocol simulator),
planner child = h_out_planner.py (V3-T1 + policy). Usage: python sim_out_h.py --v3-ckpt PATH --policy h0|h1|h2|h3|h1deg|h3d1
--group V2M --scene S --start K --root DIR   (env H_CONFIG optional; default experiments/exp6_v3/H/FROZEN_CONFIG.yaml)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "h_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = "b2"
        env["V3_CKPT"] = os.environ["V3_CKPT"]
        env["H_POLICY"] = os.environ["H_POLICY"]
        env["H_CONFIG"] = os.environ.get("H_CONFIG", os.path.join(os.path.dirname(HERE), "FROZEN_CONFIG.yaml"))
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    for flag, env in (("--v3-ckpt", "V3_CKPT"), ("--policy", "H_POLICY")):
        i = sys.argv.index(flag)
        os.environ[env] = os.path.abspath(sys.argv[i + 1]) if env == "V3_CKPT" else sys.argv[i + 1]
        del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
