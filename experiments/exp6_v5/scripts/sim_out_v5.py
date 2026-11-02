"""EXP6 V5 -- Macarons++ closed loop (magician env): exp6_v4/scripts/sim_out_v4.py with the planner child v5_out_planner.py.
Usage: python sim_out_v5.py --v4-ckpt PATH --group V2M --scene S --start K --root DIR (PATH = a V5 checkpoint)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import sim_out_v4 as SO4  # noqa: E402

T24, LC, C, CFG, T6, T6S, VS = SO4.T24, SO4.LC, SO4.C, SO4.CFG, SO4.T6, SO4.T6S, SO4.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "v5_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = os.environ.get("V4_P1_MODE", "b2")      # frozen: D100 input + S10 sparse DistP1
        env["V4_CKPT"] = os.environ["V4_CKPT"]
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    i = sys.argv.index("--v4-ckpt")
    os.environ["V4_CKPT"] = os.path.abspath(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
