"""EXP6 V3 -- Macarons++ closed loop (magician env): task2/scripts/t24_out_sim.py unchanged, planner child =
v3_out_planner.py. Usage: python sim_out_v3.py --v3-ckpt PATH --group V2M --scene S --start K --root DIR"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "v3_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = "b2"
        env["V3_CKPT"] = os.environ["V3_CKPT"]
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    i = sys.argv.index("--v3-ckpt")
    os.environ["V3_CKPT"] = os.path.abspath(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
