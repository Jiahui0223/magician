"""EXP Density A/B -- Macarons++ closed loop (magician env): task3 t33_out_sim.py (task2 t24_out_sim -> task1 v2_sim
unchanged) with the planner child replaced by da_out_planner.py. Usage: python da_out_sim.py --p1 MODE --group V2M
--scene S --start K --root DIR   (checkpoint override: env T3_CKPT)"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t33_out_sim as T33  # noqa: E402

T24, LC, C, CFG, T6, T6S, VS = T33.T24, T33.LC, T33.C, T33.CFG, T33.T6, T33.T6S, T33.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        assert a.exp == "V2M", a.exp
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "da_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["DA_P1_MODE"] = os.environ["DA_P1_MODE"]
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


if __name__ == "__main__":
    i = sys.argv.index("--p1")
    os.environ["DA_P1_MODE"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
