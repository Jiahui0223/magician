"""EXP6 V5.2-M (Scratch70k) -- Macarons++ closed loop (magician env): exp6_v4/scripts/sim_out_v4.py with the planner child
v52_out_planner.py. Usage: python sim_out_v52.py --v4-ckpt V52_CKPT --lam L --group V2M --scene S --start K --root DIR"""
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
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "v52m_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = os.environ.get("V4_P1_MODE", "b2")
        env["V4_CKPT"] = os.environ["V4_CKPT"]
        env["V52_LAMBDA"] = os.environ["V52_LAMBDA"]
        env["V52_STEPLOG"] = os.path.join(a.root, "v52_steps", f"V2M_{a.scene}_s{a.start}_r{a.rep}.jsonl")
        if os.environ.get("V52_SCORE_CFG"):
            env["V52_SCORE_CFG"] = os.environ["V52_SCORE_CFG"]
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
    i = sys.argv.index("--lam")
    os.environ["V52_LAMBDA"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
