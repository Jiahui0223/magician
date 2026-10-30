"""EXP5 Task 3 / E1 -- Macarons++ closed loop (magician env): task2/scripts/t24_out_sim.py -> task1/scripts/v2_sim.run
unchanged (official settings grid, legal list, hover, seeded observation, official coverage, per-step checkpoints,
100 moves) with only the V2 planner child replaced by t32_out_planner.py.
Usage: python t33_out_sim.py --p1 sparse|full --group V2M --scene S --start K --root DIR [--planner-gpu G]"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        assert a.exp in ("V2O", "V2M"), a.exp
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "t32_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = os.environ["T3_P1_MODE"]
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
    os.environ["T3_P1_MODE"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    VS.Planner = Planner
    T24.main()
