"""Task 2 -- Macarons++ closed loop (magician env): task2/scripts/t24_out_sim.py -> v2_sim.run unchanged, planner child =
t2_out_planner.py. Usage: python t2_out_sim.py --model B2|T1 --scorer S0|... --scene S --start K --root DIR [--moves N]
(group V2M; episode tag V2M_<scene>_s<start>_r0 under <root>)."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(E6))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS
T1_CKPT = os.path.join(REPO, "work_dirs", "exp6_v3", "task1_xattn", "checkpoints", "final.pt")
OPT = {}


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "t2_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "T3_P1_MODE": "b2", "T2_SCORER": OPT["scorer"], "T2_FRONTIER": "none",
                    "T2_LOG": os.path.join(a.root, "planner_logs", f"t2_{LC.tag(a.exp, a.scene, a.start, a.rep)}.jsonl")})
        if OPT["model"] == "T1":
            env["V3_CKPT"] = T1_CKPT
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        os.makedirs(os.path.join(a.root, "planner_logs"), exist_ok=True)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")
        self.hello["t2"] = dict(OPT)


def pop(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    OPT.update({"model": pop("--model", "B2"), "scorer": pop("--scorer", "S0")})
    sys.argv += ["--group", "V2M"]
    VS.Planner = Planner
    T24.main()
