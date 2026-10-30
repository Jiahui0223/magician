"""EXP5 Task 2 -- outdoor (Macarons++) closed loop of the final comparison (magician env; simulator / GT side).

= the Task 1.4 simulator (task1/scripts/v2_sim.py: frozen M-protocol EpisodeSim -- official settings grid, official
neighbours -> get_valid_neighbors -> GT segment test, hover at dead ends, seeded observation and the MAGICIAN-style
coverage; per-step checkpoint and bitwise resume) run unchanged, with the Task 2 groups and planner children:
  V2O / V2M     t24_v2_out_planner.py (seen2scene env)  checkpoints / sha256 from task2/configs/eval_outdoor_v1.json
  MAG-G / MAG-B t24_mag_out_planner.py (magician env)   official e697009 core
Budget: C0 + 100 moves = 101 frames. Output <root> = W2/eval/outdoor/closed_loop/<group>/ (episodes/, svc/, ckpt/, ...).
Usage: python t24_out_sim.py --group G --scene S --start K [--moves N] [--root DIR]
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.EXP5, "task1", "scripts"))
import v2_sim as VS  # noqa: E402

LC, T6, T6S = VS.LC, VS.T6, VS.T6S
CFG = json.load(open(os.path.join(C.T2, "configs", "eval_outdoor_v1.json")))
for g, d in CFG["groups"].items():
    LC.ARMS[g] = {"name": f"task2_{g}", "score": d.get("score", "magician"), "K": d.get("K"), "refresh": "R10"}
    if g not in LC.RUN_ORDER:
        LC.RUN_ORDER.append(g)


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        tg = LC.tag(a.exp, a.scene, a.start, a.rep)
        svc = LC.dirs(a.root, tg)["svc"]
        if a.exp.startswith("MAG-"):
            cmd = [LC.MAGICIAN_PYTHON, "-I", os.path.join(HERE, "t24_mag_out_planner.py"), "--method", a.exp, "--scene", a.scene,
                   "--start", str(a.start), "--svc-dir", svc, "--resume-t", str(resume_t)]
            env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
        else:
            cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "t24_v2_out_planner.py"), "--exp", a.exp, "--scene", a.scene,
                   "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
                   "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
            env = T6.seen2scene_env(os.environ)
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        if a.planner_gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(a.planner_gpu)
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True,
                                  bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


VS.Planner = Planner


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(CFG["groups"]))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--moves", type=int, default=CFG["budget"]["moves"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--planner-gpu", type=int, default=None)
    a = ap.parse_args()
    a.exp, a.rep = a.group, 0
    a.root = a.root or os.path.join(C.W2, "eval", "outdoor", "closed_loop", a.group)
    VS.run(a)


if __name__ == "__main__":
    main()
