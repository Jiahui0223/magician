"""E3 closed loop (magician env): frozen M-protocol simulator (task2 t24_out_sim -> task1 v2_sim.run, unchanged) with the planner
child e3_planner.py (D-v1 d1 + D-v2 switch). Same Planner construction as sim_out_d.Planner, only the child path / env differ.
Usage: python e3_sim.py --v3-ckpt PATH --dv2 off|full|first --scene S --start K --rep N --root DIR"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E3 = os.path.dirname(HERE)
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(E3))))
DD = os.path.join(X, "exp6_v3", "ablation_study", "experiment_D_h3_lookahead")
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t24_out_sim as T24  # noqa: E402

LC, C, CFG, T6, T6S, VS = T24.LC, T24.C, T24.CFG, T24.T6, T24.T6S, T24.VS


class Planner(T6S.PlannerClient):
    def __init__(self, a, resume_t, err_path):
        g = CFG["groups"][a.exp]
        cmd = [LC.SEEN2SCENE_PYTHON, os.path.join(HERE, "e3_planner.py"), "--exp", a.exp, "--scene", a.scene,
               "--start", str(a.start), "--rep", str(a.rep), "--root", a.root, "--resume-t", str(resume_t),
               "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"]]
        env = T6.seen2scene_env(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["T3_P1_MODE"] = "b2"
        env["V3_CKPT"] = os.environ["V3_CKPT"]
        env["D_POLICY"] = "d1"
        env["E3_DV2"] = os.environ["E3_DV2"]
        env["E3_DV2_CONFIG"] = os.environ.get("E3_DV2_CONFIG", os.path.join(E3, "configs", "dv2_v2.yaml"))
        env["D_CONFIG"] = os.path.join(DD, "configs", "d_config.yaml")
        self.err = open(err_path, "a")
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1, cwd=LC.REPO, env=env)
        self.hello = self._read()
        if not self.hello.get("ready"):
            raise RuntimeError(f"planner did not start: {self.hello}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v3-ckpt", required=True)
    ap.add_argument("--dv2", required=True, choices=("off", "full", "first"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    os.environ["V3_CKPT"] = os.path.abspath(a.v3_ckpt)
    os.environ["E3_DV2"] = a.dv2
    VS.Planner = Planner
    ns = argparse.Namespace(exp="V2M", group="V2M", scene=a.scene, start=a.start, rep=a.rep, root=os.path.abspath(a.root), moves=100, planner_gpu=None)
    VS.run(ns)


if __name__ == "__main__":
    main()
