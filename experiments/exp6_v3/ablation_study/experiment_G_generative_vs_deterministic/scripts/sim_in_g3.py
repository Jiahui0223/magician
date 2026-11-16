"""Experiment G3 -- MP3D closed loop (magician env; protocol G3-v1): task2 t23_in_sim.py unchanged, child = g3_in_planner.py
(the EXP7 OURS child + the m_combine hooks). Same construction as experiments/cvpr_results/exp7/scripts/sim_in_d.py; the EXP7 D config
(configs/d_config_mp3d.yaml) and pose tables are used unchanged.
Usage: python sim_in_g.py --v3-ckpt PATH --g3-mode off|raw|depth --scan S --level L --start K --group V2M --out DIR [--moves N]"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE))))          # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, S2E = T23.S, T23.C, T23.S2E
D_CONFIG = os.path.join(X, "cvpr_results", "exp7", "configs", "d_config_mp3d.yaml")


def child(group, key, tag, od, cfg):
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, T3_P1_MODE="b2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", D_POLICY="d1", D_CONFIG=D_CONFIG)
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "g3_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {}).update({"t3_p1_mode": "b2", "v3_ckpt": env["V3_CKPT"], "d_policy": "d1", "d_config": D_CONFIG,
                                         "g3_mode": env["G3_MODE"], "protocol": "G3-v1"})
    return p, hello


if __name__ == "__main__":
    for flag, env in (("--v3-ckpt", "V3_CKPT"), ("--g3-mode", "G3_MODE")):
        i = sys.argv.index(flag)
        os.environ[env] = os.path.abspath(sys.argv[i + 1]) if env == "V3_CKPT" else sys.argv[i + 1]
        del sys.argv[i:i + 2]
    assert os.environ["G3_MODE"] in ("off", "raw", "depth")
    T23.child = child
    T23.main()
