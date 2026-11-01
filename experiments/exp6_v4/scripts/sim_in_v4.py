"""EXP6 V4 -- MP3D closed loop (magician env): task2/scripts/t23_in_sim.py unchanged, planner child = v4_in_planner.py
(B2-style P1, V4 checkpoint). Usage: python sim_in_v4.py --v4-ckpt PATH --scan S --level L --start K --group V2M --out DIR"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, S2E = T23.S, T23.C, T23.S2E


def child(group, key, tag, od, cfg):
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, T3_P1_MODE="b2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "v4_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {}).update({"t3_p1_mode": "b2", "v4_ckpt": env["V4_CKPT"]})
    return p, hello


if __name__ == "__main__":
    i = sys.argv.index("--v4-ckpt")
    os.environ["V4_CKPT"] = os.path.abspath(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    T23.child = child
    T23.main()
