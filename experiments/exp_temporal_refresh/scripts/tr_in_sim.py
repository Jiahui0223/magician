"""EXP Temporal Refresh -- MP3D closed loop (magician env): task3 t33_in_sim.py (task2 t23_in_sim unchanged) with the planner
child replaced by tr_in_planner.py. Usage: python da_in_sim.py --p1 s10s10|d100s10|d100d100|s10d100 --scan S --level L
--start K --group V2M --out DIR   (checkpoint override: env T3_CKPT)"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "scripts"))
import t33_in_sim as T33  # noqa: E402

T23, S, C, S2E = T33.T23, T33.S, T33.C, T33.S2E


def child(group, key, tag, od, cfg):
    assert group == "V2M", group
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "tr_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {})["da_p1_mode"] = env["DA_P1_MODE"]
    hello["info"]["t3_ckpt"] = env.get("T3_CKPT")
    return p, hello


if __name__ == "__main__":
    i = sys.argv.index("--p1")
    os.environ["DA_P1_MODE"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    T23.child = child
    T23.main()
