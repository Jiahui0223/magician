"""Camera Pose Phase 1 -- MP3D closed loop for C0 / C1 (magician env): Task 3 t33_in_sim.py -> Task 2 t23_in_sim.py
unchanged (1 m grid, 5 pitch x 8 yaw, GT segment legality, start snapping, sensor, seeds, C_obs, 150 moves) with only
the planner child replaced by c1_in_planner.py; step log <episode dir>/c1_steps.jsonl.
Usage: CP_LAMBDA=x [T3_CKPT=ckpt] python c1_in_sim.py --p1 b2 --scan S --level L --start K --group V2M --out DIR"""
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
    assert group in ("V2O", "V2M"), group
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, CP_STEPLOG=os.path.join(od, "c1_steps.jsonl"))
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "c1_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {})["t3_p1_mode"] = env["T3_P1_MODE"]
    hello["info"]["cp_lambda"] = float(env.get("CP_LAMBDA", "0"))
    return p, hello


if __name__ == "__main__":
    i = sys.argv.index("--p1")
    os.environ["T3_P1_MODE"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    T23.child = child
    T23.main()
