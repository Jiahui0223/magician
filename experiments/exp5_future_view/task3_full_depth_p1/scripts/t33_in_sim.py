"""EXP5 Task 3 / E1 -- MP3D closed loop (magician env): task2/scripts/t23_in_sim.py run unchanged (grid, legality,
start snapping, sensor, sampling seeds, C_obs, service records, 150 moves) with only the V2 planner child replaced by
t32_in_planner.py. Usage: python t33_in_sim.py --p1 sparse|full --scan S --level L --start K --group V2M --out DIR"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP5 = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, S2E = T23.S, T23.C, T23.S2E


def child(group, key, tag, od, cfg):
    assert group in ("V2O", "V2M"), group
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO)
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "t32_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {})["t3_p1_mode"] = env["T3_P1_MODE"]
    return p, hello


if __name__ == "__main__":
    i = sys.argv.index("--p1")
    os.environ["T3_P1_MODE"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    T23.child = child
    T23.main()
