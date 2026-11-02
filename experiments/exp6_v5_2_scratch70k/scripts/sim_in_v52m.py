"""EXP6 V5.2-M (Scratch70k) -- MP3D closed loop (magician env): exp6_v4/scripts/sim_in_v4.py with the planner child v52_in_planner.py.
Usage: python sim_in_v52.py --v4-ckpt V52_CKPT --lam L --scan S --level L --start K --group V2M --out DIR"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "exp6_v4", "scripts"))
import sim_in_v4 as SI4  # noqa: E402

T23, S, C, S2E = SI4.T23, SI4.S, SI4.C, SI4.S2E


def child(group, key, tag, od, cfg):
    g = cfg["groups"][group]
    mode = os.environ.get("V4_P1_MODE", "b2")
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, T3_P1_MODE=mode,
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", V52_STEPLOG=os.path.join(od, "v52_steps.jsonl"))
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "v52m_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"]
    hello.setdefault("info", {}).update({"t3_p1_mode": mode, "v4_ckpt": env["V4_CKPT"], "arch": "v52m", "lambda": float(os.environ["V52_LAMBDA"])})
    return p, hello


if __name__ == "__main__":
    i = sys.argv.index("--v4-ckpt")
    os.environ["V4_CKPT"] = os.path.abspath(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    i = sys.argv.index("--lam")
    os.environ["V52_LAMBDA"] = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    T23.child = child
    T23.main()
