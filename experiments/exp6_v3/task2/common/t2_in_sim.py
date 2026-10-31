"""Task 2 -- MP3D closed loop (magician env): task2/scripts/t23_in_sim.py unchanged, planner child = t2_in_planner.py.
Usage: python t2_in_sim.py --model B2|T1 --scorer S0|... [--frontier none|bonus|trigger --lambda L --theta TH]
                           --scan S --level L --start K --out DIR [--moves N]
The episode is written to <out>/V2M/<tag>/ (group V2M = the V2-Mix checkpoint for B2; T1 = V3 checkpoint override)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
E6 = os.path.dirname(os.path.dirname(HERE))
REPO = os.path.dirname(os.path.dirname(E6))
EXP5 = os.path.join(REPO, "experiments", "exp5_future_view")
sys.path.insert(0, os.path.join(EXP5, "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

S, C, S2E = T23.S, T23.C, T23.S2E
T1_CKPT = os.path.join(REPO, "work_dirs", "exp6_v3", "task1_xattn", "checkpoints", "final.pt")
OPT = {}


def child(group, key, tag, od, cfg):
    g = cfg["groups"][group]
    env = dict(os.environ, LD_PRELOAD=f"{S2E}/lib/libtbb.so", LD_LIBRARY_PATH=f"{S2E}/lib", PYTHONPATH=S.REPO, T3_P1_MODE="b2",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", T2_SCORER=OPT["scorer"], T2_FRONTIER=OPT["frontier"],
               T2_LAMBDA=str(OPT["lam"]), T2_THETA=str(OPT["theta"]), T2_LOG=os.path.join(od, "t2_steps.jsonl"))
    if OPT["model"] == "T1":
        env["V3_CKPT"] = T1_CKPT
    cmd = [f"{S2E}/bin/python", os.path.join(HERE, "t2_in_planner.py"), "--group", group, "--card", key, "--tag", tag,
           "--ckpt", os.path.join(C.REPO, g["checkpoint"]), "--sha256", g["sha256"], "--svc-dir", os.path.join(od, "svc")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"), text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello
    hello.setdefault("info", {}).update({"t3_p1_mode": "b2", "t2": dict(OPT), "v3_ckpt": env.get("V3_CKPT")})
    return p, hello


def pop(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


if __name__ == "__main__":
    OPT.update({"model": pop("--model", "B2"), "scorer": pop("--scorer", "S0"), "frontier": pop("--frontier", "none"),
                "lam": float(pop("--lambda", "0")), "theta": float(pop("--theta", "0"))})
    sys.argv += ["--group", "V2M"]
    T23.child = child
    T23.main()
