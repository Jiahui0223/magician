"""EXP9-SCONE -- MP3D-150 indoor closed loop, simulator side (magician env; GT side).

= experiments/exp5_future_view/task2/scripts/t23_in_sim.py main() run UNCHANGED (T0.12 grid simulator through t12_sim / mp3d_sim:
grid, start snapping, official grid neighbours, GT single-segment legality, frozen depth sensor, 10 % sampling seeds, C_obs
evaluator, service-record format, 150 moves), with two in-memory additions of this wrapper only (no frozen file is edited; same
pattern as fisherrf_reproduction/scripts/fisherrf_sim_in.py):
  1. its --group option also accepts "SCONE";
  2. t23.child is replaced by the scone_planner_grid.py child (magician env, python -I, LD_PRELOAD / PYTHONPATH removed).
SCONE receives depth only (the MP3D sensor renders no RGB; nothing else is added to the service record).
Usage: python scone_sim_in.py --scan S --level L --start K --moves N --out DIR [--diag-steps 0 5 10]
Output: DIR/SCONE/<level>_S<start>/{svc/, episode.json, planner.jsonl, planner.err, planner_info.json[, diag/]}
"""
import json
import os
import subprocess
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
X = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))           # experiments/
sys.path.insert(0, os.path.join(X, "exp5_future_view", "task2", "scripts"))
import t23_in_sim as T23  # noqa: E402

GROUP = "SCONE"
MAGICIAN_PYTHON = "/bsuscratch/jiahuizhang/envs/magician/bin/python"
CTX = {"diag_steps": []}


def child(group, key, tag, od, cfg):
    assert group == GROUP, group
    env = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH")}
    cmd = [MAGICIAN_PYTHON, "-I", os.path.join(HERE, "scone_planner_grid.py"), "--card", key, "--tag", tag]
    if CTX["diag_steps"]:
        cmd += ["--diag-dir", os.path.join(od, "diag"), "--diag-steps"] + [str(x) for x in CTX["diag_steps"]]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(os.path.join(od, "planner.err"), "w"),
                         text=True, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello
    return p, hello


class _AP(T23.argparse.ArgumentParser):
    def add_argument(self, *a, **k):
        if a and a[0] == "--group":
            k["choices"] = list(k["choices"]) + [GROUP]
        return super().add_argument(*a, **k)


if __name__ == "__main__":
    if "--diag-steps" in sys.argv:
        i = sys.argv.index("--diag-steps")
        j = i + 1
        while j < len(sys.argv) and not sys.argv[j].startswith("--"):
            j += 1
        CTX["diag_steps"] = [int(v) for v in sys.argv[i + 1:j]]
        del sys.argv[i:j]
    T23.child = child
    T23.argparse = types.SimpleNamespace(ArgumentParser=_AP)
    sys.argv += ["--group", GROUP]
    T23.main()
