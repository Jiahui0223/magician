"""Experiment U Phase E -- queue task generator (no GPU; does NOT launch anything unless --enqueue is given).
One task = one 100-move episode + official coverage replay (scripts/u_episode.sh; idempotent, skip-if-done, claim-safe).
Task files expU_e_<policy>_<NN>_<scene>_s<k>.sh (NN = frozen scene order index, so the shared queue takes them policy by policy,
scene by scene) are written to work_dirs/exp6_v3/experiment_U/staging/phaseE/ and, with --enqueue, moved atomically into the shared
queue pending/ directory. Episodes already finished (valid replay metrics) or already present in pending/ / running/ are skipped.
Usage: python make_queue_tasks.py --policies u1,u2,u3 [--scenes all|s1,s2] [--starts 0,1,2,3,4] [--enqueue]"""
import argparse
import glob
import json
import os
import shutil
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "cvpr_results", "scripts"))
import cvpr_lib as L  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
Q = os.path.join(R, "work_dirs", "cvpr_results", "gpu_queue")
WU = os.path.join(R, "work_dirs", "exp6_v3", "experiment_U")
POL = ("c1", "u1", "u2", "u3")


def done(p, s, k):
    m = os.path.join(WU, "closed_loop", p, f"{s}_s{k}", "metrics", "V2M", f"V2M_{s}_s{k}_r0.json")
    if not os.path.exists(m):
        return False
    j = json.load(open(m))
    return bool(j.get("official_equal_stored")) and j.get("moves") == 100 and len(j["cov_official"]) == 101


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", required=True)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--starts", default="0,1,2,3,4")
    ap.add_argument("--enqueue", action="store_true")
    a = ap.parse_args()
    pols = a.policies.split(",")
    assert all(p in POL for p in pols), pols
    scenes = L.SCENE_ORDER if a.scenes == "all" else a.scenes.split(",")
    assert all(s in L.SCENE_ORDER for s in scenes), scenes
    starts = [int(x) for x in a.starts.split(",")]
    st = os.path.join(WU, "staging", "phaseE")
    os.makedirs(st, exist_ok=True)
    queued = {os.path.basename(f).split("__")[0].replace(".sh", "") for f in glob.glob(os.path.join(Q, "pending", "*.sh")) + glob.glob(os.path.join(Q, "running", "*.sh"))}
    n_new = n_done = n_q = 0
    for p in pols:
        for s in scenes:
            for k in starts:
                name = f"expU_e_{p}_{L.SCENE_ORDER.index(s):02d}_{s}_s{k}"
                if done(p, s, k):
                    n_done += 1
                    continue
                if name in queued:
                    n_q += 1
                    continue
                f = os.path.join(st, name + ".sh")
                with open(f, "w") as fh:
                    fh.write(f"#!/bin/bash\n# Experiment U Phase E: policy {p}, {s} start {k}, 100 moves + official coverage replay (protocol U-v1)\n"
                             f"bash {HERE}/u_episode.sh {p} {s} {k}\n")
                n_new += 1
                if a.enqueue:
                    shutil.move(f, os.path.join(Q, "pending", name + ".sh"))
    print(f"policies {pols}: new {n_new} ({'ENQUEUED' if a.enqueue else 'staged only, not enqueued'} -> {Q if a.enqueue else st}), already done {n_done}, already queued {n_q}")


if __name__ == "__main__":
    main()
