"""Unified outdoor-only benchmark -- entry point of every arm (seen2scene env; torchrun, 4 ranks).

  torchrun --standalone --nproc_per_node 4 experiments/exp_unified_outdoor/scripts/train_unified.py --arm <arm> \
      [--out work_dirs/exp_unified_outdoor/runs/<arm>] [extra historical-trainer args, e.g. --smoke short]

Loads configs/<arm>.json, applies the uo_common patches (exact per-rank RNG resume, data roots, V3-T1 switch + pooled
sampler, V4@48k parent of the V4C / V5 arms) and calls the UNCHANGED historical trainer main() with
--config <abs config> --out <dir> (+ --arm v5|v4c for train_v5). Resume is automatic: every historical trainer resumes
from <out>/ckpt_last.pt; the patch restores the per-rank RNG of that step from <out>/rng_states/.
Rank 0 appends one line per launch to <out>/unified_launches.jsonl (arm, config sha256, patches, git, host, GPUs).
"""
import argparse
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import uo_common as U  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(U.ARMS))
    ap.add_argument("--out", default=None)
    a, extra = ap.parse_known_args()
    arm = U.ARMS[a.arm]
    out = a.out or os.path.join(U.RUNS, a.arm)
    cfg_p = arm["config"]
    cfg = json.load(open(cfg_p))
    assert cfg["unified"]["arm"] == a.arm, (cfg["unified"]["arm"], a.arm)
    tr = arm["trainer"]
    U.add_paths(tr)
    rank = U.env_rank()
    say = (lambda *x: print(time.strftime("%H:%M:%S"), "[unified]", *x, flush=True)) if rank == 0 else (lambda *x: None)  # noqa: E731
    patches = {}
    if tr == "v2":
        patches.update(U.patch_v2(cfg, say))
    elif tr == "v3t1":
        patches.update(U.patch_v3t1(cfg, say))
    elif tr == "v5":
        patches.update(U.patch_v5(cfg, out, say))
    er = U.ExactResume(log=say)
    patches["exact_resume_hook"] = U.install_exact_resume(tr, er)
    mod = __import__(U.TRAINER_MOD[tr])
    argv = [mod.__file__, "--config", cfg_p, "--out", out]
    if tr == "v5":
        argv += ["--arm", arm["v5_arm"]]
    argv += extra
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        rec = {"time": time.strftime("%F %T"), "arm": a.arm, "trainer": os.path.relpath(mod.__file__, U.REPO), "argv": argv[1:],
               "config": os.path.relpath(cfg_p, U.REPO), "config_sha256": U.sha256_file(cfg_p), "density": arm["density"],
               "data_root": cfg.get("data_root"), "patches": patches, "git": U.git_sha(), "host": socket.gethostname(),
               "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
               "world": int(os.environ.get("WORLD_SIZE", "1")), "resume_from_existing_ckpt": os.path.exists(os.path.join(out, "ckpt_last.pt"))}
        with open(os.path.join(out, "unified_launches.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        say("arm", a.arm, "trainer", rec["trainer"], "patches", json.dumps(patches))
    sys.argv = argv
    mod.main()
    if rank == 0:
        say(f"trainer returned; RNG snapshots written this launch: {er.n_snap}")


if __name__ == "__main__":
    main()
