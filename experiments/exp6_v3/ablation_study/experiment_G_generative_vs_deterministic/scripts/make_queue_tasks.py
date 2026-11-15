"""Experiment G -- writes task files for the shared exclusive-GPU queue (work_dirs/cvpr_results/gpu_queue/pending/expG_*.sh; one task =
one bash script run by experiments/cvpr_results/scripts/gpu_queue_worker.sh on one GPU; tasks are taken in name order).
  smoke      expG_a_*  closed-loop smokes (G1R regression vs V3-T0-K1D2, G0, D0 with a given checkpoint) + one fixed-state test
  fixed      expG_c_*  fixed-state workers (g_fixed.py over confirm then explore; trajectories claimed with O_EXCL, any number of workers)
  closed     expG_d_*  one task per D0 / G0 episode (All-14 x 5 official starts, 100 moves, official replay), dunnottar first
Nothing is enqueued unless --write is given (default: dry run that prints the task names).
Usage: python make_queue_tasks.py smoke --d0-ckpt PATH [--only NAME,NAME] [--write]
       python make_queue_tasks.py fixed [--workers 8] [--d0-ckpt PATH] [--write]
       python make_queue_tasks.py closed [--arms D0,G0] [--write]"""
import argparse
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
Q = os.path.join(R, "work_dirs/cvpr_results/gpu_queue")
D0_FINAL = os.path.join(R, "work_dirs/exp6_v3/experiment_G/d0/checkpoints/final.pt")
sys.path.insert(0, os.path.join(R, "experiments/cvpr_results/scripts"))
SCENE_ORDER = ["dunnottar", "colosseum", "bannerman", "pantheon", "redeemer", "liberty", "pisa", "fushimi", "alhambra", "neuschwanstein",
               "eiffel", "bridge", "barts", "sestino_museum"]        # = cvpr_lib.SCENE_ORDER (frozen)


def task(name, comment, cmd):
    return name, f"#!/bin/bash\n# Experiment G (protocol G-v1): {comment}\n{cmd}\n"


def smoke(a):
    S = f"bash {GD}/scripts/g_smoke.sh"
    fx = (f"cd {R} && S2E=/bsuscratch/jiahuizhang/envs/seen2scene && export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH={R} "
          f"CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 && "
          f"$S2E/bin/python {GD}/scripts/g_fixed.py --set confirm --only conf_liberty_s0 --tmax 5 --d0-ckpt {a.d0_ckpt} "
          f"--out {R}/work_dirs/exp6_v3/experiment_G/smoke/fixed_test")
    return [task("expG_a1_smoke_g1r_liberty_s0", "regression: flow K = 1 must reproduce V3-T0-K1D2 liberty s0 (10 moves) bitwise",
                 f"{S} G1R liberty 0 10 g1r_liberty_s0 V3-T0-K1D2"),
            task("expG_a2_smoke_d0_liberty_s0", f"D0 smoke checkpoint {a.d0_ckpt}, 10 moves", f"{S} D0 liberty 0 10 d0_liberty_s0 '' {a.d0_ckpt}"),
            task("expG_a3_smoke_g0_liberty_s0", "G0 (V3-T0, zero noise), 10 moves", f"{S} G0 liberty 0 10 g0_liberty_s0"),
            task("expG_a4_fixed_test", "fixed-state test: conf_liberty_s0 t <= 5, arms G4 / G1 / G0 / D0 (smoke ckpt); G4 must equal the Table 4 T0 rows", fx)]


def fixed(a):
    out = []
    for w in range(1, a.workers + 1):
        out.append(task(f"expG_c_fixed_w{w}", f"fixed-state worker {w} (confirm then explore; D0 = {a.d0_ckpt})",
                        f"bash {GD}/scripts/g_fixed_worker.sh {w} --d0-ckpt {a.d0_ckpt}"))
    return out


def closed(a):
    out = []
    for arm in a.arms.split(","):
        assert arm in ("D0", "G0"), arm
        for i, s in enumerate(SCENE_ORDER):
            for k in range(5):
                out.append(task(f"expG_d_{arm}_{i:02d}_{s}_s{k}", f"closed loop {arm} {s} start {k}, 100 moves + official replay",
                                f"bash {GD}/scripts/g_episode.sh {arm} {s} {k}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("smoke", "fixed", "closed"))
    ap.add_argument("--d0-ckpt", default=D0_FINAL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", default="D0,G0")
    ap.add_argument("--only", default=None, help="comma-separated task names to keep")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    if a.what in ("fixed", "closed") and "D0" in (a.arms if a.what == "closed" else "D0") and not os.path.exists(a.d0_ckpt if a.what == "fixed" else D0_FINAL):
        sys.exit(f"D0 checkpoint missing ({a.d0_ckpt if a.what == 'fixed' else D0_FINAL}): run g_select_final.py after training first")
    T = {"smoke": smoke, "fixed": fixed, "closed": closed}[a.what](a)
    if a.only:
        T = [t for t in T if t[0] in a.only.split(",")]
    os.makedirs(os.path.join(Q, "pending"), exist_ok=True)
    for name, body in T:
        if any(os.path.exists(os.path.join(Q, d, name + ".sh")) for d in ("pending",)) or any(f.startswith(name + "__") for d in ("running", "done") for f in os.listdir(os.path.join(Q, d))):
            print("exists", name)
            continue
        if a.write:
            p = os.path.join(Q, "pending", name + ".sh")
            with open(p + ".tmp", "w") as f:
                f.write(body)
            os.replace(p + ".tmp", p)
        print(("queued " if a.write else "dry-run ") + name)
    print(len(T), "tasks", "written" if a.write else "(dry run; add --write)")


if __name__ == "__main__":
    main()
