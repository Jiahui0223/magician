"""Experiment G2 -- task files for the shared exclusive-GPU queue (work_dirs/cvpr_results/gpu_queue/pending/expG2_*.sh; protocol G2-v1).
  insmoke  expG2_a_in_*   indoor smokes (10 moves): G4R (wrapper K = 4, must equal EXP7 OURS), G1IN, D0IN on the shortest OURS level
  indoor   expG2_d_in_*   88 episodes (G1IN, D0IN x 44 manifest OURS rows), ordered by the OURS wall time (longest first), G1IN / D0IN interleaved
  prob     expG2_b_prob_* items 1-2 on VALSEL-in / S-c-out (g2_prob.py)
  pool     expG2_c_pool_w* items 1, 2, 4 on the 755-state F pool (g2_pool.py workers, trajectory claims)
Dry run unless --write. Usage: python g2_queue.py insmoke|indoor|prob|pool [--workers 8] [--only NAMES] [--write]"""
import argparse
import csv
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
GD = os.path.join(R, "experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic")
Q = os.path.join(R, "work_dirs/cvpr_results/gpu_queue")
E7 = os.path.join(R, "experiments/cvpr_results/exp7")
W7 = os.path.join(R, "work_dirs/cvpr_results/exp7")
PYENV = (f"cd {R} && S2E=/bsuscratch/jiahuizhang/envs/seen2scene && export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH={R} "
         f"CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1")


def ours_rows():
    rows = [r for r in csv.DictReader(open(os.path.join(E7, "manifests", "mp3d150_manifest.csv"))) if r["method"] == "OURS"]
    for r in rows:
        r["wall"] = json.load(open(os.path.join(W7, "episodes", "mp3d150", "OURS", f"{r['scene_id']}_S{r['start_id']}.json")))["wall_s"]
    return sorted(rows, key=lambda r: (-r["wall"], r["scene_id"], int(r["start_id"])))


def task(name, comment, cmd):
    return name, f"#!/bin/bash\n# Experiment G2 (protocol G2-v1): {comment}\n{cmd}\n"


def insmoke(a):
    r = sorted(ours_rows(), key=lambda r: r["wall"])[0]
    out = []
    WS = os.path.join(R, "work_dirs/exp6_v3/experiment_G/g2/indoor_smoke")
    ref = os.path.join(W7, "episodes/mp3d150/raw/OURS/V2M", f"{r['scene_id']}_S{r['start_id']}")
    for arm, mode, k in (("G4R", "flow", 4), ("G1IN", "flow", 1), ("D0IN", "d0", 1)):
        od = os.path.join(WS, "raw", arm, "V2M", f"{r['scene_id']}_S{r['start_id']}")
        chk = (f"{PYENV} && $S2E/bin/python {GD}/scripts/g2_in_smoke_check.py --od {od} --moves 10 --mode {mode} --k {k} "
               + (f"--ref-od {ref} " if arm == "G4R" else "") + f"--out {WS}/{arm}_result.json && $S2E/bin/python -c \"import json,sys; sys.exit(0 if json.load(open('{WS}/{arm}_result.json'))['pass'] else 7)\"")
        out.append(task(f"expG2_a_in_smoke_{arm}", f"indoor smoke {arm} {r['scene_id']} S{r['start_id']} 10 moves",
                        f"export G_MOVES=10 G_WORKDIR={WS}\nbash {GD}/scripts/g_in_episode.sh {arm} {r['scan']} {r['level']} {r['start_id']} || exit $?\n{chk}"))
    return out


def indoor(a):
    out = []
    for i, r in enumerate(ours_rows()):
        for arm in ("G1IN", "D0IN"):
            out.append(task(f"expG2_d_in_{i:02d}_{arm}_{r['scene_id']}_S{r['start_id']}", f"MP3D-150 {arm} {r['scene_id']} S{r['start_id']} (OURS wall {r['wall']:.0f} s)",
                            f"bash {GD}/scripts/g_in_episode.sh {arm} {r['scan']} {r['level']} {r['start_id']}"))
    return out


PYENV_TRAIN = (f"cd {R} && S2E=/bsuscratch/jiahuizhang/envs/seen2scene && export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH={R} "
               f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6 PYTHONUNBUFFERED=1")   # = the trainer's environment (train_arm.sh)


def prob(a):
    return [task(f"expG2_b_prob_{s}", f"items 1-2 on {s} (trainer environment, for the bitwise eval.jsonl regression)",
                 f"{PYENV_TRAIN} && $S2E/bin/python {GD}/scripts/g2_prob.py --set {s}") for s in ("in_val", "out_sc")]


def pool(a):
    return [task(f"expG2_c_pool_w{w}", f"items 1, 2, 4 on the F pool, worker {w}",
                 f"{PYENV} && $S2E/bin/python {GD}/scripts/g2_pool.py --set confirm && $S2E/bin/python {GD}/scripts/g2_pool.py --set explore")
            for w in range(1, a.workers + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("insmoke", "indoor", "prob", "pool"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    T = {"insmoke": insmoke, "indoor": indoor, "prob": prob, "pool": pool}[a.what](a)
    if a.only:
        T = [t for t in T if t[0] in a.only.split(",")]
    for name, body in T:
        if os.path.exists(os.path.join(Q, "pending", name + ".sh")) or any(f.startswith(name + "__") for d in ("running", "done", "failed") for f in os.listdir(os.path.join(Q, d))):
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
