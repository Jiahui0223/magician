"""EXP7 -- frozen MP3D-150 manifest (numpy only). Written once before the full runs; re-running only refreshes `status`.

Episode list = the archived EXP5 Task 0.3 rule (work_dirs/exp5_future_view/task0/manifests/indoor_scenes.json: every MP3D
official-val house, every level with a panorama graph, 2 seeded starts per level, rule fixed before any model output):
22 levels x 2 starts = 44 episodes. The 10 Task 2 / EXP6 indoor episodes (5 levels x 2 starts) are a subset.
Split label (experiments/exp5_future_view/task2/scripts/t2_common.py):
  HO     houses of the Task 2 evaluation levels (never read for training or checkpoint selection of V2M / V3)
  VALSEL the other 6 val houses (indoor checkpoint-selection / validation-monitoring data of V2M / V3; not training)
Protocol (all methods): T0.12 grid (1.0 m, 5 elev x 8 azim, official neighbours, official single-segment GT legality in the
simulator only), frozen MAGICIAN sensor 456 x 256 / 60 deg / 10 m, 150 moves (151 frames), level-scale C_obs evaluator
(fixed 300k GT surface points of the level box, fixed visible set, eps 0.08 model units) -- simulator t12_sim.py (MAG-B via
task2 t23_in_sim.py, which wraps it unchanged).
Methods:
  OURS     V3-T0@40K + D1 (scripts/sim_in_d.py)                                  new runs
  MAG-B    MAGICIAN beam adapter (task2 t23_in_sim.py --group MAG-B)              reuse the 10 Task 2 episodes, run the rest
  RANDOM   seeded uniform among unvisited legal poses (t12_sim.py RAND-NV)        reuse the 10 T0.12 episodes, run the rest
  OURS-H0  V3-T0@40K + H0 one-step (exp6_v3/shared/sim_in_v3.py; depth ablation)  reuse the 10 EXP6 episodes, run the rest
Reuse requires: 151 coverage values, moves 150, the same grid size, same start pose, and (models) the same checkpoint sha;
the Round-1 smoke reproduced the first 5 moves of MAG-B and RANDOM bitwise (report/EXP7_AUDIT.md)."""
import csv
import hashlib
import json
import os
import sys

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task2", "scripts"))
import t2_common as C  # noqa: E402

E7 = os.path.join(R, "experiments", "cvpr_results", "exp7")
W7 = os.path.join(R, "work_dirs", "cvpr_results", "exp7")
RAW = os.path.join(W7, "episodes", "mp3d150", "raw")
T0_SHA = "7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6"
MAG_SHA = "34c5247f13bedda08bb57e7e7e7a5baba53af6da7faf3d5f6bb57a27c2dbb59c"  # official trained_macarons.pth (adapter magician_core.py)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


SIM_FILES = ["experiments/exp5_future_view/task0/magician_comparison/scripts/t12_sim.py", "experiments/exp5_future_view/task0/scripts/mp3d_sim.py",
             "experiments/exp5_future_view/task0/magician_comparison/adapters/grid_env.py", "experiments/exp5_future_view/task2/scripts/t23_in_sim.py",
             "experiments/exp5_future_view/task2/configs/eval_indoor_v1.json"]
SIM_HASH = hashlib.sha256("".join(sha(os.path.join(R, f)) for f in SIM_FILES).encode()).hexdigest()[:16]
MODEL_FILES = {
    "OURS": ["experiments/cvpr_results/exp7/scripts/sim_in_d.py", "experiments/cvpr_results/exp7/scripts/d_in_planner.py",
             "experiments/cvpr_results/exp7/configs/d_config_mp3d.yaml", "experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py",
             "experiments/exp6_v3/H/scripts/h_policy.py", "experiments/exp6_v3/shared/planner_v3.py", "experiments/exp5_future_view/task2/scripts/t23_v2_in_planner.py"],
    "OURS-H0": ["experiments/exp6_v3/shared/sim_in_v3.py", "experiments/exp6_v3/shared/v3_in_planner.py", "experiments/exp6_v3/shared/planner_v3.py",
                "experiments/exp5_future_view/task2/scripts/t23_v2_in_planner.py"],
    "MAG-B": ["experiments/exp5_future_view/task0/magician_comparison/adapters/magician_planner_grid.py"],
    "RANDOM": ["experiments/exp5_future_view/task0/magician_comparison/scripts/t12_sim.py"],
}
MODEL_HASH = {m: hashlib.sha256("".join(sha(os.path.join(R, f)) for f in fs).encode()).hexdigest()[:16] for m, fs in MODEL_FILES.items()}
OUT_DIR = {"OURS": ("OURS", "V2M"), "OURS-H0": ("OURS-H0", "V2M"), "MAG-B": ("MAG-B", "MAG-B"), "RANDOM": ("RANDOM", "G-RAND")}
ARCHIVE = {"MAG-B": os.path.join(R, "work_dirs/exp5_future_view/task2/eval/indoor/closed_loop/MAG-B"),
           "RANDOM": os.path.join(R, "work_dirs/exp5_future_view/task0/magician_comparison/grid_protocol/closed_loop/G-RAND"),
           "OURS-H0": os.path.join(R, "work_dirs/exp6_v3/eval/closed_loop/indoor/T0/V2M")}
SEED = {"OURS": "noise keys exp5-V2|task0cl|<tag>|t<t_r>|k<k>; NV fallback T2|V2M|<tag>", "OURS-H0": "same as OURS",
        "MAG-B": "MAGICIAN adapter seeds (deterministic, smoke-verified)", "RANDOM": "sha256(exp5-T0.12|rand|<tag>|<t>)"}
METHODS = ["OURS", "MAG-B", "RANDOM", "OURS-H0"]


def archived_ok(m, tag, grid_n):
    p = os.path.join(ARCHIVE[m], tag, "episode.json")
    if not os.path.exists(p):
        return None
    e = json.load(open(p))
    ok = len(e["coverage"]) == 151 and e.get("moves", 150) == 150 and list(e["grid"]["n"]) == list(grid_n)
    if m == "OURS-H0":
        info = json.load(open(os.path.join(ARCHIVE[m], tag, "planner_info.json")))
        ok &= info.get("ckpt_sha256") == T0_SHA and info.get("t3_p1_mode") == "b2"
    return p if ok else None


def main():
    sc = json.load(open(os.path.join(C.W0, "manifests", "indoor_scenes.json")))
    import numpy as np
    sys.path.insert(0, os.path.join(R, "experiments", "exp5_future_view", "task0", "magician_comparison", "adapters"))
    import grid_env as GE
    rows = []
    t2 = {(s, lv, st) for s, lv, st in C.EVAL_LEVELS}
    for scan in sorted(sc["houses"]):
        for lv, info in sorted(sc["houses"][scan].items(), key=lambda kv: int(kv[0])):
            if "starts_pano_idx" not in info:
                continue
            key = f"{scan}_L{lv}"
            card = json.load(open(os.path.join(C.W0, "preprocessing", "scenes", f"{key}.json")))
            gn = GE.grid_def(card)["n"]
            split = "HO" if scan in C.EVAL_HOUSES else ("VALSEL" if scan in C.VALSEL_HOUSES else "OTHER")
            for st in info["starts_pano_idx"]:
                tag = f"{key}_S{st}"
                for m in METHODS:
                    a, g = OUT_DIR[m]
                    out = os.path.join(RAW, a, g, tag, "episode.json")
                    src = archived_ok(m, tag, gn) if m in ARCHIVE else None
                    if os.path.exists(out):
                        status = "COMPLETE"
                    elif src:
                        status, out = "REUSED", src
                    else:
                        status = "PENDING"
                    rows.append({"scene_id": key, "start_id": st, "method": m,
                                 "checkpoint_sha": T0_SHA if m.startswith("OURS") else (MAG_SHA if m == "MAG-B" else "none"),
                                 "protocol_id": "EXP7-MP3D150-v1 (T0.12 grid, 150 moves, C_obs)", "model_config_hash": MODEL_HASH[m],
                                 "sim_config_hash": SIM_HASH, "random_seed": SEED[m], "output_path": os.path.relpath(out, R), "status": status,
                                 "scan": scan, "level": int(lv), "split": split, "task2_subset": int((scan, int(lv), int(st)) in t2),
                                 "grid_n": "x".join(str(x) for x in gn)})
    os.makedirs(os.path.join(E7, "manifests"), exist_ok=True)
    mf = os.path.join(E7, "manifests", "mp3d150_manifest.csv")
    with open(mf, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    # job lists: slot A = GPU-heavy V3 planners (OURS first, then OURS-H0), slot B = MAGICIAN then RANDOM; big grids first
    big = lambda r: -int(np.prod([int(x) for x in r["grid_n"].split("x")]))
    jobs = {"A": [r for r in rows if r["method"] == "OURS" and r["status"] == "PENDING"],
            "B": [r for r in rows if r["method"] == "MAG-B" and r["status"] == "PENDING"]}
    jobs["A"] = sorted(jobs["A"], key=big) + sorted([r for r in rows if r["method"] == "OURS-H0" and r["status"] == "PENDING"], key=big)
    jobs["B"] = sorted(jobs["B"], key=big) + [r for r in rows if r["method"] == "RANDOM" and r["status"] == "PENDING"]
    os.makedirs(os.path.join(W7, "jobs"), exist_ok=True)
    for s, lst in jobs.items():
        if "--write-jobs" not in sys.argv:              # never rewrite a job list that running workers are reading
            continue
        with open(os.path.join(W7, "jobs", f"mp3d150_slot{s}.txt"), "w") as f:
            for r in lst:
                f.write(f"{r['method']}:{r['scan']}:{r['level']}:{r['start_id']}\n")
    from collections import Counter
    print("rows", len(rows), Counter((r["method"], r["status"]) for r in rows))
    print("splits", Counter((r["split"]) for r in rows if r["method"] == "OURS"), "task2 subset", sum(r["task2_subset"] for r in rows if r["method"] == "OURS"))
    print("jobs", {s: len(l) for s, l in jobs.items()}, "sim_hash", SIM_HASH, "model_hash", MODEL_HASH)


if __name__ == "__main__":
    main()
