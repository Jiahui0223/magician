"""Run NBV episodes (exp1-task7 onwards).

Examples (one GPU per shard):
  # exp1-task7 calibration (Random vs Oracle on dev50)
  CUDA_VISIBLE_DEVICES=0 python -m seen2scene.tools.nbv_run --split dev --subset dev50 \
      --planners random oracle --N 32 64 --T 20 --shard 0 --num-shards 4 \
      --out work_dirs/exp1_infra/task7_calibration
  # exp2 planner baselines (frozen N=64, T=10; S0-F world model with S=10 for s1/s2)
  CUDA_VISIBLE_DEVICES=0 python -m seen2scene.tools.nbv_run --split dev --planners s1 \
      --shard 0 --num-shards 4 --out work_dirs/exp2_baseline/task5_s1/dev100
Each episode writes <out>/<planner>_N<N>/<scene>.json with the per-step log. Existing files
are skipped, so the command can be re-run to resume.
"""
import argparse
import csv
import glob
import os

from seen2scene.active.episode import MODEL_PLANNERS, Episode, EpisodeConfig, make_planner
from seen2scene.active.world_model import WorldModel, enable_deterministic

SPLITS = "assets/splits/front3d_heldout"
GT = "work_dirs/exp1_infra/task2_gt_eval/gt"
KEYS = ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--subset", default=None, help="scene list file stem, e.g. dev50")
    ap.add_argument("--scenes", nargs="+", default=None, help="only these scenes (name prefixes), applied before sharding")
    ap.add_argument("--planners", nargs="+", default=["random", "oracle"],
                    help="random, oracle, frontier, s1, s2, vhd (s1/s2/vhd load the S0-F world model once per process)")
    ap.add_argument("--N", nargs="+", type=int, default=[64])
    ap.add_argument("--T", type=int, default=10, help="frozen in exp1-task7: T=10")
    ap.add_argument("--steps", type=int, default=10, help="world-model sampling steps (exp2-task4: S=10)")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    # exp2: every planner runs with the same deterministic kernels (also used by the fusion), set before any
    # CUDA call, so that all planners share the same simulator numerics as the world model.
    enable_deterministic()

    rows = list(csv.DictReader(open(os.path.join(SPLITS, f"{args.split}_patches.csv"))))
    if args.subset:
        keep = set(open(os.path.join(SPLITS, f"{args.subset}.txt")).read().split())
        rows = [r for r in rows if r["scene_name"] in keep]
    if args.scenes:
        rows = [r for r in rows if any(r["scene_name"].startswith(p) for p in args.scenes)]
    rows = rows[args.shard::args.num_shards]
    wm = WorldModel(split=args.split, steps=args.steps) if set(args.planners) & set(MODEL_PLANNERS) else None
    planners = {p: make_planner(p, wm) for p in args.planners}
    for r in rows:
        s = r["scene_name"]
        bbox = [float(r[k]) for k in KEYS]
        gt_dir = glob.glob(os.path.join(GT, args.split, f"{s}_*"))[0]
        for n in args.N:
            for p in args.planners:
                out = os.path.join(args.out, f"{p}_N{n}", f"{s}.json")
                if os.path.exists(out):
                    continue
                ep = Episode(s, bbox, gt_dir, EpisodeConfig(T=args.T, N=n), episode_id=args.episode)
                log = ep.run(planners[p], out)
                ep.close()
                fb = sum(1 for x in log if x.get("planner_fallback"))
                zf = sum(1 for x in log if x.get("vhd_zero_fallback"))
                print(f"{s[:8]} {p} N={n}: coverage {log[0]['surface_coverage']:.3f} -> {log[-1]['surface_coverage']:.3f}"
                      f"{f' (fallback {fb})' if fb else ''}{f' (vhd_zero_fallback {zf})' if zf else ''}", flush=True)


if __name__ == "__main__":
    main()
