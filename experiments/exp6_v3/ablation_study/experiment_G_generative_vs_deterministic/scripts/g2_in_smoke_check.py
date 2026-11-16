"""Experiment G2 item 5 -- check of a short indoor smoke (g_in_episode.sh with G_MOVES < 150). Checks: episode written with MOVES moves,
finite scores, planner.err has the G_MODE line with the expected mode / K, planner_info ckpt sha; --ref OURS: per-step scores and picks of
planner.jsonl and the executed poses equal the archived EXP7 OURS raw episode (bitwise) for the first MOVES steps.
Usage: python g2_in_smoke_check.py --od DIR --moves N --mode flow|d0 --k 1|4 [--ref-od DIR] --out JSON"""
import argparse
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--od", required=True)
ap.add_argument("--moves", type=int, required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--k", type=int, required=True)
ap.add_argument("--ref-od", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
ep = json.load(open(os.path.join(a.od, "episode.json")))
pl = [json.loads(x) for x in open(os.path.join(a.od, "planner.jsonl"))]
info = json.load(open(os.path.join(a.od, "planner_info.json")))
err = open(os.path.join(a.od, "planner.err")).read()
gl = [json.loads(x[len("G_MODE "):]) for x in err.splitlines() if x.startswith("G_MODE ")]
res = {"od": a.od, "moves": ep.get("moves"), "n_steps": len(pl), "scores_finite": all(all(math.isfinite(s) for s in p["scores"]) for p in pl),
       "g_line": gl[0] if gl else None, "planner_info": info, "decision_s_mean": sum(p["timing"]["decision_s"] for p in pl) / max(1, len(pl)),
       "peak_mem_gb": max(p.get("peak_mem_gb") or 0 for p in pl), "cov_last": ep["coverage"][-1]}
ok = ep.get("moves") == a.moves and len(pl) == a.moves and res["scores_finite"] and bool(gl) and gl[0]["g_mode"] == a.mode and gl[0]["K"] == a.k
ok = ok and info.get("g_mode") == a.mode and info.get("g_k") == a.k and '"d_policy": "d1"' in err
if a.ref_od:
    ref = [json.loads(x) for x in open(os.path.join(a.ref_od, "planner.jsonl"))][:a.moves]
    rep = json.load(open(os.path.join(a.ref_od, "episode.json")))
    res["ref"] = {"scores_bitwise_equal_steps": sum(p["scores"] == r["scores"] for p, r in zip(pl, ref)), "picks_equal_steps": sum(p["pick"] == r["pick"] for p, r in zip(pl, ref)),
                  "executed_equal": ep["executed"] == rep["executed"][:a.moves + 1], "coverage_prefix_equal": ep["coverage"] == rep["coverage"][:a.moves + 1]}
    ok = ok and res["ref"]["scores_bitwise_equal_steps"] == a.moves and res["ref"]["picks_equal_steps"] == a.moves and res["ref"]["executed_equal"] and res["ref"]["coverage_prefix_equal"]
res["pass"] = bool(ok)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "planner_info"}))
