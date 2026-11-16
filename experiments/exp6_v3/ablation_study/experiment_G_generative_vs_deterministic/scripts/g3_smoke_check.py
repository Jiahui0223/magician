"""Experiment G3 -- check of a short closed-loop smoke (outdoor: g3_episode.sh, indoor: g3_in_episode.sh with G_MOVES < full).
Checks: MOVES steps, finite scores, G3_MODE line with the expected mode, K recorded (outdoor planner log K / indoor info); with a reference
episode: per-step scores / picks, executed poses and coverage prefix bitwise equal; outdoor also the official replay equality.
Usage: python g3_smoke_check.py --domain out|in --dir DIR --moves N --mode off|raw|depth [--ref-dir DIR] --out JSON
  out: --dir = episode root (closed_loop/<ARM>/<scene>_s<k>), --ref-dir = the archived Table 1 root .../V3-T0-D1/V2M
  in:  --dir = raw/<ARM>/V2M/<tag>, --ref-dir = the archived EXP7 OURS raw dir"""
import argparse
import glob
import json
import math
import os

ap = argparse.ArgumentParser()
ap.add_argument("--domain", required=True, choices=("out", "in"))
ap.add_argument("--dir", required=True)
ap.add_argument("--moves", type=int, required=True)
ap.add_argument("--mode", required=True)
ap.add_argument("--ref-dir", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
K_exp = 4 if a.mode == "off" else 1
res = {"dir": a.dir, "domain": a.domain, "mode": a.mode, "moves": a.moves}
if a.domain == "out":
    tag = os.path.basename(glob.glob(os.path.join(a.dir, "episodes", "ep_*.json"))[0])[3:-5]
    ep = json.load(open(os.path.join(a.dir, "episodes", f"ep_{tag}.json")))
    mt = json.load(open(os.path.join(a.dir, "metrics", "V2M", f"{tag}.json")))
    steps = [json.loads(x) for x in open(os.path.join(a.dir, "planner_logs", f"pl_{tag}.steps.jsonl"))]
    err = open(os.path.join(a.dir, "planner_logs", f"pl_{tag}.stderr")).read() if os.path.exists(os.path.join(a.dir, "planner_logs", f"pl_{tag}.stderr")) else ""
    gl = [json.loads(x[len("G3_MODE "):]) for x in err.splitlines() if x.startswith("G3_MODE ")]
    res.update({"n_steps": len(steps), "replay_official_equal_stored": mt.get("official_equal_stored"),
                "K_recorded": sorted({len(k) for s in steps for k in (s["all_scores"].get("distp1_k") or [])}),
                "g3_line": gl[0] if gl else None, "decision_s_mean": sum(s["decision_s"] for s in steps) / max(1, len(steps)), "cov_last": mt["cov_official"][-1],
                "scores_finite": all(all(math.isfinite(x) for x in s["scores"]) for s in steps)})
    ok = bool(mt.get("official_equal_stored")) and len(steps) == a.moves and res["scores_finite"] and res["K_recorded"] == [K_exp] and bool(gl) and gl[0]["g3_mode"] == a.mode
    if a.ref_dir:
        rs = {s["t"]: s for s in map(json.loads, open(os.path.join(a.ref_dir, "planner_logs", f"pl_{tag}.steps.jsonl")))}
        rep = json.load(open(os.path.join(a.ref_dir, "episodes", f"ep_{tag}.json")))
        res["ref"] = {"scores_bitwise_equal_steps": sum(s["scores"] == rs[s["t"]]["scores"] for s in steps),
                      "picks_equal_steps": sum(s["choice"]["pick"] == rs[s["t"]]["choice"]["pick"] for s in steps),
                      "executed_keys_equal": ep["executed_keys"] == rep["executed_keys"][:a.moves + 1], "coverage_prefix_equal": ep["coverage"] == rep["coverage"][:a.moves + 1]}
        ok = ok and res["ref"]["scores_bitwise_equal_steps"] == a.moves and res["ref"]["picks_equal_steps"] == a.moves and res["ref"]["executed_keys_equal"] and res["ref"]["coverage_prefix_equal"]
else:
    ep = json.load(open(os.path.join(a.dir, "episode.json")))
    pl = [json.loads(x) for x in open(os.path.join(a.dir, "planner.jsonl"))]
    err = open(os.path.join(a.dir, "planner.err")).read()
    gl = [json.loads(x[len("G3_MODE "):]) for x in err.splitlines() if x.startswith("G3_MODE ")]
    res.update({"n_steps": len(pl), "g3_line": gl[0] if gl else None, "scores_finite": all(all(math.isfinite(s) for s in p["scores"]) for p in pl),
                "decision_s_mean": sum(p["timing"]["decision_s"] for p in pl) / max(1, len(pl)), "cov_last": ep["coverage"][-1]})
    ok = ep.get("moves") == a.moves and len(pl) == a.moves and res["scores_finite"] and bool(gl) and gl[0]["g3_mode"] == a.mode and gl[0]["K"] == K_exp
    if a.ref_dir:
        ref = [json.loads(x) for x in open(os.path.join(a.ref_dir, "planner.jsonl"))][:a.moves]
        rep = json.load(open(os.path.join(a.ref_dir, "episode.json")))
        res["ref"] = {"scores_bitwise_equal_steps": sum(p["scores"] == r["scores"] for p, r in zip(pl, ref)), "picks_equal_steps": sum(p["pick"] == r["pick"] for p, r in zip(pl, ref)),
                      "executed_equal": ep["executed"] == rep["executed"][:a.moves + 1], "coverage_prefix_equal": ep["coverage"] == rep["coverage"][:a.moves + 1]}
        ok = ok and res["ref"]["scores_bitwise_equal_steps"] == a.moves and res["ref"]["picks_equal_steps"] == a.moves and res["ref"]["executed_equal"] and res["ref"]["coverage_prefix_equal"]
res["pass"] = bool(ok)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res))
