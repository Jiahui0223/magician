"""Experiment G -- check of a short closed-loop smoke written by g_episode.sh (G_MOVES < 100; G_WORKDIR = the smoke root).
Checks: replay official == stored; scores finite; K = 1 recorded; with --ref MODEL (an archived Table 1 / Table 3 model): per-step planner
scores and picks, executed keys and the coverage prefix equal the archived episode bitwise.
Usage: python g_smoke_check.py --root EPISODE_ROOT --scene S --start K --moves N [--ref V3-T0-K1D2] --out JSON"""
import argparse
import glob
import json
import math
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--moves", type=int, required=True)
ap.add_argument("--ref", default=None)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tag = f"V2M_{a.scene}_s{a.start}_r0"
ep = json.load(open(os.path.join(a.root, "episodes", f"ep_{tag}.json")))
mt = json.load(open(os.path.join(a.root, "metrics", "V2M", f"{tag}.json")))
steps = [json.loads(x) for x in open(os.path.join(a.root, "planner_logs", f"pl_{tag}.steps.jsonl"))]
pl = json.load(open(os.path.join(a.root, "planner_logs", f"pl_{tag}.json")))
res = {"root": a.root, "scene": a.scene, "start": a.start, "moves": a.moves, "n_steps": len(steps),
       "replay_official_equal_stored": mt.get("official_equal_stored"), "cov": mt["cov_official"],
       "scores_finite": all(all(math.isfinite(x) for x in s["scores"]) for s in steps),
       "K_recorded": sorted({len(k) for s in steps for k in (s["all_scores"].get("distp1_k") or [])}),
       "fallbacks": sum(1 for s in steps if (s.get("choice") or {}).get("fallback")),
       "decision_s_mean": sum(s["decision_s"] for s in steps) / max(1, len(steps)),
       "peak_mem_gb": max((s.get("peak_mem_gb") or 0) for s in steps),
       "ckpt": pl.get("ckpt"), "g": (pl.get("planner") or {}).get("g"), "kd": (pl.get("planner") or {}).get("kd")}
ok = bool(res["replay_official_equal_stored"]) and res["scores_finite"] and res["K_recorded"] == [1] and len(steps) == a.moves
if a.ref:
    r = json.load(open(os.path.join(R, "work_dirs/cvpr_results/table1/episodes", a.ref, f"{a.scene}_s{a.start}.json")))
    rep = json.load(open(os.path.join(R, r["episode_file"])))
    rsteps_p = os.path.join(os.path.dirname(os.path.dirname(os.path.join(R, r["episode_file"]))), "planner_logs", f"pl_{tag}.steps.jsonl")
    rsteps = {s["t"]: s for s in map(json.loads, open(rsteps_p))}
    sc_eq = [s["scores"] == rsteps[s["t"]]["scores"] for s in steps]
    pk_eq = [s["choice"]["pick"] == rsteps[s["t"]]["choice"]["pick"] for s in steps]
    res["ref"] = {"model": a.ref, "scores_bitwise_equal_steps": sum(sc_eq), "picks_equal_steps": sum(pk_eq),
                  "executed_keys_equal": ep["executed_keys"] == rep["executed_keys"][:a.moves + 1],
                  "coverage_prefix_equal": mt["cov_official"] == r["coverage_official"][:a.moves + 1]}
    ok = ok and all(sc_eq) and all(pk_eq) and res["ref"]["executed_keys_equal"] and res["ref"]["coverage_prefix_equal"]
res["pass"] = bool(ok)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "cov"}))
