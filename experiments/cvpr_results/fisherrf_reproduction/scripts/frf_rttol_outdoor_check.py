"""Outdoor regression check after adding --rt-tol (default 1e-3) to fisherrf_full_planner.py (2026-11-14, F1c-in I9).
Compares a fresh short f1c4 outdoor run (default arguments, i.e. the outdoor path) with the stored full-run episode of the same
scene / start (full/outdoor/closed_loop_f1c4, not modified): executed poses, picks, fallbacks, replan reasons / best goals / candidate
counts, coverage per step, and float diagnostics (n_gauss, top EIG). FisherRF is not bitwise reproducible (R1: atomicAdd in the
rasterizer backward, random keyframe selection), so discrete decisions must match and float diagnostics are reported.
Usage: python frf_rttol_outdoor_check.py --new ROOT --ref ROOT --tag barts_s0 --steps N"""
import argparse
import glob
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("--new", required=True)
ap.add_argument("--ref", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--steps", type=int, required=True)
a = ap.parse_args()


def load(root):
    rows = [json.loads(l) for l in open(glob.glob(os.path.join(root, "planner_logs", "*.steps.jsonl"))[0])]
    ep = json.load(open(glob.glob(os.path.join(root, "episodes", "*.json"))[0]))
    return rows, ep


rn, en = load(a.new)
rr, er = load(a.ref)
n = a.steps
disc = lambda r: {"pose": r["pose"], "pick": r["pick"], "pick_pose": r["pick_pose"], "fallback": r["fallback"], "n_legal": r["n_legal"],  # noqa: E731
                  "queue_len": r["queue_len"], "skipped": r["skipped_no_valid_depth"],
                  "replans": [(x["reason"], x["info"].get("best_goal"), x["info"]["cand"].get("n_unique"), x["info"]["cand"].get("reach_filter")) for x in r["replans"]]}
same = [disc(x) == disc(y) for x, y in zip(rn[:n], rr[:n])]
res = {"tag": a.tag, "steps": n, "decisions_equal_per_step": same, "all_decisions_equal": all(same) and len(rn) >= n,
       "executed_equal": en["executed_keys"][:n + 1] == er["executed_keys"][:n + 1],
       "coverage_new": en["coverage"][:n + 1], "coverage_ref": er["coverage"][:n + 1],
       "coverage_equal": en["coverage"][:n + 1] == er["coverage"][:n + 1],
       "n_gauss_new_ref": [(x["map"].get("n_gauss"), y["map"].get("n_gauss")) for x, y in zip(rn[:n], rr[:n])],
       "top_eig_new_ref": [([z["info"].get("top_eig", [None])[0] for z in x["replans"]], [z["info"].get("top_eig", [None])[0] for z in y["replans"]]) for x, y in zip(rn[:n], rr[:n])],
       "rt_max_err_new": max(x["rt_max_err"] for x in rn[:n]),
       "hello_new": {k: en["planner_hello"][-1]["info"].get(k) for k in ("frontier_mode", "reach_filter", "init_pose")}}
json.dump(res, open(os.path.join(a.new, "rttol_regression.json"), "w"), indent=1, default=float)
print(json.dumps({k: res[k] for k in ("all_decisions_equal", "executed_equal", "coverage_equal", "rt_max_err_new", "hello_new")}))
raise SystemExit(0 if res["all_decisions_equal"] and res["executed_equal"] and res["coverage_equal"] else 1)
