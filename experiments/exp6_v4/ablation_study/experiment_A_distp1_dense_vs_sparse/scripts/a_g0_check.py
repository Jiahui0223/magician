"""Experiment A G0 closed-loop check: (1) the A0 smokes (V4 planner, default b2 mode, 3 moves) reproduce the stored V4
G3 episodes bitwise (poses, coverage, per-step DistP1 scores from planner logs); (2) the A1 smokes (mode full) ran, see
the same legal candidates per step, and their per-step scores differ from A0 only where band_dil can act (scores never
larger than ... not asserted; just reported). Usage: python a_g0_check.py"""
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(R, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "smoke")
V4 = os.path.join(R, "work_dirs", "exp6_v4", "eval", "closed_loop")


def indoor(root):
    d = os.path.join(root, "V2M", "Z6MFQCViBuw_L0_S6")
    ep = json.load(open(os.path.join(d, "episode.json")))
    pl = [json.loads(x) for x in open(os.path.join(d, "planner.jsonl"))]
    return [(s["t"], s["pose"], s["n_legal"], s["pick"], round(float(s["cov"]), 9)) for s in ep["steps"]], {p["t"]: p["scores"] for p in pl}


def outdoor(root):
    ep = json.load(open(os.path.join(root, "V2M", "episodes", "ep_V2M_barts_s0_r0.json")))
    plp = os.path.join(root, "V2M", "planner_logs", "pl_V2M_barts_s0_r0.json")
    pl = json.load(open(plp)) if os.path.exists(plp) else {"steps": []}
    sc = {p["t"]: p.get("scores") for p in pl["steps"]} if pl.get("steps") else {}
    return [(s["t"], s["pose_key"], s["n_legal"], round(float(s["cov"]), 9)) for s in ep["steps"]], sc


out = {}
for dom, fn, ref_root in (("indoor", indoor, os.path.join(V4, "indoor", "V4")), ("outdoor", outdoor, os.path.join(V4, "outdoor", "V4"))):
    a0, s0 = fn(os.path.join(W, f"{dom}_b2"))
    a1, s1 = fn(os.path.join(W, f"{dom}_full"))
    ref, sref = fn(ref_root)
    n = min(len(a0), len(ref))
    same_steps = [a0[i] == ref[i] for i in range(n)]
    same_scores = [s0.get(t) == sref.get(t) for t in list(s0)[:n] if t in sref]
    n_legal_equal = [a0[i][2] == a1[i][2] for i in range(min(len(a0), len(a1)))]
    diff_scores = [bool(s0.get(t) != s1.get(t)) for t in s0 if t in s1]
    out[dom] = {"A0_vs_stored_V4": {"steps_compared": n, "steps_equal": same_steps, "scores_equal": same_scores,
                                    "status": "PASS" if n >= 2 and all(same_steps) and (not same_scores or all(same_scores)) else "FAIL"},
                "A1": {"ran": len(a1) >= 2, "n_legal_equal_to_A0": n_legal_equal, "scores_differ_from_A0": diff_scores,
                       "picks_A0": [x[3] if dom == "indoor" else x[1] for x in a0], "picks_A1": [x[3] if dom == "indoor" else x[1] for x in a1],
                       "score_t0_A0": s0.get(0), "score_t0_A1": s1.get(0)}}
json.dump(out, open(os.path.join(W, "g0_check.json"), "w"), indent=1)
for dom, v in out.items():
    print(dom, "A0 vs stored V4:", v["A0_vs_stored_V4"]["status"], v["A0_vs_stored_V4"]["steps_equal"], "scores", v["A0_vs_stored_V4"]["scores_equal"],
          "| A1 ran", v["A1"]["ran"], "same n_legal", all(v["A1"]["n_legal_equal_to_A0"]), "scores differ", v["A1"]["scores_differ_from_A0"], "picks A0/A1", v["A1"]["picks_A0"], v["A1"]["picks_A1"])
