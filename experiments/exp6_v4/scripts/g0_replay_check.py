"""G0-11b: compare the 2-move replay smokes (g0_replay.sh) with the stored formal episodes of the same planners
(T0 = V3 final.pt under exp6_v3/eval, V2M = Task 2 V2-Mix sparse planner): the chosen poses of the common steps must be
identical (same candidates, noise seeds, scores -> same arg-max). Usage: python g0_replay_check.py"""
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W = os.path.join(R, "work_dirs", "exp6_v4", "eval", "g0_replay")
E6 = os.path.join(R, "work_dirs", "exp6_v3", "eval", "closed_loop")
T2 = os.path.join(R, "work_dirs", "exp5_future_view", "task2", "eval")
CASES = {
    "indoor_T0": (f"{W}/indoor_T0/V2M/Z6MFQCViBuw_L0_S6/episode.json", f"{E6}/indoor/T0/V2M/Z6MFQCViBuw_L0_S6/episode.json"),
    "indoor_V2M": (f"{W}/indoor_V2M/V2M/Z6MFQCViBuw_L0_S6/episode.json", f"{T2}/indoor/closed_loop/V2M/Z6MFQCViBuw_L0_S6/episode.json"),
    "outdoor_T0": (f"{W}/outdoor_T0/V2M/episodes/ep_V2M_barts_s0_r0.json", f"{E6}/outdoor/T0/V2M/episodes/ep_V2M_barts_s0_r0.json"),
    "outdoor_V2M": (f"{W}/outdoor_V2M/V2M/episodes/ep_V2M_barts_s0_r0.json", f"{T2}/outdoor/closed_loop/V2M/episodes/ep_V2M_barts_s0_r0.json"),
}


def poses(ep):
    st = ep["steps"]
    return [(s["t"], s.get("pose", s.get("pose_key")), s.get("pick"), round(float(s["cov"]), 9)) for s in st]


out = {}
for name, (smoke, ref) in CASES.items():
    if not os.path.exists(smoke):
        out[name] = {"status": "MISSING", "path": smoke}
        continue
    a, b = poses(json.load(open(smoke))), poses(json.load(open(ref)))
    n = min(len(a), len(b))
    same = [x[1] == y[1] for x, y in zip(a[:n], b[:n])]
    cov_same = [abs(x[3] - y[3]) < 1e-9 for x, y in zip(a[:n], b[:n])]
    out[name] = {"status": "PASS" if n >= 2 and all(same) and all(cov_same) else "FAIL", "steps_compared": n,
                 "smoke": a[:n], "reference": b[:n], "pose_equal": same, "coverage_equal": cov_same}
json.dump(out, open(os.path.join(W, "replay_check.json"), "w"), indent=1)
for k, v in out.items():
    print(k, v["status"], v.get("steps_compared"), v.get("pose_equal"), v.get("coverage_equal"))
