"""Experiment XA -- regression of the XA evaluation chain with the V3-T1 checkpoint (CPU): the outdoor D1 episode barts s0 must equal the CVPR Table 1 record V3-D1-H3-NORISK
(executed poses and official coverage, all 101 values) and the indoor H0 episode Z6MFQCViBuw_L0 S6 must equal the archived exp6_v3 T1 episode (poses and C_obs, 151 values).
Also lists the short closed-loop smokes of the six variants. -> results/regression.json"""
import glob
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
X = os.path.join(R, "experiments", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
W = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_XA_multiscale_ray_scene")
out = {}
e = json.load(open(f"{W}/eval/REG_T1/outdoor/V2M/episodes/ep_V2M_barts_s0_r0.json"))
m = json.load(open(f"{W}/eval/REG_T1/metrics/V2M/V2M_barts_s0_r0.json"))
t = json.load(open(f"{R}/work_dirs/cvpr_results/table1/episodes/V3-D1-H3-NORISK/barts_s0.json"))
out["outdoor_barts_s0_D1"] = {"moves": e["moves"], "values": len(e["coverage"]), "executed_keys_equal": e["executed_keys"] == t["executed_keys"], "coverage_bitwise_equal": e["coverage"] == t["coverage_official"],
                              "max_abs_coverage_diff": max(abs(a - b) for a, b in zip(e["coverage"], t["coverage_official"])), "replay_official_equal_stored": m["official_equal_stored"],
                              "ckpt_sha_equal": (e["planner_hello"][0] or {}).get("ckpt_sha256") == t["ckpt_sha256"], "final": e["coverage"][-1]}
p = f"{W}/eval/REG_T1/indoor/V2M/Z6MFQCViBuw_L0_S6/episode.json"
if os.path.exists(p):
    a = json.load(open(p))
    b = json.load(open(f"{R}/work_dirs/exp6_v3/eval/closed_loop/indoor/T1/V2M/Z6MFQCViBuw_L0_S6/episode.json"))
    ka, kb = a["executed"], b["executed"]
    out["indoor_Z6MFQCViBuw_L0_S6_H0"] = {"moves": a["moves"], "values": len(a["coverage"]), "coverage_bitwise_equal": a["coverage"] == b["coverage"], "executed_equal": ka == kb and len(ka) > 0, "picks_equal": [x["pick"] for x in a["steps"]] == [x["pick"] for x in b["steps"]],
                                          "max_abs_coverage_diff": max(abs(x - y) for x, y in zip(a["coverage"], b["coverage"])), "final": a["coverage"][-1]}
sm = {}
for d in sorted(glob.glob(f"{W}/eval/SMOKE_*")):
    r = {}
    for f in glob.glob(f"{d}/outdoor/V2M/episodes/*.json"):
        x = json.load(open(f))
        mm = glob.glob(f"{d}/metrics/V2M/*.json")
        r["outdoor"] = {"moves": x["moves"], "values": len(x["coverage"]), "final": x["coverage"][-1], "replay_ok": bool(mm and json.load(open(mm[0]))["official_equal_stored"])}
    for f in glob.glob(f"{d}/indoor/V2M/*/episode.json"):
        x = json.load(open(f))
        r["indoor"] = {"moves": x["moves"], "values": len(x["coverage"]), "final": x["coverage"][-1]}
    q = f"{d}/logs/queue.log"
    r["queue"] = [l.strip() for l in open(q)] if os.path.exists(q) else []
    sm[os.path.basename(d)] = r
out["smoke_closed_loops"] = sm
out["PASS"] = bool(out["outdoor_barts_s0_D1"]["executed_keys_equal"] and out["outdoor_barts_s0_D1"]["coverage_bitwise_equal"] and out["outdoor_barts_s0_D1"]["replay_official_equal_stored"]
                   and out.get("indoor_Z6MFQCViBuw_L0_S6_H0", {}).get("coverage_bitwise_equal") and out["indoor_Z6MFQCViBuw_L0_S6_H0"]["executed_equal"]
                   and len(sm) == 6 and all("outdoor" in r and "indoor" in r and r["outdoor"]["replay_ok"] for r in sm.values()))
json.dump(out, open(f"{X}/results/regression.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
