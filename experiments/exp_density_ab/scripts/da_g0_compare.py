"""EXP Density A/B -- G0 closed-loop reproduction (seen2scene env, CPU; FROZEN_PROTOCOL.md §4).
For the 5 historical configs re-run through the da_* wrappers (work_dirs/exp_density_ab/g0_replay), compare with the
historical Task 3 / E2 episodes: executed actions and coverage curves must be identical (bitwise), plus planner scores.
Also recompute the historical G3 AUC / Final of the 5 groups from their episode files (all 10 + 30 episodes).
Output results/g0_closed_loop.json
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(EXP))
W3 = os.path.join(REPO, "work_dirs", "exp5_future_view", "task3_full_depth_p1")
G0 = os.path.join(REPO, "work_dirs", "exp_density_ab", "g0_replay")
HIST = {"B2_S10_S10": "B0", "B2_D100_S10": "B2", "B2_D100_D100": "B1", "E2S_S10_S10": "E2S", "E2D_D100_S10": "E2D"}
REPORTED = {"B0": ((0.322, 0.469), (0.729, 0.884)), "B2": ((0.334, 0.530), (0.730, 0.906)), "B1": ((0.330, 0.472), (0.691, 0.869)),
            "E2S": ((0.347, 0.519), (0.719, 0.880)), "E2D": ((0.364, 0.553), (0.711, 0.880))}


def main():
    out = {"replay": {}, "historical_recompute": {}}
    for cfg, grp in HIST.items():
        r = {}
        tag = "2azQ1b91cZZ_L0_S60"
        pn = os.path.join(G0, "closed_loop", "indoor", cfg, "V2M", tag)
        ph = os.path.join(W3, "closed_loop", "indoor", grp, "V2M", tag)
        if os.path.exists(os.path.join(pn, "episode.json")):
            a, b = json.load(open(os.path.join(pn, "episode.json"))), json.load(open(os.path.join(ph, "episode.json")))
            la = [json.loads(x) for x in open(os.path.join(pn, "planner.jsonl"))]
            lb = [json.loads(x) for x in open(os.path.join(ph, "planner.jsonl"))]
            sk = [k for k in la[0] if "score" in k or k in ("pick", "choice")]
            r["indoor"] = {"tag": tag, "executed_equal": a["executed"] == b["executed"], "coverage_equal": a["coverage"] == b["coverage"],
                           "moves": [a["moves"], b["moves"]], "planner_score_keys": sk,
                           "planner_scores_equal": all(all(x.get(k) == y.get(k) for k in sk) for x, y in zip(la, lb)) and len(la) == len(lb),
                           "final": [a["coverage"][-1], b["coverage"][-1]]}
        tg = "V2M_colosseum_s0_r0"
        en = os.path.join(G0, "closed_loop", "outdoor", cfg, "V2M", "episodes", f"ep_{tg}.json")
        mn = os.path.join(G0, "metrics", "outdoor", cfg, "V2M", f"{tg}.json")
        if os.path.exists(en):
            a = json.load(open(en))
            b = json.load(open(os.path.join(W3, "closed_loop", "outdoor", grp, "V2M", "episodes", f"ep_{tg}.json")))
            r["outdoor"] = {"tag": tg, "executed_keys_equal": a["executed_keys"] == b["executed_keys"], "coverage_official_equal": a["coverage"] == b["coverage"]}
            if os.path.exists(mn):
                ma, mb = json.load(open(mn)), json.load(open(os.path.join(W3, "metrics", "outdoor", grp, "V2M", f"{tg}.json")))
                r["outdoor"].update({"cov_mono_equal": ma["cov_mono"] == mb["cov_mono"], "auc_mono": [ma["auc_mono"], mb["auc_mono"]]})
        out["replay"][cfg] = r
    # historical recompute (scene mean of episode means, as t34 summarize)
    for cfg, grp in HIST.items():
        res = {}
        eps = sorted(glob.glob(os.path.join(W3, "closed_loop", "indoor", grp, "V2M", "*", "episode.json")))
        sc = {}
        for f in eps:
            e = json.load(open(f))
            c = np.asarray(e["coverage"])
            sc.setdefault(os.path.basename(os.path.dirname(f)).split("_")[0], []).append((float(np.trapz(c, dx=1) / e["moves"]), float(c[-1])))
        res["indoor"] = {"n": len(eps), "auc": float(np.mean([np.mean([x[0] for x in v]) for v in sc.values()])),
                         "final": float(np.mean([np.mean([x[1] for x in v]) for v in sc.values()]))}
        ms = sorted(glob.glob(os.path.join(W3, "metrics", "outdoor", grp, "V2M", "V2M_*.json")))
        sc = {}
        for f in ms:
            m = json.load(open(f))
            sc.setdefault(os.path.basename(f).split("_")[1], []).append((float(m["auc_mono"]), float(m["cov_mono_final"])))
        res["outdoor"] = {"n": len(ms), "auc": float(np.mean([np.mean([x[0] for x in v]) for v in sc.values()])),
                          "final": float(np.mean([np.mean([x[1] for x in v]) for v in sc.values()]))}
        rep = REPORTED[grp]
        res["matches_reported_3dp"] = bool(round(res["indoor"]["auc"], 3) == rep[0][0] and round(res["indoor"]["final"], 3) == rep[0][1]
                                           and round(res["outdoor"]["auc"], 3) == rep[1][0] and round(res["outdoor"]["final"], 3) == rep[1][1])
        res["reported"] = rep
        out["historical_recompute"][cfg] = res
    os.makedirs(os.path.join(EXP, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "results", "g0_closed_loop.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
