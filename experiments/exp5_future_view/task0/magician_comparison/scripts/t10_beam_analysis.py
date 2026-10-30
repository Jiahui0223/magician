"""EXP5 Task 0 / T0.10 -- planner diagnostics of the MAGICIAN groups and the POST-HOC beam analysis (protocol v1.2 §14).
From planner.jsonl (layer-0 gains of the valid legal candidates, pick, diag) and episode.json (real C_obs):
  per step: is the beam pick != the single-step argmax of the layer-0 gains ("beam overrides greedy"), the rank of the
  pick's immediate gain, the real coverage gain of the step; diag means: expanded nodes, Gaussians, history, timing
  split (observe / occupancy / novelty / beam), peak GPU memory. CPU. Output W/magician_comparison/analysis/t10_planner.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mc_common as M  # noqa: E402

OUT = os.path.join(M.MC, "shared_protocol", "closed_loop")


def main():
    res = {"doc": __doc__}
    for g in ("MAG-G", "MAG-B"):
        rows, steps = [], []
        for tag, scan, lv, st in M.episodes():
            pl = os.path.join(OUT, g, tag, "planner.jsonl")
            if not os.path.exists(pl):
                continue
            L = [json.loads(x) for x in open(pl)]
            c = np.asarray(json.load(open(os.path.join(OUT, g, tag, "episode.json")))["coverage"])
            ov, gains_ov, gains_gr = 0, [], []
            for x in L:
                sc = [(-1e30 if s is None else s) for s in x["scores"]]
                greedy = int(np.argmax(sc)) if sc else None
                gain = float(c[x["t"] + 1] - c[x["t"]])
                valid = sorted([s for s in x["scores"] if s is not None], reverse=True)
                rank = valid.index(x["scores"][x["pick"]]) if x["pick"] is not None and x["scores"][x["pick"]] is not None else None
                o = x["pick"] != greedy
                ov += o
                (gains_ov if o else gains_gr).append(gain)
                steps.append({"tag": tag, "t": x["t"], "override": bool(o), "pick_rank": rank, "n_valid": len(valid), "real_gain": gain,
                              **{k: x["diag"][k] for k in ("expanded", "n_gaussians", "n_history")},
                              **{k: x["timing"][k] for k in ("decision_s", "observe_s", "occ_s", "novelty_s", "beam_s")},
                              "peak_mem_gb": x["peak_mem_gb"]})
            rows.append({"tag": tag, "house": scan, "override_share": ov / len(L), "real_gain_override_mean": float(np.mean(gains_ov)) if gains_ov else None,
                         "real_gain_greedy_mean": float(np.mean(gains_gr)) if gains_gr else None})
        S = steps
        res[g] = {"n_episodes": len(rows), "per_episode": rows,
                  "override_share": float(np.mean([s["override"] for s in S])) if S else None,
                  "real_gain_override_steps": float(np.mean([s["real_gain"] for s in S if s["override"]])) if any(s["override"] for s in S) else None,
                  "real_gain_greedy_steps": float(np.mean([s["real_gain"] for s in S if not s["override"]])) if S else None,
                  "pick_rank_hist": {str(k): int(sum(s["pick_rank"] == k for s in S)) for k in range(6)},
                  "max_peak_mem_gb": float(max(s["peak_mem_gb"] for s in S)) if S else None}
        for k in ("expanded", "n_gaussians", "decision_s", "observe_s", "occ_s", "novelty_s", "beam_s"):
            res[g][f"mean_{k}"] = float(np.mean([s[k] for s in S])) if S else None
        print(g, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[g].items() if k != "per_episode"})
    json.dump(res, open(os.path.join(M.MC, "analysis", "t10_planner.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
