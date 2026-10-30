"""EXP5 Ablation -- automatic case selection for the A1 / A2 figures (protocol §5; rules fixed before the results).

Per dataset, model V2M (the same states are also drawn for V2O):
  success          V2M pick tie-aware Top-1 and GTD Top-1; largest oracle max gain
  prediction_fail  V2M normalised regret >= 0.5 and GTD Top-1; largest normalised regret (ties: larger regret)
  scoring_fail     GTD normalised regret >= 0.5; largest normalised regret (ties: larger regret)
  stochastic       V2M: >= 3 distinct K = 1 picks or a pivotal sample; largest dominance of the picked candidate
A missing category is reported as such. Output A2_stochastic_future_views/results/cases.json
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ABL = os.path.dirname(HERE)
A_W = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(ABL))), "work_dirs", "exp5_future_view", "ablation_study")


def read(p):
    return list(csv.DictReader(open(p)))


def f(x):
    return float(x) if x not in ("", None, "None") else None


def main():
    out = {}
    for dataset in ("indoor", "outdoor"):
        a1 = read(os.path.join(ABL, "A1_prediction_scoring", "results", f"a1_states_{dataset}.csv"))
        a2 = read(os.path.join(ABL, "A2_stochastic_future_views", "results", f"a2_states_{dataset}.csv"))
        by = {}
        for r in a1:
            by.setdefault(r["sid"], {})[r["scorer"]] = r
        k2 = {r["sid"]: r for r in a2 if r["model"] == "V2M"}
        cases = {}
        succ = [(f(v["GTD"]["max_gain"]), s) for s, v in by.items() if f(v["V2M"]["top1"]) == 1 and f(v["GTD"]["top1"]) == 1]
        pf = [((f(v["V2M"]["nregret"]), f(v["V2M"]["regret"])), s) for s, v in by.items()
              if f(v["V2M"]["nregret"]) is not None and f(v["V2M"]["nregret"]) >= 0.5 and f(v["GTD"]["top1"]) == 1]
        sf = [((f(v["GTD"]["nregret"]), f(v["GTD"]["regret"])), s) for s, v in by.items()
              if f(v["GTD"]["nregret"]) is not None and f(v["GTD"]["nregret"]) >= 0.5]
        st = [(f(k2[s]["pick_dominance"]) or 0.0, s) for s in k2
              if f(k2[s]["k1_distinct"]) >= 3 or f(k2[s]["loo_pivotal"]) == 1]
        # DEVIATIONS #9: the pre-registered 'stochastic' rule can pick a degenerate state (all scores ~0); an additional case
        # restricted to states whose picked V2M score is at least the median picked score of the dataset
        import numpy as _np
        pickscore = {}
        for sid in by:
            v = json.load(open(os.path.join(A_W, "states", dataset, sid, "v2.json")))
            pickscore[sid] = float(v["models"]["V2M"]["distp1"][int(by[sid]["V2M"]["pick"])])
        med = float(_np.median(list(pickscore.values())))
        st2 = [(f(k2[s]["pick_dominance"]) or 0.0, s) for s in k2
               if (f(k2[s]["k1_distinct"]) >= 3 or f(k2[s]["loo_pivotal"]) == 1) and pickscore[s] >= med]
        for nm, cand in (("success", succ), ("prediction_fail", pf), ("scoring_fail", sf), ("stochastic", st),
                         ("stochastic_nondegenerate", st2)):
            if cand:
                key, sid = max(cand, key=lambda x: (x[0], x[1]))
                cases[nm] = {"sid": sid, "key": key, "n_eligible": len(cand), "t": int(by[sid]["V2M"]["t"]),
                             "scene": by[sid]["V2M"]["scene"]}
            else:
                cases[nm] = {"sid": None, "n_eligible": 0}
        out[dataset] = cases
        print(dataset, {k: (v["sid"], v["n_eligible"]) for k, v in cases.items()})
    json.dump(out, open(os.path.join(ABL, "A2_stochastic_future_views", "results", "cases.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
