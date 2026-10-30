"""EXP5 Task 1.2 LOSO decomposition on the pisa S-c views (CPU, evaluation side; protocol §3.6).

(2) main model on the pisa held-out starts (scene seen, starts unseen) vs (3) the LOSO probe on the same views (scene
unseen), strata t = 0, t <= 20, GT coverage < 0.10 and 0.10-0.20; (1) the main model's training fit comes from
W/eval_sets (arclength coordinates, not comparable in absolute value with the ray-distance ES here).
Trigger (pre-registered): (3) M_unknown fair ES worse than (2) by > 25% on all pisa S-c views -> M-SCALE3.
Output W/analysis/t12_loso.json (trigger; written once) and, after M-SCALE3, W/analysis/t12_loso_scale3.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import v2_frozen as VF  # noqa: E402
import t12_analyze as TA  # noqa: E402

import numpy as np  # noqa: E402

STRATA = [("all", lambda t, c: True), ("t=0", lambda t, c: t == 0), ("t<=20", lambda t, c: t <= 20),
          ("cov<0.10", lambda t, c: c < 0.10), ("cov0.10-0.20", lambda t, c: 0.10 <= c < 0.20)]


def pisa_metrics(name):
    r = TA.load_rays(name)
    vm, spread, skill = TA.view_metrics(r)
    meta = []
    for kk in r["keys"]:
        sc, tj, t, j = kk.split("|")
        meta.append((sc, tj, int(t), TA.coverage(sc, tj, int(t)) if sc == "pisa" else None))
    vid = r["view_id"]
    hit, lab = r["hit"].astype(bool), r["lab"]
    out = {}
    for nm, f in STRATA:
        mv = np.array([m[0] == "pisa" and f(m[2], m[3]) for m in meta])
        mr = mv[vid]
        es = [v["es"] for v in vm if mv[v["view"]]]
        esc = [v["es_collapsed"] for v in vm if mv[v["view"]]]
        nh = mr & (lab == TA.CAT_NOHIT)
        out[nm] = {"views": len(es), "rays": int(mr.sum()), "fair_es_vox": float(np.mean(es)) if es else None,
                   "es_over_collapsed": float(np.mean(es) / np.mean(esc)) if es else None,
                   "spread_vox": float(spread[mr].mean()) if mr.any() else None,
                   "skill_vox": float(skill[mr].mean()) if mr.any() else None,
                   "false_wall": float(hit[nh].mean()) if nh.any() else None,
                   "hit_ratio": float(hit[mr].mean() / max(1e-12, (lab[mr] == TA.CAT_HIT).mean())) if mr.any() else None}
    return out


def main():
    res = {"rule": "protocol.md v1.0 §3.6", "main_pisa_Sc": pisa_metrics("main"), "loso_pisa_Sc": pisa_metrics("loso_pisa")}
    if os.path.exists(os.path.join(VF.WORK, "eval_sc", "scale3", "run_0.json")):     # M-SCALE3 (after the trigger)
        res["scale3_pisa_Sc"] = pisa_metrics("scale3")
    a, b = res["main_pisa_Sc"]["all"]["fair_es_vox"], res["loso_pisa_Sc"]["all"]["fair_es_vox"]
    res["degradation_all"] = b / a - 1.0
    res["degradation_by_stratum"] = {k: (res["loso_pisa_Sc"][k]["fair_es_vox"] / res["main_pisa_Sc"][k]["fair_es_vox"] - 1.0)
                                     if res["main_pisa_Sc"][k]["fair_es_vox"] else None for k, _ in STRATA}
    res["scale3_triggered"] = bool(res["degradation_all"] > 0.25)
    es = {}
    for n in ("main", "loso_pisa", "scale3"):
        p = os.path.join(VF.WORK, "eval_sets", f"{n}.json")
        if os.path.exists(p):
            es[n] = json.load(open(p))
    res["eval_sets_arclength"] = es
    os.makedirs(TA.OUT, exist_ok=True)
    # t12_loso.json is the trigger source recorded in frozen_values["scale3"] -> never overwritten; the M-SCALE3
    # extension goes to t12_loso_scale3.json
    outp = os.path.join(TA.OUT, "t12_loso_scale3.json" if "scale3_pisa_Sc" in res else "t12_loso.json")
    assert not (outp.endswith("t12_loso.json") and os.path.exists(outp)), "t12_loso.json exists (frozen trigger source)"
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("degradation_all", "degradation_by_stratum", "scale3_triggered")}, indent=1))
    if "scale3_pisa_Sc" in res:
        res["scale3_vs_loso_by_stratum"] = {k: res["scale3_pisa_Sc"][k]["fair_es_vox"] / res["loso_pisa_Sc"][k]["fair_es_vox"] - 1.0
                                            for k, _ in STRATA}
        json.dump(res, open(outp, "w"), indent=1, default=float)
        print("scale3 vs loso", json.dumps(res["scale3_vs_loso_by_stratum"]))
        for k, _ in STRATA:
            x = res["scale3_pisa_Sc"][k]
            print(f"scale3 {k:14s} ES {x['fair_es_vox']:.3f} fw {x['false_wall']:.4f} hr {x['hit_ratio']:.3f} spread {x['spread_vox']:.2f} skill {x['skill_vox']:.2f}")
    for k, _ in STRATA:
        m, l_ = res["main_pisa_Sc"][k], res["loso_pisa_Sc"][k]
        print(f"{k:14s} views {m['views']:4d}  ES main {m['fair_es_vox']:.3f} loso {l_['fair_es_vox']:.3f}  "
              f"fw {m['false_wall']:.4f}/{l_['false_wall']:.4f}  hr {m['hit_ratio']:.3f}/{l_['hit_ratio']:.3f}  "
              f"spread {m['spread_vox']:.2f}/{l_['spread_vox']:.2f}  skill {m['skill_vox']:.2f}/{l_['skill_vox']:.2f}")


if __name__ == "__main__":
    main()
