"""Camera Pose Phase 1 -- Stage B, part 2: fixed-state candidate analysis of C0 vs C1 (seen2scene env, CPU).

Per development state (MP3D VALSEL 90, Macarons++ Dev-3 275) and model (B2, E2-D100, V4, V5): the historical per-candidate
DistP1 lists (K = 4 samples; same states / candidates / seeds as the V3-V5.2-M fixed-state evaluations: B2 / E2-D100 /
V4 from the exp6_v4 merged rows, V5 from exp6_v5 rows) + the new FrontierPrior (fixed_prior.py) ->
Score_C1(lam) = D / max D + lam F / max F for lam in {0, 0.1, 0.3} (lam 0 = the historical C0 score). The pick uses the
planner's own rule (ab_load.pick_of: indoor prefer-unvisited nv_choice, outdoor choose). Metrics vs the oracle one-step
gain (GT, evaluation only): selected gain (pp), oracle best gain (must not depend on lam: same candidate set), Top-1,
Spearman, share of states whose pick changes, candidate counts, prior timing; paired scene-level CIs C1 - C0; the
pre-registered lam* (protocol.md §2). Output reports/fixed_state.{json,md}. Usage: python analyze_fixed.py --prior DIR"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
sys.path.insert(0, R)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v4", "scripts"))
import analyze_g12_v4 as A4  # noqa: E402
import frontier as FR  # noqa: E402

EA = A4.EA
W5 = os.path.join(R, "work_dirs", "exp6_v5", "eval_fixed")
BASE = ("B2", "E2D", "V4", "V5")
LAMS = (0.0, 0.1, 0.3)
DUMMY = {**{f"sae_{g}": 0.0 for g in ("all", "unknown", "new", "known", "unknown_new")}, **{f"n_{g}": 0 for g in ("all", "unknown", "new", "known", "unknown_new")},
         "hit_agree": 0, "n_rays": 1, "rel_std_sum": 0.0, "n_allhit": 0, "hit_dis": 0, "rho_mean": None}


def key(m, lam):
    return f"{m}@{lam:g}"


def load_prior(d, sset):
    out = {}
    for f in glob.glob(os.path.join(d, sset, "*", "*.jsonl")):
        for x in open(f):
            r = json.loads(x)
            out[r["sid"]] = r
    return out


def load(sset, prior):
    base = {r["sid"]: r for r in A4.load(sset, "s_final_v4", "s_final_v31")}
    v5 = A4.read_rows(os.path.join(W5, f"{sset}_s_final_v5"))
    rows = []
    for sid, r in base.items():
        if sid not in prior or sid not in v5:
            continue
        assert v5[sid]["scores"]["B2"] == r["scores"]["B2"], ("B2 replay differs", sid)
        P = prior[sid]
        sc = {"B2": r["scores"]["B2"], "E2D": r["scores"]["E2D"], "V4": r["scores"]["V4"], "V5": v5[sid]["scores"]["V5"]}
        F = P["F"]
        assert all(len(s) == len(F) for s in sc.values()), ("candidate count mismatch", sid, len(F), {m: len(s) for m, s in sc.items()})
        r = dict(r)
        r["scores"], r["pred"] = {}, {}
        for m in BASE:
            D = [float(np.mean(k)) for k in sc[m]]
            for lam in LAMS:
                r["scores"][key(m, lam)] = [[s] for s in FR.combine(D, F, lam)]
                r["pred"][key(m, lam)] = DUMMY
        r["_prior"] = P
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior", required=True)
    ap.add_argument("--out", default=os.path.join(E, "reports"))
    a = ap.parse_args()
    models = tuple(key(m, l) for m in BASE for l in LAMS)
    pairs = tuple((key(m, l), key(m, 0.0)) for m in BASE for l in LAMS[1:])
    A4.MODELS = models
    EA.MODELS, EA.PAIRS = models, pairs
    EA.LAB = {k: k for k in models}
    EA.COL = {k: "#495057" for k in models}
    EA.OUT = EA.FIG = a.out
    EA.ST.setup()
    res = {"lams": LAMS, "models": BASE, "sets": {}}
    for sset in ("dev3", "valsel"):
        prior = load_prior(a.prior, sset)
        rows = load(sset, prior)
        recs = A4.rows_to_recs(sset, rows)
        S = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        x = {"n_states": len(recs), "summary": S}
        nl = [r["_prior"]["n_legal"] for r in rows]
        x["candidates"] = {"mean": float(np.mean(nl)), "min": int(min(nl)), "max": int(max(nl))}
        x["prior_time_s"] = {"view_mean": float(np.mean([r["_prior"]["view_s"] for r in rows])), "frontier_mean": float(np.mean([r["_prior"]["frontier_s"] for r in rows]))}
        x["F_all_zero_share"] = float(np.mean([max(r["_prior"]["F"]) == 0 for r in rows]))
        x["F_constant_share"] = float(np.mean([len(set(r["_prior"]["F"])) == 1 for r in rows]))
        ch = {}
        for m in BASE:
            o0 = [rc[f"{key(m, 0.0)}_oracle"] for rc in recs]
            for lam in LAMS[1:]:
                k = key(m, lam)
                assert [rc[f"{k}_oracle"] for rc in recs] == o0, ("oracle differs across lam", m, lam)
                ch[k] = float(np.mean([rc[f"{k}_pick"] != rc[f"{key(m, 0.0)}_pick"] for rc in recs]))
        x["pick_changed_share"] = ch
        x["oracle_identical_across_lam"] = True
        res["sets"][sset] = x
    # pre-registered lam*: mean over the two domains of the mean selected gain; ties -> smaller lam
    star = {}
    for m in BASE:
        val = {lam: float(np.mean([res["sets"][s]["summary"]["all"]["means"][f"{key(m, lam)}_gain"]["mean"] for s in ("dev3", "valsel")])) for lam in LAMS}
        best = max(LAMS, key=lambda l: (round(val[l], 9), -l))
        star[m] = {"domain_mean_gain": val, "lam_star": best}
    res["lam_star"] = star
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(os.path.join(a.out, "fixed_state.json"), "w"), indent=1, default=float)
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    L = ["# Stage B —— 固定状态候选分析（C0 = λ 0，C1 = λ 0.1 / 0.3；同一状态、同一候选、同一历史 DistP1；GT 只用于评估）", ""]
    for sset, nm in (("dev3", "Macarons++ Dev-3"), ("valsel", "MP3D VALSEL")):
        x = res["sets"][sset]
        L += [f"## {nm}（{x['n_states']} 态；候选数 均值 {x['candidates']['mean']:.2f}，范围 {x['candidates']['min']}–{x['candidates']['max']}；"
              f"先验 F 全 0 的状态 {x['F_all_zero_share']:.3f}，F 全相同 {x['F_constant_share']:.3f}；视图 {x['prior_time_s']['view_mean']:.3f} s + 先验 {x['prior_time_s']['frontier_mean']:.3f} s / 态）", "",
              "| 模型 | λ | Selected gain（pp） | Oracle best（pp） | Top-1 | Spearman | 选择改变比例 | Δgain vs λ0 [95% CI] | ΔTop-1 vs λ0 |", "|---|---|---|---|---|---|---|---|---|"]
        S = x["summary"]["all"]
        for m in BASE:
            for lam in LAMS:
                k = key(m, lam)
                mm = S["means"]
                pg = S["paired"].get(f"{k}-{key(m, 0.0)}|gain")
                pt = S["paired"].get(f"{k}-{key(m, 0.0)}|top1")
                L.append(f"| {m} | {lam:g} | {f3(mm[f'{k}_gain']['mean'])} | {f3(mm[f'{k}_oracle']['mean']) if f'{k}_oracle' in mm else '—'} | {f3(mm[f'{k}_top1']['mean'])} | {f3(mm[f'{k}_rho']['mean'])} | "
                         f"{x['pick_changed_share'].get(k, 0.0):.3f} | " + (f"{pg['mean']:+.3f} [{pg['ci95'][0]:+.3f}, {pg['ci95'][1]:+.3f}]" if pg else "—") + " | " + (f"{pt['mean']:+.3f}" if pt else "—") + " |")
        L.append("")
    L += ["## 预注册 λ*（两域 mean selected gain 的平均；并列取小）", "", "| 模型 | λ 0 | λ 0.1 | λ 0.3 | λ* |", "|---|---|---|---|---|"]
    for m, v in star.items():
        L.append(f"| {m} | " + " | ".join(f"{v['domain_mean_gain'][l]:.4f}" for l in LAMS) + f" | **{v['lam_star']:g}** |")
    open(os.path.join(a.out, "fixed_state.md"), "w").write("\n".join(L) + "\n")
    print(json.dumps(star, indent=1))


if __name__ == "__main__":
    main()
