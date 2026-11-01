"""EXP6 V4 / V3.1 -- G1 / G2 analysis on the development fixed states (seen2scene env, CPU): exp6_v3/shared/analyze_g12
readers with the models B2 / E2D (E2 rows), T0 / T1 (V3 rows, exp6_v3/eval_fixed) and V31 / V4 (exp6_v4/eval_fixed).
Main fixed-state metrics (user rule 2026-10-31): true one-step coverage gain of the pick (pp, higher is better) and
Top-1; regret only in the appendix. G1: MAE on unknown / new / known rays, hit agreement, K = 4 spread.
Output exp6_v4/reports/g12_v4_<tag>.{json,md} + figures/g12_v4_<tag>.png. Usage: python analyze_g12_v4.py [--tag s_final]"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
sys.path.insert(0, os.path.join(REPO, "experiments", "exp6_v3", "shared"))
import analyze_g12 as A12  # noqa: E402

EA, EP = A12.EA, A12.EP
E4 = os.path.join(REPO, "experiments", "exp6_v4")
W3 = os.path.join(REPO, "work_dirs", "exp6_v3", "eval_fixed")
W4 = os.path.join(REPO, "work_dirs", "exp6_v4", "eval_fixed")
MODELS = ("B2", "E2D", "T0", "T1", "V31", "V4")
LAB = dict(A12.LAB, V31="V3.1 Outdoor-only", V4="V4 Outdoor-only (Ray X-Attn)")
COL = dict(A12.COL, V31="#0b7285", V4="#d6336c")
PAIRS = (("V4", "V31"), ("V4", "B2"), ("V31", "B2"), ("V4", "T1"), ("V31", "T0"), ("V4", "E2D"), ("V31", "E2D"), ("V31", "T1"))
NAME = EA.NAME


def read_rows(root):
    out = {}
    for f in sorted(glob.glob(os.path.join(root, "*", "*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            out[r["sid"]] = r
    return out


def load(sset, tag):
    e2 = read_rows(os.path.join(EP.W, "fixed", f"{sset}_s5000"))
    v3 = read_rows(os.path.join(W3, f"{sset}_s_final"))
    v4 = read_rows(os.path.join(W4, f"{sset}_{tag}"))
    rows = []
    for sid, r in v4.items():
        if sid not in e2 or sid not in v3:
            continue
        e, t = e2[sid], v3[sid]
        assert r["scores"]["B2"] == e["scores"]["B2"] == t["scores"]["B2"], ("B2 replay differs", sid)
        m = dict(e)
        m["scores"] = {"B2": e["scores"]["B2"], "E2D": e["scores"]["D100"], "T0": t["scores"]["T0"], "T1": t["scores"]["T1"], "V31": r["scores"]["V31"], "V4": r["scores"]["V4"]}
        m["pred"] = {"B2": e["pred"]["B2"], "E2D": e["pred"]["D100"], "T0": t["pred"]["T0"], "T1": t["pred"]["T1"], "V31": r["pred"]["V31"], "V4": r["pred"]["V4"]}
        m["timing"] = {**t.get("timing", {}), **r.get("timing", {})}
        rows.append(m)
    return rows


def rows_to_recs(sset, rows):
    from scipy.stats import kendalltau, spearmanr
    import ab_load as AL
    dname = "indoor" if sset == "valsel" else "outdoor"
    if sset == "valsel":
        man = {s["sid"]: s for s in json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]}
        roots = {s: os.path.join(EP.W, "valsel") for s in man}
    else:
        m0 = {s["sid"]: s for s in json.load(open(os.path.join(EA.ORIG_W, "manifests", "states_outdoor.json")))["states"]}
        m2 = {s["sid"]: s for s in json.load(open(os.path.join(EA.P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
        man, roots = {}, {}
        for r in rows:
            man[r["sid"]] = m0[r["sid"]] if r["stratum"] == "B0" else m2[r["sid"]]
            roots[r["sid"]] = EA.ORIG_W if r["stratum"] == "B0" else EA.P.ROOT_B2
    recs = []
    for r in rows:
        st = man[r["sid"]]
        o = np.load(os.path.join(roots[r["sid"]], "states", dname, r["sid"], "oracle.npz"))
        C = np.asarray(st["cand"], int)
        g = np.asarray(o["gain"], float)[C] * 100.0
        src = st["sources"][0] if sset == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M")
        rec = {k: np.asarray(x) for k, x in np.load(os.path.join(REPO, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
        loc = {int(c): i for i, c in enumerate(C)}
        x = {"sid": r["sid"], "scene": r["scene"], "start": r["start"], "t": r["t"], "phase": r["phase"], "src": r["stratum"]}
        for m in MODELS:
            s = np.asarray([float(np.mean(k)) for k in r["scores"][m]])
            pk = AL.pick_of(dname, st, s, rec, "V2M")[0]
            gp = g[loc[pk]]
            srt = np.sort(s[C])[::-1]
            x.update({f"{m}_gain": float(gp), f"{m}_oracle": float(g.max()), f"{m}_top1": float(gp >= g.max() - EA.TAU), f"{m}_regret": float(g.max() - gp),
                      f"{m}_rho": EA.corr(spearmanr, s[C], g), f"{m}_tau": EA.corr(kendalltau, s[C], g), f"{m}_pick": int(pk),
                      f"{m}_margin": float((srt[0] - srt[1]) / max(abs(srt[0]), 1e-9)) if len(srt) > 1 else None})
            p = r["pred"][m]
            for grp in ("all", "unknown", "new", "known", "unknown_new"):
                x[f"{m}_mae_{grp}"] = EA.ratio(p[f"sae_{grp}"], p[f"n_{grp}"])
            x[f"{m}_hit_agree"] = EA.ratio(p["hit_agree"], p["n_rays"])
            x[f"{m}_rel_std"] = EA.ratio(p.get("rel_std_sum", 0.0), p["n_allhit"])
            x[f"{m}_hit_dis"] = EA.ratio(p["hit_dis"], p["n_rays"] / 4)
            x[f"{m}_spread_err_rho"] = p.get("rho_mean")
            tm = r.get("timing", {}).get(m)
            if tm:
                x[f"{m}_time_s"] = float(tm.get("total_s", 0.0))
        recs.append(x)
    return recs


def gates(S):
    """V3 protocol §4 G2 -> G3 rule (per arm vs B2; informative only: G3 runs regardless)."""
    out = {}
    for arm in ("V31", "V4"):
        per = {}
        for d in ("valsel", "dev3"):
            M, E = S[d]["all"]["means"], S[d]["early"]["means"]
            mv = lambda k, MM=M: MM[k]["mean"]  # noqa: E731
            per[d] = {"regret_ok": mv(f"{arm}_regret") <= 1.05 * mv("B2_regret"), "top1_ok": mv(f"{arm}_top1") >= mv("B2_top1") - 0.03,
                      "early_le_B2": mv(f"{arm}_regret", E) <= mv("B2_regret", E), "not_degraded": mv(f"{arm}_regret") <= 1.10 * mv("B2_regret"),
                      "values": {k: mv(k) for k in (f"{arm}_gain", "B2_gain", f"{arm}_top1", "B2_top1", f"{arm}_regret", "B2_regret")}}
        a1 = all(per[d]["regret_ok"] and per[d]["top1_ok"] for d in per)
        a2 = any(per[d]["early_le_B2"] and per[o]["not_degraded"] for d, o in (("valsel", "dev3"), ("dev3", "valsel")))
        out[arm] = {"per": per, "G2_pass": bool(a1 or a2), "rule": "A" if a1 else ("B" if a2 else None)}
    return out


def tables(S, G, tag):
    f3 = EA.f3
    L = [f"# V4 / V3.1 G1 / G2（开发集固定状态，checkpoint tag {tag}，自动生成）", "",
         "固定状态主指标 = 被选动作的真实一步覆盖增益（pp，越大越好）与 Top-1；regret 只在附录。oracle = 候选中的最大真实增益。", "",
         "G2 → G3 放行（V3 协议 §4，仅供参考）：" + "；".join(f"{arm} {'PASS' if G[arm]['G2_pass'] else 'FAIL'}（规则 {G[arm]['rule']}）" for arm in ("V31", "V4")), ""]
    for d in ("valsel", "dev3"):
        L += [f"## {NAME[d]}（{S[d]['n']} 个状态）", "", "| 阶段 | 状态数 | 指标 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
        for sub in ("early", "middle", "late", "all"):
            if not S[d].get(sub):
                continue
            for met in ("gain", "oracle", "top1", "rho", "mae_unknown", "mae_new", "mae_known", "mae_all", "hit_agree", "rel_std", "spread_err_rho", "time_s"):
                if f"{MODELS[0]}_{met}" not in S[d][sub]["means"] and f"V4_{met}" not in S[d][sub]["means"]:
                    continue
                L.append(f"| {sub} | {S[d][sub]['n_states']} | {met} | " + " | ".join(f3(S[d][sub]['means'].get(f'{m}_{met}')) for m in MODELS) + " |")
        L += ["", "| 配对差（全部状态） | " + " | ".join(f"{x}−{y}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("gain", "top1", "mae_unknown", "mae_new", "mae_known", "rel_std"):
            L.append(f"| {met} | " + " | ".join(f"{f3(S[d]['all']['paired'][f'{x}-{y}|{met}'])}（低 {S[d]['all']['paired'][f'{x}-{y}|{met}']['lower']} / 高 {S[d]['all']['paired'][f'{x}-{y}|{met}']['higher']}）" for x, y in PAIRS) + " |")
        L += ["", "逐场景 被选动作真实增益 pp（全部状态）：", ""]
        for m in MODELS:
            L.append(f"- {LAB[m]}：" + "，".join(f"{k} {v:.3f}" for k, v in S[d]["all"]["per_scene"][f"{m}_gain"].items()))
        L.append("")
    L += ["## 附录：regret（oracle − 被选增益，pp）", "", "| 集 | 阶段 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    for d in ("valsel", "dev3"):
        for sub in ("early", "middle", "late", "all"):
            if S[d].get(sub):
                L.append(f"| {NAME[d]} | {sub} | " + " | ".join(f3(S[d][sub]['means'].get(f'{m}_regret')) for m in MODELS) + " |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s_final")
    a = ap.parse_args()
    EA.MODELS, EA.LAB, EA.COL, EA.PAIRS = MODELS, LAB, COL, PAIRS
    EA.OUT, EA.FIG = os.path.join(E4, "reports"), os.path.join(E4, "reports", "figures")
    os.makedirs(EA.OUT, exist_ok=True)
    os.makedirs(EA.FIG, exist_ok=True)
    EA.ST.setup()
    S = {}
    for d in ("valsel", "dev3"):
        recs = rows_to_recs(d, load(d, a.tag))
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
    G = gates(S)
    json.dump({"tag": a.tag, "summary": S, "gates": G}, open(os.path.join(EA.OUT, f"g12_v4_{a.tag}.json"), "w"), indent=1, default=float)
    open(os.path.join(EA.OUT, f"g12_v4_{a.tag}.md"), "w").write(tables(S, G, a.tag))
    try:
        EA.fig_g12(S, f"v4_{a.tag}")
    except Exception as e:                                   # noqa: BLE001
        print("figure skipped:", e)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
