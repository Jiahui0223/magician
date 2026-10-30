"""EXP6 V3 -- G1 / G2 analysis on the development fixed states (seen2scene env, CPU).

Models: B2 and E2-D100 (scores / predictions taken from the E2 pilot rows, fixed/<set>_s5000), T0 and T1 (eval_fixed
rows). Ranking under the Task 2 choice rule (ab_load.pick_of), prediction groups, K = 4 spread; aggregation state ->
episode -> scene, scene-level bootstrap; paired T1-T0, T0-B2, T1-B2, T0-E2D, T1-E2D; phases; the frozen G2 -> G3 rule
(protocol §4). Output exp6_v3/reports/g12_<tag>.{json,md} and figures/g12_<tag>.png.
Usage: python analyze_g12.py --tag s_final
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

from seen2scene import PROJECT_ROOT

REPO = str(PROJECT_ROOT)
E2S = os.path.join(REPO, "experiments", "exp5_future_view", "task3_full_depth_p1", "e2_pilot", "scripts")
sys.path.insert(0, E2S)
import e2_analyze as EA  # noqa: E402

EP = EA.EP
E6 = os.path.join(REPO, "experiments", "exp6_v3")
W6 = os.path.join(REPO, "work_dirs", "exp6_v3", "eval_fixed")
MODELS = ("B2", "E2D", "T0", "T1")
LAB = {"B2": "B2 frozen V2-Mix, D100", "E2D": "E2-D100 fine-tuned", "T0": "V3-VAE (T0)", "T1": "V3-VAE-XAttn (T1)"}
COL = {"B2": EA.ST.BLUE[11], "E2D": "#eb6834", "T0": "#1baf7a", "T1": "#8e44ad"}
PAIRS = (("T1", "T0"), ("T0", "B2"), ("T1", "B2"), ("T0", "E2D"), ("T1", "E2D"))
NAME = EA.NAME
plt = EA.plt


def load(sset, tag):
    """Rows keyed by sid: E2 rows (B2 / D100 scores + preds) joined with the V3 rows (T0 / T1)."""
    e2 = {}
    for f in sorted(glob.glob(os.path.join(EP.W, "fixed", f"{sset}_s5000", "*", "*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            e2[r["sid"]] = r
    v3 = {}
    for f in sorted(glob.glob(os.path.join(W6, f"{sset}_{tag}", "*", "*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            v3[r["sid"]] = r
    rows = []
    for sid, r in v3.items():
        if sid not in e2:
            continue
        e = e2[sid]
        assert r["scores"]["B2"] == e["scores"]["B2"], ("B2 replay differs", sid)
        m = dict(e)
        m["scores"] = {"B2": e["scores"]["B2"], "E2D": e["scores"]["D100"], "T0": r["scores"]["T0"], "T1": r["scores"]["T1"]}
        m["pred"] = {"B2": e["pred"]["B2"], "E2D": e["pred"]["D100"], "T0": r["pred"]["T0"], "T1": r["pred"]["T1"]}
        rows.append(m)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="s_final")
    a = ap.parse_args()
    EA.MODELS, EA.LAB, EA.COL, EA.PAIRS = MODELS, LAB, COL, PAIRS
    EA.OUT, EA.FIG = os.path.join(E6, "reports"), os.path.join(E6, "reports", "figures")
    os.makedirs(EA.OUT, exist_ok=True)
    os.makedirs(EA.FIG, exist_ok=True)
    EA.ST.setup()
    S = {}
    for d in ("valsel", "dev3"):
        rows = load(d, a.tag)
        # reuse the E2 loader's metric construction by feeding it pre-merged rows
        recs = EA.load_rows(d, rows) if hasattr(EA, "load_rows") else None
        if recs is None:
            recs = rows_to_recs(d, rows)
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
    G = gates(S)
    json.dump({"tag": a.tag, "summary": S, "gates": G}, open(os.path.join(EA.OUT, f"g12_{a.tag}.json"), "w"), indent=1, default=float)
    open(os.path.join(EA.OUT, f"g12_{a.tag}.md"), "w").write(tables(S, G, a.tag))
    EA.fig_g12(S, a.tag)
    print(json.dumps(G, indent=1, default=float))


def rows_to_recs(sset, rows):
    """The metric rows of e2_analyze.load() for pre-merged rows (scores / pred of MODELS)."""
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
            x.update({f"{m}_top1": float(gp >= g.max() - EA.TAU), f"{m}_regret": float(g.max() - gp), f"{m}_rho": EA.corr(spearmanr, s[C], g),
                      f"{m}_tau": EA.corr(kendalltau, s[C], g), f"{m}_pick": int(pk),
                      f"{m}_margin": float((srt[0] - srt[1]) / max(abs(srt[0]), 1e-9)) if len(srt) > 1 else None})
            p = r["pred"][m]
            for grp in ("all", "unknown", "new", "known", "unknown_new"):
                x[f"{m}_mae_{grp}"] = EA.ratio(p[f"sae_{grp}"], p[f"n_{grp}"])
            x[f"{m}_hit_agree"] = EA.ratio(p["hit_agree"], p["n_rays"])
            x[f"{m}_rel_std"] = EA.ratio(p.get("rel_std_sum", 0.0), p["n_allhit"])
            x[f"{m}_hit_dis"] = EA.ratio(p["hit_dis"], p["n_rays"] / 4)
            x[f"{m}_spread_err_rho"] = p.get("rho_mean")
        recs.append(x)
    return recs


def tables(S, G, tag):
    f3 = EA.f3
    L = [f"# V3 G1 / G2（开发集固定状态，checkpoint tag {tag}，自动生成）", "", "G2 → G3 放行（协议 §4）：" +
         "；".join(f"{arm} {'PASS' if G[arm]['G2_pass'] else 'FAIL'}（规则 {G[arm]['rule']}）" for arm in ("T0", "T1")), "",
         "```", json.dumps(G, indent=1, default=float), "```", ""]
    for d in ("valsel", "dev3"):
        L += [f"## {NAME[d]}（{S[d]['n']} 个状态）", "", "| 阶段 | 状态数 | 指标 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|---|" + "---|" * len(MODELS)]
        for sub in ("early", "middle", "late", "all"):
            if not S[d].get(sub):
                continue
            for met in ("regret", "top1", "rho", "margin", "mae_all", "mae_unknown", "mae_new", "mae_known", "rel_std", "hit_dis", "spread_err_rho", "hit_agree"):
                L.append(f"| {sub} | {S[d][sub]['n_states']} | {met} | " + " | ".join(f3(S[d][sub]['means'].get(f'{m}_{met}')) for m in MODELS) + " |")
        L += ["", "| 配对差（全部状态） | " + " | ".join(f"{x}−{y}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("regret", "top1", "mae_unknown", "mae_new", "mae_known", "rel_std"):
            L.append(f"| {met} | " + " | ".join(f"{f3(S[d]['all']['paired'][f'{x}-{y}|{met}'])}（低 {S[d]['all']['paired'][f'{x}-{y}|{met}']['lower']} / 高 {S[d]['all']['paired'][f'{x}-{y}|{met}']['higher']}）" for x, y in PAIRS) + " |")
        L += ["", "| 配对差（Early） | " + " | ".join(f"{x}−{y}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("regret", "top1"):
            L.append(f"| {met} | " + " | ".join(f3(S[d]['early']['paired'][f'{x}-{y}|{met}']) for x, y in PAIRS) + " |")
        L += ["", "逐场景 regret（全部状态）：", ""]
        for m in MODELS:
            L.append(f"- {LAB[m]}：" + "，".join(f"{k} {v:.3f}" for k, v in S[d]["all"]["per_scene"][f"{m}_regret"].items()))
        L.append("")
    return "\n".join(L)


def gates(S):
    """Protocol §4 G2 -> G3 (per arm, vs B2): overall regret <= 1.05 B2 and Top-1 >= B2 - 0.03 on both sets, or Early
    regret <= B2 on one set with the other not degraded (regret <= 1.10 B2)."""
    out = {}
    for arm in ("T0", "T1"):
        per = {}
        for d in ("valsel", "dev3"):
            M, E = S[d]["all"]["means"], S[d]["early"]["means"]
            mv = lambda k, MM=M: MM[k]["mean"]  # noqa: E731
            per[d] = {"regret_ok": mv(f"{arm}_regret") <= 1.05 * mv("B2_regret"), "top1_ok": mv(f"{arm}_top1") >= mv("B2_top1") - 0.03,
                      "early_le_B2": mv(f"{arm}_regret", E) <= mv("B2_regret", E), "not_degraded": mv(f"{arm}_regret") <= 1.10 * mv("B2_regret"),
                      "values": {k: mv(k) for k in (f"{arm}_regret", "B2_regret", f"{arm}_top1", "B2_top1")} | {f"{arm}_early": mv(f"{arm}_regret", E), "B2_early": mv("B2_regret", E)}}
        a1 = all(per[d]["regret_ok"] and per[d]["top1_ok"] for d in per)
        a2 = any(per[d]["early_le_B2"] and per[o]["not_degraded"] for d, o in (("valsel", "dev3"), ("dev3", "valsel")))
        out[arm] = {"per": per, "G2_pass": bool(a1 or a2), "rule": "A" if a1 else ("B" if a2 else None)}
    return out


if __name__ == "__main__":
    main()
