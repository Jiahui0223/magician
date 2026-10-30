"""EXP5 Task 3 / E2 pilot -- G1 / G2 analysis of the development fixed states + training curves (seen2scene env, CPU).

Ranking under the Task 2 choice rule (ab_load.pick_of) of B0, B2, S10 (E2-S10), D100 (E2-D100), all with sparse DistP1;
prediction groups (all / unknown / truly new / known), K = 4 spread; aggregation state -> episode -> scene, scene-level
bootstrap; paired D100-B2, S10-B0, D100-S10, D100-B0; frozen gates G1 / G2 (protocol §5).
Usage: python e2_analyze.py [--step 5000]   Outputs analysis/g12_s<step>.json, analysis/g12_s<step>.md, figures/*.png
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2_paths as EP  # noqa: E402

sys.path.insert(0, os.path.join(EP.EXP5, "task3_full_depth_p1", "density_ablation", "scripts"))
import e12_paths as P  # noqa: E402

A = P.A
ORIG_W = A.W
import ab_load as AL  # noqa: E402
import ab_style as ST  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

MODELS = ("B0", "B2", "S10", "D100")
LAB = {"B0": "B0 frozen, D10", "B2": "B2 frozen, D100", "S10": "E2-S10", "D100": "E2-D100"}
COL = {"B0": ST.BLUE[6], "B2": ST.BLUE[11], "S10": "#f2a07b", "D100": "#eb6834"}
PAIRS = (("D100", "B2"), ("S10", "B0"), ("D100", "S10"), ("D100", "B0"))
TAU = 1e-4 * 100
OUT, FIG = os.path.join(EP.E2, "analysis"), os.path.join(EP.E2, "figures")
NAME = {"valsel": "MP3D VALSEL (dev)", "dev3": "Macarons++ Dev-3"}
plt = ST.plt


def corr(f, a, b):
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    r = f(a, b)
    r = r.correlation if hasattr(r, "correlation") else r[0]
    return float(r) if np.isfinite(r) else None


def ratio(a, b):
    return float(a) / float(b) if b else None


def load(sset, step):
    rows = []
    for f in sorted(glob.glob(os.path.join(EP.W, "fixed", f"{sset}_s{step}", "*", "*.jsonl"))):
        rows += [json.loads(x) for x in open(f)]
    if sset == "valsel":
        man = {s["sid"]: s for s in json.load(open(os.path.join(EP.W, "valsel", "manifests", "states_valsel.json")))["states"]}
        roots = {s: os.path.join(EP.W, "valsel") for s in man}
        dname = "indoor"
    else:
        m0 = {s["sid"]: s for s in json.load(open(os.path.join(ORIG_W, "manifests", "states_outdoor.json")))["states"]}
        m2 = {s["sid"]: s for s in json.load(open(os.path.join(P.ROOT_B2, "manifests", "states_outdoor.json")))["states"]}
        man, roots = {}, {}
        for r in rows:
            man[r["sid"]] = m0[r["sid"]] if r["stratum"] == "B0" else m2[r["sid"]]
            roots[r["sid"]] = ORIG_W if r["stratum"] == "B0" else P.ROOT_B2
        dname = "outdoor"
    recs = []
    for r in rows:
        st = man[r["sid"]]
        o = np.load(os.path.join(roots[r["sid"]], "states", dname, r["sid"], "oracle.npz"))
        C = np.asarray(st["cand"], int)
        g = np.asarray(o["gain"], float)[C] * 100.0
        src = st["sources"][0] if sset == "valsel" else next(e for e in st["sources"] if e["source"] == "V2M")
        rec = {k: np.asarray(x) for k, x in np.load(os.path.join(EP.REPO, src["svc"])).items() if k in ("legal_idx", "executed_pose_idx_history")}
        loc = {int(c): i for i, c in enumerate(C)}
        x = {"sid": r["sid"], "scene": r["scene"], "start": r["start"], "t": r["t"], "phase": r["phase"], "src": r["stratum"]}
        for m in MODELS:
            s = np.asarray([float(np.mean(k)) for k in r["scores"][m]])
            pk = AL.pick_of(dname, st, s, rec, "V2M")[0]
            gp = g[loc[pk]]
            x.update({f"{m}_top1": float(gp >= g.max() - TAU), f"{m}_regret": float(g.max() - gp), f"{m}_rho": corr(spearmanr, s[C], g),
                      f"{m}_tau": corr(kendalltau, s[C], g), f"{m}_pick": int(pk)})
            p = r["pred"][m]
            for grp in ("all", "unknown", "new", "known", "unknown_new"):
                x[f"{m}_mae_{grp}"] = ratio(p[f"sae_{grp}"], p[f"n_{grp}"])
            x[f"{m}_hit_agree"] = ratio(p["hit_agree"], p["n_rays"])
            x[f"{m}_rel_std"] = ratio(p.get("rel_std_sum", 0.0), p["n_allhit"])
            x[f"{m}_hit_dis"] = ratio(p["hit_dis"], p["n_rays"] / 4)
            x[f"{m}_hit_dis_unk"] = ratio(p["hit_dis_unk"], p["n_unk_rays"])
            x[f"{m}_spread_err_rho"] = p.get("rho_mean")
        if sset == "valsel":
            assert x["B0_pick"] is not None
        recs.append(x)
    return recs


def hm(rr, key):
    return AL.hier_mean(rr, key)


def summ(recs, sub):
    rr = recs if sub == "all" else [r for r in recs if r["phase"] == sub]
    keys = sorted({k for r in rr for k in r if k.split("_")[0] in MODELS and not k.endswith("_pick")})
    out = {"n_states": len(rr), "means": {k: AL.boot(hm(rr, k)) for k in keys}, "per_scene": {k: hm(rr, k) for k in keys}, "paired": {}}
    for a, b in PAIRS:
        for met in sorted({k.split("_", 1)[1] for k in keys}):
            ha, hb = hm(rr, f"{a}_{met}"), hm(rr, f"{b}_{met}")
            dd = {s: ha[s] - hb[s] for s in ha if s in hb}
            bo = AL.boot(dd)
            bo.update({"lower": int(sum(v < -1e-12 for v in dd.values())), "higher": int(sum(v > 1e-12 for v in dd.values())), "per_scene": dd})
            out["paired"][f"{a}-{b}|{met}"] = bo
    return out


def gates(S):
    g1, g2 = {}, {}
    for d in ("valsel", "dev3"):
        M = S[d]["all"]["means"]
        mv = lambda k: M[k]["mean"]  # noqa: E731
        g1[d] = {"a_unknown": mv("D100_mae_unknown") <= mv("B2_mae_unknown"), "a_new": mv("D100_mae_new") <= mv("B2_mae_new"),
                 "b_spread": (mv("D100_rel_std") >= 0.8 * mv("B2_rel_std")) or ((mv("D100_spread_err_rho") or -9) >= (mv("B2_spread_err_rho") or 9) - 0.02),
                 "values": {k: mv(k) for k in ("D100_mae_unknown", "B2_mae_unknown", "D100_mae_new", "B2_mae_new", "D100_rel_std", "B2_rel_std",
                                               "D100_spread_err_rho", "B2_spread_err_rho")}}
        E = S[d]["early"]["means"]
        g2[d] = {"regret_ok": mv("D100_regret") <= 1.05 * mv("B2_regret"), "top1_ok": mv("D100_top1") >= mv("B2_top1") - 0.03,
                 "early_le_B2": E["D100_regret"]["mean"] <= E["B2_regret"]["mean"],
                 "values": {"D100_regret": mv("D100_regret"), "B2_regret": mv("B2_regret"), "D100_top1": mv("D100_top1"), "B2_top1": mv("B2_top1"),
                            "D100_early": E["D100_regret"]["mean"], "B2_early": E["B2_regret"]["mean"]}}
    G1 = all(g1[d]["a_unknown"] and g1[d]["a_new"] and g1[d]["b_spread"] for d in g1)
    G2 = all(g2[d]["regret_ok"] and g2[d]["top1_ok"] for d in g2) and any(g2[d]["early_le_B2"] for d in g2)
    return {"G1_ab": g1, "G1_ab_pass": bool(G1), "G2": g2, "G2_pass": bool(G2)}


def training_c():
    out = {}
    for arm in ("s10", "d100"):
        d = os.path.join(EP.W, "training", arm)
        ev = [json.loads(x) for x in open(os.path.join(d, "eval.jsonl"))] if os.path.exists(os.path.join(d, "eval.jsonl")) else []
        lg = [json.loads(x) for x in open(os.path.join(d, "log.jsonl"))] if os.path.exists(os.path.join(d, "log.jsonl")) else []
        out[arm] = {"eval": ev, "log": lg}
    return out


def fm_of(ev, nm):
    v = ev.get(f"fm_{nm}")
    if isinstance(v, dict):
        for k in ("loss", "fm", "mean", "all"):
            if k in v:
                return v[k]
        vals = [x for x in v.values() if isinstance(x, (int, float))]
        return float(np.mean(vals)) if vals else None
    return v


def fig_training(TC):
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for arm, col in (("s10", COL["S10"]), ("d100", COL["D100"])):
        lg = TC[arm]["log"]
        if lg:
            axs[0].plot([r["step"] for r in lg], [r["loss"] for r in lg], color=col, lw=1.4, label=f"E2-{arm.upper()} train loss (mean of 50 steps)")
        ev = TC[arm]["eval"]
        for k, ax, nm in (("in_val", axs[1], "VALSEL"), ("out_sc", axs[2], "S-c")):
            xs = [e["step"] for e in ev if fm_of(e, k) is not None]
            ys = [fm_of(e, k) for e in ev if fm_of(e, k) is not None]
            if xs:
                ax.plot(xs, ys, marker="o", color=col, label=f"E2-{arm.upper()}")
            ax.set_title(f"validation FM loss, {nm} (each arm on its own P1 density)", loc="left")
    axs[0].set_title("training loss", loc="left")
    for ax in axs:
        ax.set_xlabel("optimizer step")
        ST.clean(ax)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "training_curves.png"))
    plt.close(fig)


def fig_g12(S, step):
    fig, axs = plt.subplots(2, 4, figsize=(18, 7.4))
    for i, d in enumerate(("valsel", "dev3")):
        for j, (met, lab) in enumerate((("regret", "oracle regret (pp, lower = better)"), ("top1", "Top-1"), ("mae_unknown", "MAE, unknown rays"),
                                        ("mae_new", "MAE, truly-new surface"))):
            ax = axs[i, j]
            subs = ("early", "middle", "late", "all")
            for k, m in enumerate(MODELS):
                ys = [S[d][s]["means"][f"{m}_{met}"]["mean"] if S[d].get(s) else np.nan for s in subs]
                ax.bar(np.arange(len(subs)) + (k - 1.5) * 0.2, [np.nan if y is None else y for y in ys], 0.19, color=COL[m], label=LAB[m])
            ax.set_xticks(range(len(subs)))
            ax.set_xticklabels(subs)
            ax.set_title(f"{NAME[d]}: {lab}", loc="left", fontsize=9)
            ST.clean(ax)
            if j == 0:
                ax.legend(fontsize=7.5)
    fig.suptitle(f"Task 3 / E2 pilot — G1 / G2 on development fixed states (step {step}; DistP1 always sparse; measured)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"g12_s{step}.png"))
    plt.close(fig)


def f3(b, nd=3):
    if b is None or b.get("mean") is None:
        return "—"
    if b.get("scenes", 0) < 3 or b.get("ci95", [None])[0] is None:
        return f"{b['mean']:.{nd}f}"
    return f"{b['mean']:.{nd}f} [{b['ci95'][0]:.{nd}f}, {b['ci95'][1]:.{nd}f}]"


def tables(S, G, step):
    L = [f"# E2 pilot — G1 / G2（开发集固定状态，step {step}，自动生成）", "", f"G1-a/b 通过：{G['G1_ab_pass']}；G2 通过：{G['G2_pass']}", "",
         "```", json.dumps({k: v for k, v in G.items() if k in ("G1_ab", "G2")}, indent=1, default=float), "```", ""]
    for d in ("valsel", "dev3"):
        L += [f"## {NAME[d]}", "", "| 阶段 | 状态数 | 指标 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|---|" + "---|" * 4]
        for sub in ("early", "middle", "late", "all"):
            if not S[d].get(sub):
                continue
            for met in ("regret", "top1", "rho", "mae_all", "mae_unknown", "mae_new", "mae_known", "rel_std", "hit_dis", "spread_err_rho", "hit_agree"):
                L.append(f"| {sub} | {S[d][sub]['n_states']} | {met} | " + " | ".join(f3(S[d][sub]['means'].get(f'{m}_{met}')) for m in MODELS) + " |")
        L += ["", "| 配对差（全部状态） | " + " | ".join(f"{a}−{b}" for a, b in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("regret", "top1", "mae_unknown", "mae_new", "mae_known", "rel_std"):
            L.append(f"| {met} | " + " | ".join(f"{f3(S[d]['all']['paired'][f'{a}-{b}|{met}'])}（低 {S[d]['all']['paired'][f'{a}-{b}|{met}']['lower']} / 高 {S[d]['all']['paired'][f'{a}-{b}|{met}']['higher']}）" for a, b in PAIRS) + " |")
        L += ["", "逐场景 regret（全部状态）：", ""]
        for m in MODELS:
            L.append(f"- {LAB[m]}：" + "，".join(f"{k} {v:.3f}" for k, v in S[d]["all"]["per_scene"][f"{m}_regret"].items()))
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=5000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    ST.setup()
    S = {}
    for d in ("valsel", "dev3"):
        recs = load(d, a.step)
        S[d] = {sub: summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
    G = gates(S)
    TC = training_c()
    json.dump({"step": a.step, "summary": S, "gates": G}, open(os.path.join(OUT, f"g12_s{a.step}.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, f"g12_s{a.step}.md"), "w").write(tables(S, G, a.step))
    fig_g12(S, a.step)
    fig_training(TC)
    print(json.dumps(G, indent=1, default=float))


if __name__ == "__main__":
    main()
