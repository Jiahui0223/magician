"""EXP6 V5.2-M -- G1 / G2 analysis of one evaluated checkpoint (seen2scene env, CPU; read-only diagnostics).

G2: the V5.2-M rows (eval_fixed_v52m.py output: <rows>/<set>/<stratum>/*.jsonl) are merged by sid with the historical
fixed-state rows -- B2 / E2-D100 (E2), V4 (exp6_v4), V5 / V4C (exp6_v5), old V5.2 pilot (exp6_v5_2, lambda 0 / 0.1) --
same states / candidates / choice rule (analyze_g12_v4.rows_to_recs -> ab_load.pick_of). Metrics: true one-step gain of
the pick (pp), Top-1, Spearman, paired scene-level CI, pick-difference shares, lambda flips, potential share.
G1: first-hit of every candidate view vs oracle t* (NLL, top interval, no-hit reliability, high-confidence false no-hit
p_nohit > 0.95 on true hits, FREE violations, error by distance band, UNKNOWN vs known) and the same calibration-bin
statistic as the old pilot report ("p_nohit in [0.9, 1] -> actual no-hit share"), old V5.2 alongside.
Usage: python analyze_g12_v52m.py --rows DIR --name TAG --out-json J --out-md MD"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, R)
sys.path.insert(0, os.path.join(R, "experiments", "exp6_v4", "scripts"))
import analyze_g12_v4 as A4  # noqa: E402

EA = A4.EA
W5 = os.path.join(R, "work_dirs", "exp6_v5", "eval_fixed")
W52 = os.path.join(R, "work_dirs", "exp6_v5_2", "eval_fixed")
MODELS = ("B2", "E2D", "V4", "V4C", "V5", "V52L0", "V52L01", "V52ML0", "V52ML01")
LAB = dict(A4.LAB, V4="V4@40k", V4C="V4-Continue12k", V5="V5-Warm12k", V52L0="old V5.2 λ0", V52L01="old V5.2 λ0.1", V52ML0="V5.2-M λ0", V52ML01="V5.2-M λ0.1")
COL = dict(A4.COL, V4C="#495057", V5="#7048e8", V52L0="#94d82d", V52L01="#ffa94d", V52ML0="#2b8a3e", V52ML01="#d9480f")
PAIRS = (("V52ML0", "V52L0"), ("V52ML01", "V52L01"), ("V52ML0", "V5"), ("V52ML01", "V5"), ("V52ML01", "V52ML0"), ("V52ML0", "V4C"), ("V52ML0", "B2"), ("V52ML0", "E2D"))
DUMMY = {**{f"sae_{g}": 0.0 for g in ("all", "unknown", "new", "known", "unknown_new")}, **{f"n_{g}": 0 for g in ("all", "unknown", "new", "known", "unknown_new")},
         "hit_agree": 0, "n_rays": 1, "rel_std_sum": 0.0, "n_allhit": 0, "hit_dis": 0, "rho_mean": None}


def rows_of(root):
    out = {}
    import glob
    for f in glob.glob(os.path.join(root, "*", "*.jsonl")):
        for x in open(f):
            r = json.loads(x)
            out[r["sid"]] = r
    return out


def load(sset, rows_dir):
    base = {r["sid"]: r for r in A4.load(sset, "s_final_v4", "s_final_v31")}
    hist = [A4.read_rows(os.path.join(W5, f"{sset}_s_final_{a}")) for a in ("v5", "v4c")]
    old = A4.read_rows(os.path.join(W52, f"{sset}_s_final"))
    new = rows_of(os.path.join(rows_dir, sset))
    rows, extra, oldx = [], {}, {}
    for sid, r in base.items():
        if sid not in new or sid not in old or not all(sid in h for h in hist):
            continue
        for h in hist:
            n = h[sid]
            assert n["scores"]["B2"] == r["scores"]["B2"], ("B2 replay differs", sid)
            for m in ("V5", "V4C"):
                if m in n["scores"]:
                    r["scores"][m], r["pred"][m] = n["scores"][m], n["pred"][m]
        o, x = old[sid], new[sid]
        r["scores"]["V52L0"] = [[s] for s in o["v52"]["score_l0"]]; r["scores"]["V52L01"] = [[s] for s in o["v52"]["score_l0.1"]]
        r["scores"]["V52ML0"] = [[s] for s in x["v52m"]["score_l0"]]; r["scores"]["V52ML01"] = [[s] for s in x["v52m"]["score_l0.1"]]
        for m in ("V52L0", "V52L01", "V52ML0", "V52ML01"):
            r["pred"][m] = DUMMY
        extra[sid], oldx[sid] = x, o
        rows.append(r)
    return rows, extra, oldx


def merge(a, b):
    for g, v in b.items():
        if g not in a:
            a[g] = json.loads(json.dumps(v))
            continue
        for k, x in v.items():
            a[g][k] = [p + q for p, q in zip(a[g][k], x)] if isinstance(x, list) else a[g][k] + x
    return a


def fh_new(acc):
    out = {}
    for g, v in acc.items():
        n = max(v["n"], 1)
        cn = np.asarray(v["cal_n"], float)
        out[g] = {"rays": v["n"], "nll": v["nll"] / n, "top_acc": v["top"] / n, "hit_rate": v["n_hit"] / n,
                  "nohit_ece": float(np.abs(np.asarray(v["cal_p"]) - np.asarray(v["cal_y"])).sum() / max(cn.sum(), 1)),
                  "hc_false_nohit_rate": v["hc_nohit_true_hit"] / max(v["n_hit"], 1), "hc_nohit_precision": 1 - v["hc_nohit_true_hit"] / max(v["hc_nohit_pred"], 1),
                  "hc_nohit_pred_share": v["hc_nohit_pred"] / n, "bin_09_1_actual_nohit": (v["cal_y"][-1] / v["cal_n"][-1]) if v["cal_n"][-1] else None,
                  "bin_09_1_rays": v["cal_n"][-1], "free_violations": v["n_conflict"],
                  **{f"mode_err_vox_{b}": v[f"err_{b}"] / max(v[f"n_err_{b}"], 1) for b in ("near", "mid", "far")},
                  "reliability": [{"pred": p / max(c, 1), "obs": y / max(c, 1), "n": int(c)} for p, y, c in zip(v["cal_p"], v["cal_y"], v["cal_n"])]}
    return out


def fh_old(acc):
    out = {}
    for g, v in acc.items():
        n = max(v["n"], 1)
        out[g] = {"rays": v["n"], "nll": v["nll"] / n, "top_acc": v["top"] / n, "hit_rate": v["n_hit"] / n,
                  "nohit_ece": float(np.abs(np.asarray(v["cal_p"]) - np.asarray(v["cal_y"])).sum() / max(sum(v["cal_n"]), 1)),
                  "bin_09_1_actual_nohit": (v["cal_y"][-1] / v["cal_n"][-1]) if v["cal_n"][-1] else None, "bin_09_1_rays": v["cal_n"][-1],
                  "mode_err_vox": v["err_mode_vox"] / max(v["n_dm"], 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args()
    A4.MODELS, A4.PAIRS, A4.LAB = MODELS, PAIRS, LAB
    EA.MODELS, EA.LAB, EA.COL, EA.PAIRS = MODELS, LAB, COL, PAIRS
    EA.OUT, EA.FIG = os.path.dirname(a.out_json), os.path.dirname(a.out_json)
    EA.ST.setup()
    S, X, G1n, G1o = {}, {}, {}, {}
    for d in ("valsel", "dev3"):
        rows, extra, oldx = load(d, a.rows)
        if not rows:
            continue
        recs = A4.rows_to_recs(d, rows)
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
        fl = [r for r in recs if r["V52ML0_pick"] != r["V52ML01_pick"]]
        share = [extra[r["sid"]]["v52m"]["parts_l0.1"]["pot_share"][r["V52ML01_pick"]] for r in recs]
        X[d] = {"pick_differs": {f"{m}_vs_{b}": float(np.mean([r[f"{m}_pick"] != r[f"{b}_pick"] for r in recs])) for m in ("V52ML0", "V52ML01") for b in ("V5", "V4C", "B2", "V52L0")},
                "lambda_flip_rate": float(np.mean([r["V52ML0_pick"] != r["V52ML01_pick"] for r in recs])), "n_flips": len(fl),
                "flip_gain_delta_pp": float(np.mean([r["V52ML01_gain"] - r["V52ML0_gain"] for r in fl])) if fl else None,
                "pot_share_of_pick_mean": float(np.mean(share)), "score_s_per_state_mean": float(np.mean([extra[r["sid"]]["timing"]["v52m_score_s"] for r in recs])),
                "free_frac_mean": float(np.mean([np.mean(extra[r["sid"]]["v52m"]["free_frac_mean"]) for r in recs])),
                "visible_mass_on_D100_free_frac_mean": float(np.mean([np.mean([dg["mass_free"] / max(dg["mass_dom"], 1e-9) for dg in extra[r["sid"]]["v52m"]["diag"]]) for r in recs]))}
        an, ao = {}, {}
        for r in recs:
            an = merge(an, extra[r["sid"]]["firsthit"])
            ao = merge(ao, oldx[r["sid"]]["firsthit"])
        G1n[d], G1o[d] = fh_new(an), fh_old(ao)
    out = {"name": a.name, "rows": a.rows, "summary": S, "v52m": X, "g1_new": G1n, "g1_old_v52": G1o}
    json.dump(out, open(a.out_json, "w"), indent=1, default=float)
    f3 = EA.f3
    ff = lambda x: '—' if x is None else f'{x:.3f}'  # noqa: E731
    L = [f"# V5.2-M G1 / G2 —— {a.name}（固定状态，只读诊断；自动生成）", ""]
    for d, nm in (("dev3", "Macarons++ Dev-3"), ("valsel", "MP3D VALSEL")):
        if d not in S:
            continue
        L += [f"## {nm}（{S[d]['n']} 态）", "", "| 阶段 | 指标 | " + " | ".join(LAB[m] for m in MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
        for sub in ("early", "all"):
            for met in ("gain", "top1", "rho"):
                L.append(f"| {sub} | {met} | " + " | ".join(f3(S[d][sub]["means"].get(f"{m}_{met}")) for m in MODELS) + " |")
        L += ["", "| 配对差（全部状态） | " + " | ".join(f"{x}−{y}" for x, y in PAIRS) + " |", "|---|" + "---|" * len(PAIRS)]
        for met in ("gain", "top1"):
            L.append(f"| {met} | " + " | ".join(f"{f3(S[d]['all']['paired'][f'{x}-{y}|{met}'])}" for x, y in PAIRS) + " |")
        x = X[d]
        L += ["", "- " + "，".join(f"{k} {v:.3f}" for k, v in x["pick_differs"].items()),
              f"- λ 翻转 {x['lambda_flip_rate']:.3f}（{x['n_flips']} 态），λ0.1 被选候选潜力占比 {x['pot_share_of_pick_mean']:.3f}；打分 {x['score_s_per_state_mean']:.2f} s/态；强制 FREE 区间比例 {x['free_frac_mean']:.3f}；可见质量落在 D100 FREE 的比例 {x['visible_mass_on_D100_free_frac_mean']:.3f}", "",
              "| First-Hit | 射线组 | 射线 | NLL | 最高区间准确率 | no-hit ECE | p_nohit∈[0.9,1] 实际 no-hit | p_nohit>0.95 的真命中率（false no-hit） | 众数误差 near/mid/far vox | FREE 冲突 |", "|---|---|---|---|---|---|---|---|---|---|"]
        for grp in ("all", "unknown", "known"):
            v = G1n[d].get(grp)
            if v:
                L.append(f"| V5.2-M | {grp} | {v['rays']:,} | {v['nll']:.3f} | {v['top_acc']:.3f} | {v['nohit_ece']:.3f} | {ff(v['bin_09_1_actual_nohit'])}（{v['bin_09_1_rays']:,}） | "
                         f"{v['hc_false_nohit_rate']:.4f} | {v['mode_err_vox_near']:.1f} / {v['mode_err_vox_mid']:.1f} / {v['mode_err_vox_far']:.1f} | {v['free_violations']} |")
        for grp in ("all", "unknown"):
            v = G1o[d].get(grp)
            if v:
                L.append(f"| 旧 V5.2 | {grp} | {v['rays']:,} | {v['nll']:.3f} | {v['top_acc']:.3f} | {v['nohit_ece']:.3f} | {ff(v['bin_09_1_actual_nohit'])}（{v['bin_09_1_rays']:,}） | — | {v['mode_err_vox']:.1f}（全部） | — |")
        L.append("")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    print(json.dumps({d: {m: {k: S[d]["all"]["means"][f"{m}_{k}"]["mean"] for k in ("gain", "top1")} for m in ("V5", "V52L0", "V52ML0", "V52ML01")} for d in S}, default=float))


if __name__ == "__main__":
    main()
