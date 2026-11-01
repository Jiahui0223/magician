"""Experiment A -- fixed-state analysis (CPU): rows of a_fixed.py -> per set (MP3D VALSEL / Macarons++ Dev-3) and phase:
gain of the pick (pp, true one-step coverage gain of the chosen camera; higher is better), Top-1, mean DistP1 score,
fraction of predicted hit rays judged known, band_dil voxel count, fraction of states where A0 and A1 choose different
cameras; scene-level bootstrap and paired A1-A0. Output results/a_fixed_summary.json + report/A_FIXED_STATE.md."""
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
A = os.path.join(REPO, "experiments", "exp6_v4", "ablation_study", "experiment_A_distp1_dense_vs_sparse")
W = os.path.join(REPO, "work_dirs", "exp6_v4", "ablation_study", "experiment_A", "fixed")
MODELS = ("A0", "A1")
LAB = {"A0": "A0 V4-SparseScore（S10 band_dil）", "A1": "A1 V4-DenseScore（D100 band_dil）"}
PAIRS = (("A1", "A0"),)
NAME = {"valsel": "MP3D VALSEL（zero-shot 开发状态）", "dev3": "Macarons++ Dev-3"}


def read_rows(sset):
    out = []
    for f in sorted(glob.glob(os.path.join(W, sset, "*", "*.jsonl"))):
        out += [json.loads(x) for x in open(f)]
    return out


def recs_of(sset, rows):
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
        picks = {}
        for m in MODELS:
            s = np.asarray([float(np.mean(k)) for k in r["scores"][m]])
            pk = AL.pick_of(dname, st, s, rec, "V2M")[0]
            picks[m] = int(pk)
            gp = g[loc[pk]]
            kn = r["known"][m]
            x.update({f"{m}_gain": float(gp), f"{m}_oracle": float(g.max()), f"{m}_top1": float(gp >= g.max() - EA.TAU), f"{m}_regret": float(g.max() - gp),
                      f"{m}_score_mean": float(np.mean(s[C])), f"{m}_score_pick": float(s[pk]), f"{m}_known_frac": (1.0 - kn["n_new"] / kn["n_hit"]) if kn["n_hit"] else None,
                      f"{m}_band_vox": float(r["band_vox"][m]), f"{m}_pick": picks[m]})
        x["A1_diff_pick"] = float(picks["A1"] != picks["A0"])
        x["A1_gain_vs_A0"] = x["A1_gain"] - x["A0_gain"]
        x["A1_band_ratio"] = x["A1_band_vox"] / max(x["A0_band_vox"], 1.0)
        recs.append(x)
    return recs


def f3(b, nd=3):
    return EA.f3(b, nd)


def main():
    EA.MODELS, EA.LAB, EA.PAIRS = MODELS, LAB, PAIRS
    EA.ST.setup()
    S, L = {}, ["# Experiment A — 固定状态 A0 / A1（同一 V4 预测，不同评分 band_dil；自动生成 `a_fixed_analyze.py`）", "",
                "主指标 = 被选 Camera 的真实一步覆盖增益 pp（越大越好）与 Top-1；regret 在附录。均值 = 场景 / 楼层均值的平均，[CI] = 场景级 bootstrap 95 %。", ""]
    for d in ("valsel", "dev3"):
        rows = read_rows(d)
        recs = recs_of(d, rows)
        S[d] = {sub: EA.summ(recs, sub) for sub in ("early", "middle", "late", "all")}
        S[d]["n"] = len(recs)
        S[d]["repro_A0_eq_V4rows"] = int(sum(bool(r.get("repro_A0_eq_V4rows")) for r in rows))
        S[d]["timing"] = {k: float(np.mean([r["timing"][k] for r in rows if k in r["timing"]])) for k in ("predict_s", "score_dense_s", "score_sparse_s")}
        L += [f"## {NAME[d]}（{S[d]['n']} 个状态；A0 与存档 V4 行逐位相同 {S[d]['repro_A0_eq_V4rows']} / {S[d]['n']}）", "",
              "| 阶段 | 状态数 | 指标 | A0 SparseScore | A1 DenseScore | A1 − A0 [CI]（低 / 高） |", "|---|---|---|---|---|---|"]
        for sub in ("early", "middle", "late", "all"):
            Ssub = S[d][sub]
            if not Ssub.get("n_states"):
                continue
            for met, lab in (("gain", "被选 Camera 真实增益 pp"), ("oracle", "oracle 上限 pp"), ("top1", "Top-1"), ("score_mean", "平均 DistP1 分数"), ("score_pick", "被选候选 DistP1 分数"),
                             ("known_frac", "预测 Hit 被判已知比例"), ("band_vox", "band_dil 体素数"), ("regret", "regret pp（附录）")):
                p = Ssub["paired"].get(f"A1-A0|{met}")
                L.append(f"| {sub} | {Ssub['n_states']} | {lab} | {f3(Ssub['means'].get(f'A0_{met}'))} | {f3(Ssub['means'].get(f'A1_{met}'))} | "
                         + (f"{f3(p)}（{p['lower']} / {p['higher']}）" if p else "—") + " |")
            L.append(f"| {sub} | {Ssub['n_states']} | A0 / A1 选择不同的状态比例 | — | {f3(Ssub['means'].get('A1_diff_pick'))} | — |")
        L += ["", "逐场景 被选增益 pp（全部状态）：" + "；".join(f"{k} A0 {S[d]['all']['per_scene']['A0_gain'][k]:.3f} / A1 {S[d]['all']['per_scene']['A1_gain'][k]:.3f}" for k in S[d]["all"]["per_scene"]["A0_gain"]),
              f"每状态时间（s）：预测 {S[d]['timing']['predict_s']:.2f}，稠密评分 {S[d]['timing']['score_dense_s']:.3f}，稀疏评分 {S[d]['timing']['score_sparse_s']:.3f}", ""]
    os.makedirs(os.path.join(A, "results"), exist_ok=True)
    json.dump(S, open(os.path.join(A, "results", "a_fixed_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(A, "report", "A_FIXED_STATE.md"), "w").write("\n".join(L))
    print(json.dumps({d: {"n": S[d]["n"], "A0_gain": S[d]["all"]["means"]["A0_gain"]["mean"], "A1_gain": S[d]["all"]["means"]["A1_gain"]["mean"],
                          "A0_top1": S[d]["all"]["means"]["A0_top1"]["mean"], "A1_top1": S[d]["all"]["means"]["A1_top1"]["mean"],
                          "known_A0": S[d]["all"]["means"]["A0_known_frac"]["mean"], "known_A1": S[d]["all"]["means"]["A1_known_frac"]["mean"],
                          "diff_pick": S[d]["all"]["means"]["A1_diff_pick"]["mean"], "repro": S[d]["repro_A0_eq_V4rows"]} for d in S}, indent=1))


if __name__ == "__main__":
    main()
