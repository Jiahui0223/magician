"""EXP6 V3 Task 2 / Experiment 1 -- analysis of the fixed-state scores (CPU; protocol §3).

E1-GT : every scorer variant on the true depth t* vs the oracle gain (Spearman / Top-1 / pairwise / regret; phases;
        calibration; dev vs eval); E1-Pred: the same with the cached K = 4 samples of B2 / T1 (main) and T0 / E2D
        (control), K-mean; G2 Screening on the dev sets with the frozen rule and the selection rule; failure cases.
Output results/{e1_summary.json, e1_states.csv}, figures/*.png, report/{E1_GT_SCORING, E1_MODEL_SCORING, E1_FAILURE_CASES}.md
"""
import csv
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
T2 = os.path.dirname(EXP)
E6 = os.path.dirname(T2)
sys.path.insert(0, os.path.join(E6, "ablation_study", "shared"))
sys.path.insert(0, os.path.join(T2, "common"))
import ab_data as D  # noqa: E402
import t2_scorers as TS  # noqa: E402

REPO = D.REPO
WS = os.path.join(REPO, "work_dirs", "exp6_v3", "task2", "exp1", "scores")
OUT, FIG, REP = (os.path.join(EXP, k) for k in ("results", "figures", "report"))
SOURCES = ("GT", "B2", "T1", "T0", "E2D")
MAIN_MODELS = ("B2", "T1")
VARIANTS = list(TS.VARIANTS)
GVARIANTS = list(TS.GLOBAL_VARIANTS)
DEV3, TEST3 = D.DEV3, D.TEST3
LABEL = {"GT": "GT depth", "B2": "B2 (V2-Mix, D100)", "T1": "V3-T1", "T0": "V3-T0", "E2D": "E2-D100"}


def split_of(pool, dataset, scene):
    if dataset == "indoor":
        return "dev" if pool == "fx" else "eval"
    return "dev" if scene in DEV3 else "eval"


def load():
    recs = []
    for pool in ("cl", "fx"):
        for dataset in ("indoor", "outdoor"):
            for f in sorted(glob.glob(os.path.join(WS, pool, dataset, "*.json"))):
                j = json.load(open(f))
                if pool == "fx" and dataset == "outdoor" and j["source"] != "B2":
                    pass
                r = {"pool": pool, "dataset": dataset, "sid": j["sid"], "scene": j["scene"], "start": j["start"], "t": int(j["t"]), "tag": j["tag"],
                     "source": j["source"], "episode": f"{j['tag']}|{j['source']}", "split": split_of(pool, dataset, j["scene"]),
                     "phase": D.phase_cl(dataset, int(j["t"])) if pool == "cl" else D.phase_fx(dataset, int(j["t"])),
                     "cand": [int(x) for x in j["cand"]], "n_legal": int(j["n_legal"]), "gain": np.asarray(j["oracle_gain"], float),
                     "checks": j["checks"], "diag": j["diag"], "seconds": j.get("seconds"), "scores": {}}
                for src, sc in j["scores"].items():
                    r["scores"][src] = {}
                    for v, vals in sc.items():
                        a = np.asarray(vals, float)
                        r["scores"][src][v] = a.mean(1) if a.ndim == 2 else a          # K-mean for the models
                recs.append(r)
    # picks + metrics
    for r in recs:
        g = r["gain"][r["cand"]]
        r["metrics"], r["picks"], r["fallback"] = {}, {}, {}
        for src, sc in r["scores"].items():
            for v, s in sc.items():
                pk, fb = D.pick(r["dataset"], s, r, f"{src}|{v}")
                key = f"{src}|{v}"
                r["picks"][key], r["fallback"][key] = int(pk), bool(fb)
                r["metrics"][key] = D.rank_metrics(s[r["cand"]], g, r["cand"].index(pk))
        r["random"] = D.random_reference(g)
        r["gstar"] = float(g.max() * 100)
    recs.sort(key=lambda r: (r["dataset"], r["pool"], r["scene"], r["start"], r["episode"], r["t"]))
    return recs


def agg(recs, key, metric):
    return D.boot(D.hier_mean(recs, lambda r: (None if r["metrics"][key].get(metric) is None else float(r["metrics"][key][metric]))).values())


def scene_vals(recs, key, metric):
    return D.hier_mean(recs, lambda r: (None if r["metrics"][key].get(metric) is None else float(r["metrics"][key][metric])))


def table(recs, src, variants, title):
    L = [f"### {title}（{len(recs)} 状态）", "", "| 评分器 | Spearman(score, G) | Top-1（τ） | Pairwise | regret pp [CI] | 归一化 regret | Δregret vs S0 [CI]（场景配对；负 = 更好） | 场景 更好/更差 | fallback |", "|---|---|---|---|---|---|---|---|---|"]
    s0 = scene_vals(recs, f"{src}|S0", "regret")
    for v in variants:
        key = f"{src}|{v}"
        if any(key not in r["metrics"] for r in recs):
            continue
        rho, top1, pw, reg, nreg = (agg(recs, key, m) for m in ("rho", "top1", "pairwise", "regret", "nregret"))
        p = D.paired(scene_vals(recs, key, "regret"), s0) if v != "S0" else None
        fb = sum(r["fallback"][key] for r in recs)
        L.append(f"| {v} | {D.fmt(rho) if rho else '—'} | {D.fmt(top1)} | {D.fmt(pw) if pw else '—'} | {D.fmt(reg)} | {D.fmt(nreg) if nreg else '—'} | "
                 f"{D.fmt(p) if p else '—'} | {f'{p[chr(119)+chr(105)+chr(110)+chr(115)]} / {p[chr(108)+chr(111)+chr(115)+chr(115)+chr(101)+chr(115)]}' if p else '—'} | {fb} |")
    rr = D.boot(D.hier_mean(recs, lambda r: r["random"]["regret"]).values())
    L.append(f"| Random | — | {D.fmt(D.boot(D.hier_mean(recs, lambda r: r['random']['top1']).values()))} | — | {D.fmt(rr)} | — | — | — | — |")
    return "\n".join(L)


def phase_table(recs, src, variants):
    L = ["| 评分器 | early | mid | late |", "|---|---|---|---|"]
    for v in variants:
        key = f"{src}|{v}"
        if any(key not in r["metrics"] for r in recs):
            continue
        cells = []
        for ph in ("early", "mid", "late"):
            sub = [r for r in recs if r["phase"] == ph]
            cells.append((D.fmt(agg(sub, key, "regret")) + f" (n={len(sub)})") if sub else "—")
        L.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def per_scene_table(recs, src, variants):
    scenes = sorted({r["scene"] for r in recs})
    L = ["| 场景 | n | " + " | ".join(variants) + " |", "|---|---|" + "---|" * len(variants)]
    for s in scenes:
        sub = [r for r in recs if r["scene"] == s]
        vals = [scene_vals(sub, f"{src}|{v}", "regret").get(s) for v in variants]
        L.append(f"| {s} | {len(sub)} | " + " | ".join("—" if x is None else f"{x:.3f}" for x in vals) + " |")
    return "\n".join(L)


def calibration(recs, src, variants):
    """Pooled Spearman of score vs oracle gain over all (state, candidate) pairs and the mean within-state Spearman."""
    from scipy.stats import spearmanr
    out = {}
    for v in variants:
        key = f"{src}|{v}"
        xs, ys = [], []
        for r in recs:
            if src in r["scores"] and v in r["scores"][src]:
                xs.extend(r["scores"][src][v][r["cand"]].tolist()); ys.extend(r["gain"][r["cand"]].tolist())
        if len(xs) > 10:
            out[v] = {"pooled_rho": float(spearmanr(xs, ys).correlation), "n": len(xs)}
    return out


def gate(recs_dev, m, dataset):
    """Frozen G2 screening for model m on the dev set of a dataset -> {variant: {...}}; the GT condition uses the GT scores
    on the same dev states."""
    need = 6 if dataset == "indoor" else 2
    res = {}
    s0_gt = scene_vals(recs_dev, "GT|S0", "regret")
    s0_m = scene_vals(recs_dev, f"{m}|S0", "regret")
    for v in VARIANTS:
        if v == "S0":
            continue
        key = f"{m}|{v}"
        if any(key not in r["metrics"] for r in recs_dev):
            continue
        gt_v = scene_vals(recs_dev, f"GT|{v}", "regret")
        a_gt = float(np.mean([gt_v[s] - s0_gt[s] for s in s0_gt])) <= 0.0
        sv = scene_vals(recs_dev, key, "regret")
        p = D.paired(sv, s0_m)
        rel = p["mean"] / np.mean(list(s0_m.values())) if np.mean(list(s0_m.values())) > 0 else 0.0
        b = (p["hi"] < 0) or (rel <= -0.10 and p["wins"] >= need)
        worst = max((sv[s] - s0_m[s]) / s0_m[s] if s0_m[s] > 0 else 0.0 for s in s0_m)
        c = worst <= 0.25
        res[v] = {"gt_not_worse": bool(a_gt), "gt_delta": float(np.mean([gt_v[s] - s0_gt[s] for s in s0_gt])), "delta": p["mean"], "ci": [p["lo"], p["hi"]],
                  "rel": float(rel), "wins": p["wins"], "losses": p["losses"], "worst_rel": float(worst), "pass": bool(a_gt and b and c)}
    return res


def failure_cases(recs, src, best, dataset, n=3):
    rows = []
    for r in recs:
        m0, m1 = r["metrics"][f"{src}|S0"], r["metrics"][f"{src}|{best}"]
        kind = "S0 错 / 新对" if (not m0["top1"] and m1["top1"]) else ("S0 对 / 新错" if (m0["top1"] and not m1["top1"]) else ("皆错" if not m0["top1"] else None))
        if kind:
            rows.append((kind, r, m0, m1))
    L = ["| 类型 | 场景 | 起点 | t | 候选数 | G\\* pp | S0 regret | 新 regret | S0 选 / 新选 / oracle 最优 |", "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("S0 错 / 新对", "S0 对 / 新错", "皆错"):
        sub = sorted([x for x in rows if x[0] == kind], key=lambda x: -(x[2]["regret"] + x[3]["regret"]))[:n]
        for k, r, m0, m1 in sub:
            g = r["gain"]
            L.append(f"| {k} | {r['scene']} | {r['start']} | {r['t']} | {len(r['cand'])} | {r['gstar']:.3f} | {m0['regret']:.3f} | {m1['regret']:.3f} | "
                     f"{r['picks'][f'{src}|S0']} / {r['picks'][f'{src}|{best}']} / {int(np.argmax(g))} |")
    return "\n".join(L)


def main():
    for p in (OUT, FIG, REP):
        os.makedirs(p, exist_ok=True)
    recs = load()
    checks = {"GT_S0_eq_A": sum(r["checks"].get("GT_S0_eq_A", False) for r in recs), "B2_S0_eq_cache": sum(r["checks"].get("B2_S0_eq_cache", False) for r in recs),
              "T1_S0_eq_cache": sum(r["checks"].get("T1_S0_eq_cache", False) for r in recs), "n": len(recs),
              "GT_S0_maxrel": max(r["checks"].get("GT_S0_maxrel_A", 0) for r in recs), "B2_S0_maxrel": max(r["checks"].get("B2_S0_maxrel_cache", 0) for r in recs)}
    diag = {}
    for src in SOURCES:
        d = [x for r in recs for x in (r["diag"].get(src) or []) if x]
        diag[src] = {k: float(np.mean([x[k] for x in d if x.get(k) is not None])) for k in ("n_hit", "n_new0", "n_in_band", "n_behind_dense", "n_thin_front", "mean_cos", "mean_z_rt", "mean_novelty_lin", "frac_novel_hard", "n_hit_in_dense_empty") if any(x.get(k) is not None for x in d)}
    groups = {}
    for dataset in ("indoor", "outdoor"):
        for split in ("dev", "eval"):
            groups[(dataset, split)] = [r for r in recs if r["dataset"] == dataset and r["split"] == split]
    summary = {"checks": checks, "diag": diag, "n_states": {f"{k[0]}/{k[1]}": len(v) for k, v in groups.items()}, "tables": {}, "gate": {}, "calibration": {}}
    # ---- E1-GT report
    LG = ["# E1 — GT Depth Scoring（自动生成 `scripts/e1_analyze.py`；协议 §3）", "",
          f"复现检查：S0(t\\*) = Experiment A `gtd_distp1` 逐位 {checks['GT_S0_eq_A']} / {checks['n']}（最大相对差 {checks['GT_S0_maxrel']:.1e}）；S0(缓存样本) = 缓存 `distp1_k` 逐位 B2 {checks['B2_S0_eq_cache']} / {checks['n']}、T1 {checks['T1_S0_eq_cache']} / {checks['n']}。",
          "dev = MP3D VALSEL（Pool FX，9 楼层）/ Macarons++ Dev-3（Pool CL + FX）；eval = MP3D 5 评估楼层（Pool CL）/ Macarons++ Test-3（非盲）。均值 = 场景均值的平均，CI = 场景级 bootstrap 95%；regret 单位 pp；τ = 1e-4。全局变体 G_\\*（无区域裁剪）只在 GT 上。", ""]
    LG += ["## 命中统计（每候选均值；GT 与各模型）", "", "| 来源 | 命中射线 | S0 新命中 | 在 band_dil 内 | 稠密表面之后 | 薄前缘 (< 3 体素) | 命中在稠密 EMPTY | 平均 cosθ | 平均 z（运行） | 平均 N_lin | 硬新颖比例 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for src in SOURCES:
        x = diag[src]
        LG.append(f"| {LABEL[src]} | {x.get('n_hit', 0):.0f} | {x.get('n_new0', 0):.0f} | {x.get('n_in_band', 0):.0f} | {x.get('n_behind_dense', 0):.1f} | {x.get('n_thin_front', 0):.0f} | {x.get('n_hit_in_dense_empty', 0):.1f} | {x.get('mean_cos', 0):.3f} | {x.get('mean_z_rt', 0):.1f} | {x.get('mean_novelty_lin', 0):.3f} | {x.get('frac_novel_hard', 0):.3f} |")
    LG.append("")
    for dataset, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        for split in ("dev", "eval"):
            sub = groups[(dataset, split)]
            if not sub:
                continue
            LG += [f"## {nm} · {split}", "", table(sub, "GT", VARIANTS + GVARIANTS, f"GT depth · {nm} {split}"), "", "按相位（regret）：", "", phase_table(sub, "GT", ["S0", "S1a", "S1d", "S2", "S2S3", "S1aS2", "S1dS2", "G_soft", "G_area"]), "",
                   "逐场景 regret：", "", per_scene_table(sub, "GT", ["S0", "S1a", "S1b", "S1c", "S1d", "S2", "S2h", "S3", "S1aS2", "S1dS2", "S1aS2S3", "G_soft"]), ""]
            cal = calibration(sub, "GT", VARIANTS + GVARIANTS)
            summary["calibration"][f"GT/{dataset}/{split}"] = cal
            LG += ["校准（跨状态、跨候选的 pooled Spearman(score, gain)）：" + "，".join(f"{v} {c['pooled_rho']:.3f}" for v, c in cal.items()), ""]
            summary["tables"][f"GT/{dataset}/{split}"] = {v: {m: agg(sub, f"GT|{v}", m) for m in ("regret", "top1", "rho")} for v in VARIANTS + GVARIANTS if all(f"GT|{v}" in r["metrics"] for r in sub)}
    open(os.path.join(REP, "E1_GT_SCORING.md"), "w").write("\n".join(LG) + "\n")
    # ---- E1-Pred report + gate
    LM = ["# E1 — Frozen-Model Scoring（自动生成；B2 / T1 主，T0 / E2D 对照；同状态、同候选、同一缓存 K = 4 样本，K 均值）", "", "dev / eval 与 CI 同 `E1_GT_SCORING.md`。G2 Screening（冻结，dev 集，对每个 (模型, 数据集)）：(a) GT 阶段该变体 dev regret ≤ S0；(b) 预测阶段 Δregret CI 上界 < 0，或相对改善 ≥ 10% 且 MP3D ≥ 6/9 楼层、Macarons++ ≥ 2/3 场景改善；(c) 无 dev 场景相对恶化 > 25%。", ""]
    for dataset, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        for split in ("dev", "eval"):
            sub = groups[(dataset, split)]
            if not sub:
                continue
            LM.append(f"## {nm} · {split}")
            for src in ("B2", "T1", "T0", "E2D"):
                LM += ["", table(sub, src, VARIANTS, f"{LABEL[src]} · {nm} {split}"), "", "按相位：", "", phase_table(sub, src, ["S0", "S1a", "S1d", "S2", "S3", "S2S3", "S1aS2", "S1dS2"]), ""]
                summary["tables"][f"{src}/{dataset}/{split}"] = {v: {m: agg(sub, f"{src}|{v}", m) for m in ("regret", "top1", "rho")} for v in VARIANTS if all(f"{src}|{v}" in r["metrics"] for r in sub)}
            if split == "dev":
                LM += ["### G2 Screening（dev）", "", "| 模型 | 变体 | GT Δregret（≤ 0 ?） | 预测 Δregret [CI] | 相对 | 场景 更好/更差 | 最差场景相对变化 | 通过 |", "|---|---|---|---|---|---|---|---|"]
                for m in MAIN_MODELS:
                    g = gate(sub, m, dataset)
                    summary["gate"][f"{m}/{dataset}"] = g
                    for v, x in g.items():
                        LM.append(f"| {m} | {v} | {x['gt_delta']:+.3f} ({'✓' if x['gt_not_worse'] else '✗'}) | {x['delta']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] | {100 * x['rel']:+.1f}% | {x['wins']} / {x['losses']} | {100 * x['worst_rel']:+.1f}% | {'✅' if x['pass'] else '—'} |")
                LM.append("")
    # selection rule
    sel = {}
    for v in VARIANTS:
        if v == "S0":
            continue
        ok = all(summary["gate"].get(f"{m}/{d}", {}).get(v, {}).get("pass", False) for m in MAIN_MODELS for d in ("indoor", "outdoor"))
        rels = [summary["gate"][f"{m}/{d}"][v]["rel"] for m in MAIN_MODELS for d in ("indoor", "outdoor") if v in summary["gate"].get(f"{m}/{d}", {})]
        per_d = {d: all(summary["gate"].get(f"{m}/{d}", {}).get(v, {}).get("pass", False) for m in MAIN_MODELS) for d in ("indoor", "outdoor")}
        sel[v] = {"pass_all": ok, "pass_per_dataset": per_d, "mean_rel": float(np.mean(rels)) if rels else None}
    summary["selection"] = sel
    best_all = [v for v, x in sel.items() if x["pass_all"]]
    chosen = min(best_all, key=lambda v: sel[v]["mean_rel"]) if best_all else None
    per_ds = {d: [v for v, x in sel.items() if x["pass_per_dataset"][d]] for d in ("indoor", "outdoor")}
    summary["chosen"] = {"all": chosen, "per_dataset": {d: (min(vs, key=lambda v: np.mean([summary["gate"][f"{m}/{d}"][v]["rel"] for m in MAIN_MODELS])) if vs else None) for d, vs in per_ds.items()}}
    LM += ["## 选择规则结果", "", f"两个数据集、两个模型都通过的变体：{best_all or '无'} → G3 评分器：**{chosen or 'No-Go（无通过者）'}**；按数据集单独通过：MP3D {per_ds['indoor'] or '无'}，Macarons++ {per_ds['outdoor'] or '无'}。", ""]
    open(os.path.join(REP, "E1_MODEL_SCORING.md"), "w").write("\n".join(LM) + "\n")
    # ---- failure cases (eval + dev, GT and B2, best variant per dataset or S1dS2 as default)
    LF = ["# E1 — 失败案例（自动规则：各类按 regret 和排序前 3）", ""]
    for dataset, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        sub = [r for r in recs if r["dataset"] == dataset]
        best = summary["chosen"]["per_dataset"][dataset] or "S1dS2"
        for src in ("GT", "B2", "T1"):
            LF += [f"## {nm} · {LABEL[src]}：S0 vs {best}", "", failure_cases(sub, src, best, dataset), ""]
    open(os.path.join(REP, "E1_FAILURE_CASES.md"), "w").write("\n".join(LF) + "\n")
    # ---- csv + json
    with open(os.path.join(OUT, "e1_states.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        keys = [f"{src}|{v}" for src in SOURCES for v in VARIANTS] + [f"GT|{v}" for v in GVARIANTS]
        w.writerow(["pool", "dataset", "split", "scene", "start", "t", "phase", "sid", "n_cand", "gstar_pp"] + [f"regret:{k}" for k in keys] + [f"top1:{k}" for k in keys])
        for r in recs:
            w.writerow([r["pool"], r["dataset"], r["split"], r["scene"], r["start"], r["t"], r["phase"], r["sid"], len(r["cand"]), r["gstar"]] +
                       [r["metrics"][k]["regret"] if k in r["metrics"] else "" for k in keys] + [int(r["metrics"][k]["top1"]) if k in r["metrics"] else "" for k in keys])
    json.dump(summary, open(os.path.join(OUT, "e1_summary.json"), "w"), indent=1, default=float)
    # ---- figures: regret bars per variant (GT, B2, T1) per dataset / split
    fig, axs = plt.subplots(2, 2, figsize=(18, 9))
    for i, dataset in enumerate(("indoor", "outdoor")):
        for j, split in enumerate(("dev", "eval")):
            ax = axs[i, j]
            sub = groups[(dataset, split)]
            vs = [v for v in VARIANTS if all(f"GT|{v}" in r["metrics"] for r in sub)]
            x = np.arange(len(vs))
            for k, (src, col) in enumerate((("GT", "#d62728"), ("B2", "#1c5cab"), ("T1", "#8e44ad"))):
                vals = [agg(sub, f"{src}|{v}", "regret") for v in vs]
                ax.bar(x + (k - 1) * 0.27, [b["mean"] for b in vals], 0.27, color=col, label=LABEL[src], yerr=[[b["mean"] - b["lo"] for b in vals], [b["hi"] - b["mean"] for b in vals]], capsize=2)
            ax.set_xticks(x); ax.set_xticklabels(vs, rotation=60, fontsize=8); ax.set_ylabel("one-step regret (pp)"); ax.set_title(f"{dataset} · {split} ({len(sub)} states)", loc="left")
            ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Task 2 / E1: one-step oracle regret of the scorer variants on the fixed states (scene-level mean, 95% CI); measured", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "e1_regret_bars.png"), dpi=130); plt.close(fig)
    print(json.dumps({"checks": checks, "chosen": summary["chosen"]}, default=float))
    for d in ("indoor", "outdoor"):
        for src in ("GT", "B2", "T1"):
            t = summary["tables"].get(f"{src}/{d}/dev", {})
            print(d, src, "dev regret:", {v: round(t[v]["regret"]["mean"], 3) for v in t if t[v]["regret"]})


if __name__ == "__main__":
    main()
