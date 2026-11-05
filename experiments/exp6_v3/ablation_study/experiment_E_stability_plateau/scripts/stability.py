"""EXP6 V3 / ablation E, Task 1 -- D1 vs YBEST-H3 stability analysis (CPU; read-only CVPR Table 1 canonical records, official
coverage; nothing is re-run here).
Metrics per episode: AUC@20 / AUC@50 / AUC@100 (trapz(C[0..k]) / k, as Table 1), Cov@20, Cov@50, Final = C[100].
Scene = mean of its 5 starts; split = scene-equal mean over scenes complete for every model compared (else partial).
Splits: Train-6, Dev-5, Test-3, All-14 (Table 1) and the derived Unseen-8 = Dev-5 + Test-3 (not a Table 1 split).
Paired A - B: scene-level paired differences, scene-level bootstrap 95% CI (10k, seed 0), scene wins / losses (|d| < 1e-9 =
tie), episode-level (same start) wins / losses; concentration: leave-one-scene-out range of the mean, mean without the two
scenes with the largest |d|, share of the summed positive part from the top scene.
Degradation events: episodes whose Final is more than 0.05 below the SAME-start H0 (V3-T1 40K, frozen DistP1) episode.
2 x 2: {40K, 50K} x {lambda 0.25, lambda 0} = {V3-H3, V3-YBEST-H3} x {V3-D1-H3-NORISK, V3-YD-COMBINED-Y50K-D1}: fine-tune effect
(mean over lambda of 50K - 40K), risk-removal effect (mean over checkpoints of lambda 0 - lambda 0.25), interaction, on the
scenes complete for all four cells, joint scene bootstrap.
Outputs: results/stability_episodes.csv, results/stability_summary.json, results/STABILITY_TABLES.md,
FIGURES/stability_scene_diffs.{png,pdf}, FIGURES/stability_per_start_final.{png,pdf}."""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EX = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(EX))))
EP = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
SPLIT = {"train": ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"], "dev": ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"],
         "test": ["alhambra", "bridge", "neuschwanstein"]}
SPLIT["unseen8"] = SPLIT["dev"] + SPLIT["test"]
SPLIT["all14"] = sorted(SPLIT["train"] + SPLIT["dev"] + SPLIT["test"])
ORDER = ["train", "dev", "test", "unseen8", "all14"]
NAME = {"train": "Train-6", "dev": "Dev-5", "test": "Test-3", "unseen8": "Unseen-8 (Dev+Test，派生)", "all14": "All-14"}
MODELS = {"D1": "V3-D1-H3-NORISK", "YBEST-H3": "V3-YBEST-H3", "H3": "V3-H3", "YD (50K+λ0)": "V3-YD-COMBINED-Y50K-D1",
          "H0 (40K)": "V3", "Y50K+H0": "V3-Y50K-H0", "B0": "B0", "B2": "B2", "MAG-B": "MAG-B"}
MET = ("auc20", "auc50", "auc100", "cov20", "cov50", "final")
MLAB = {"auc20": "AUC@20", "auc50": "AUC@50", "auc100": "AUC@100", "cov20": "Cov@20", "cov50": "Cov@50", "final": "Final"}


def met(c):
    c = np.asarray(c, float)
    return {"auc20": float(np.trapz(c[:21]) / 20), "auc50": float(np.trapz(c[:51]) / 50), "auc100": float(np.trapz(c) / 100),
            "cov20": float(c[20]), "cov50": float(c[50]), "final": float(c[100])}


def load():
    E = {}
    for mid in MODELS.values():
        for f in glob.glob(os.path.join(EP, mid, "*.json")):
            r = json.load(open(f))
            E[(mid, r["scene"], int(r["start"]))] = met(r["coverage_official"])
    return E


def complete(E, mid, s):
    return all((mid, s, k) in E for k in range(5))


def smean(E, mid, s):
    return {q: float(np.mean([E[(mid, s, k)][q] for k in range(5)])) for q in MET}


def boot_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    b = v[np.random.default_rng(seed).integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def paired(E, a, b, scenes):
    sc = [s for s in scenes if complete(E, a, s) and complete(E, b, s)]
    out = {"n_scenes": len(sc), "of": len(scenes), "scenes": sc}
    for q in MET:
        d = np.asarray([smean(E, a, s)[q] - smean(E, b, s)[q] for s in sc])
        de = np.asarray([E[(a, s, k)][q] - E[(b, s, k)][q] for s in sc for k in range(5)])
        r = {"d": float(d.mean()) if len(d) else None, "ci": boot_ci(d), "wins": int((d > 1e-9).sum()), "losses": int((d < -1e-9).sum()),
             "ep_wins": int((de > 1e-9).sum()), "ep_losses": int((de < -1e-9).sum()), "ep_n": int(len(de)),
             "per_scene": {s: float(x) for s, x in zip(sc, d)}}
        if len(d) >= 3:
            loso = [float(np.delete(d, i).mean()) for i in range(len(d))]
            top2 = np.argsort(-np.abs(d))[:2]
            pos = d[d > 0].sum()
            r.update({"loso_min": min(loso), "loso_max": max(loso), "loso_sign_flip": bool(min(loso) < 0 < max(loso)) if r["d"] else None,
                      "without_top2_absd": float(np.delete(d, top2).mean()), "top2_scenes": [sc[i] for i in top2],
                      "median": float(np.median(d)), "top1_share_of_positive": float(d.max() / pos) if pos > 0 else None})
        out[q] = r
    return out


def degradation(E, mid, scenes, ref="V3", thr=0.05):
    ev = [(s, k, E[(mid, s, k)]["final"] - E[(ref, s, k)]["final"]) for s in scenes for k in range(5) if (mid, s, k) in E and (ref, s, k) in E]
    bad = [(s, k, round(d, 3)) for s, k, d in ev if d < -thr]
    return {"n": len(ev), "events": len(bad), "list": bad, "mean_drop_of_events": float(np.mean([d for *_, d in bad])) if bad else None}


def dispersion(E, mid, scenes):
    rows = [E[(mid, s, k)] for s in scenes for k in range(5) if (mid, s, k) in E]
    if not rows:
        return None
    within = [float(np.std([E[(mid, s, k)]["final"] for k in range(5)])) for s in scenes if complete(E, mid, s)]
    f = np.asarray([r["final"] for r in rows])
    return {"n": len(rows), "final_p10": float(np.quantile(f, 0.1)), "final_min": float(f.min()), "final_within_scene_std_mean": float(np.mean(within)) if within else None,
            "auc100_p10": float(np.quantile([r["auc100"] for r in rows], 0.1))}


def factorial(E, scenes):
    cell = {("40K", "l025"): "V3-H3", ("40K", "l0"): "V3-D1-H3-NORISK", ("50K", "l025"): "V3-YBEST-H3", ("50K", "l0"): "V3-YD-COMBINED-Y50K-D1"}
    sc = [s for s in scenes if all(complete(E, m, s) for m in cell.values())]
    out = {"n_scenes": len(sc), "of": len(scenes), "scenes": sc}
    if len(sc) < 2:
        return out
    for q in MET:
        M = {c: np.asarray([smean(E, m, s)[q] for s in sc]) for c, m in cell.items()}
        eff = {"ft": ((M[("50K", "l025")] - M[("40K", "l025")]) + (M[("50K", "l0")] - M[("40K", "l0")])) / 2,
               "risk_removal": ((M[("40K", "l0")] - M[("40K", "l025")]) + (M[("50K", "l0")] - M[("50K", "l025")])) / 2,
               "interaction": (M[("50K", "l0")] - M[("50K", "l025")]) - (M[("40K", "l0")] - M[("40K", "l025")]),
               "ft_at_l025": M[("50K", "l025")] - M[("40K", "l025")], "ft_at_l0": M[("50K", "l0")] - M[("40K", "l0")],
               "risk_at_40K": M[("40K", "l0")] - M[("40K", "l025")], "risk_at_50K": M[("50K", "l0")] - M[("50K", "l025")]}
        out[q] = {k: {"d": float(v.mean()), "ci": boot_ci(v), "wins": int((v > 1e-9).sum()), "losses": int((v < -1e-9).sum())} for k, v in eff.items()}
        out[q]["cells"] = {f"{c[0]}/{c[1]}": float(v.mean()) for c, v in M.items()}
    return out


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def fd(r):
    if r["d"] is None:
        return "—"
    ci = r["ci"]
    star = " ✱" if ci[0] is not None and (ci[0] > 0 or ci[1] < 0) else ""
    return f"{r['d']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] {r['wins']}/{r['losses']}{star}" if ci[0] is not None else f"{r['d']:+.3f} {r['wins']}/{r['losses']}"


def main():
    E = load()
    os.makedirs(os.path.join(EX, "results"), exist_ok=True)
    os.makedirs(os.path.join(EX, "FIGURES"), exist_ok=True)
    with open(os.path.join(EX, "results", "stability_episodes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "scene", "start", "split"] + list(MET))
        for (m, s, k), r in sorted(E.items()):
            sp = next(x for x in ("train", "dev", "test") if s in SPLIT[x])
            w.writerow([m, s, k, sp] + [f"{r[q]:.6f}" for q in MET])
    S = {"means": {}, "pairs": {}, "degradation": {}, "dispersion": {}, "factorial": {}}
    for nm, mid in MODELS.items():
        S["means"][nm] = {}
        for sp in ORDER:
            sc = [s for s in SPLIT[sp] if complete(E, mid, s)]
            S["means"][nm][sp] = {"n_scenes": len(sc), "of": len(SPLIT[sp]), **({q: float(np.mean([smean(E, mid, s)[q] for s in sc])) for q in MET} if sc else {})}
    PAIRS = [("V3-D1-H3-NORISK", "V3-YBEST-H3"), ("V3-D1-H3-NORISK", "V3-H3"), ("V3-YBEST-H3", "V3-H3"), ("V3-D1-H3-NORISK", "V3"),
             ("V3-YBEST-H3", "V3"), ("V3-Y50K-H0", "V3"), ("V3-YD-COMBINED-Y50K-D1", "V3-D1-H3-NORISK"), ("V3-YD-COMBINED-Y50K-D1", "V3-YBEST-H3"),
             ("V3-D1-H3-NORISK", "MAG-B"), ("V3-YBEST-H3", "MAG-B"), ("V3-D1-H3-NORISK", "B2"), ("V3-YBEST-H3", "B2")]
    for a, b in PAIRS:
        for sp in ORDER:
            S["pairs"][f"{a} − {b} | {sp}"] = paired(E, a, b, SPLIT[sp])
    for nm, mid in MODELS.items():
        S["degradation"][nm] = {sp: degradation(E, mid, SPLIT[sp]) for sp in ("all14", "unseen8")}
        S["dispersion"][nm] = {sp: dispersion(E, mid, SPLIT[sp]) for sp in ("all14", "unseen8")}
    for sp in ORDER:
        S["factorial"][sp] = factorial(E, SPLIT[sp])
    json.dump(S, open(os.path.join(EX, "results", "stability_summary.json"), "w"), indent=1)

    L = ["# STABILITY_TABLES — Task 1 自动生成表（`scripts/stability.py`；官方覆盖，Table 1 规范记录，未重跑）", "",
         "- 场景 = 5 个起点的均值；分组 = 场景等权均值；partial = 该分组只有部分场景完整。",
         "- ✱ = 场景级 bootstrap 95% CI 不含 0；胜/负 = 场景数；episode 胜/负 = 同起点配对。",
         "- Unseen-8 = Dev-5 + Test-3，是本分析的派生分组，不是 Table 1 分组。", "", "## 1. 各分组均值"]
    for sp in ORDER:
        L += ["", f"### {NAME[sp]}", "", "| 模型 | 场景 | " + " | ".join(MLAB[q] for q in MET) + " |", "|---|---|" + "---|" * len(MET)]
        for nm in MODELS:
            r = S["means"][nm][sp]
            tag = f"{r['n_scenes']}/{r['of']}" + ("" if r["n_scenes"] == r["of"] else "（partial）")
            L.append(f"| {nm} | {tag} | " + " | ".join(f3(r.get(q)) for q in MET) + " |")
    L += ["", "## 2. 配对差（A − B）：均值 [95% CI] 场景胜/负"]
    for key in PAIRS:
        a, b = key
        L += ["", f"### {a} − {b}", "", "| 分组 | 场景 | " + " | ".join(MLAB[q] for q in MET) + " | episode 胜/负（AUC@100 / Final） |", "|---|---|" + "---|" * len(MET) + "---|"]
        for sp in ORDER:
            p = S["pairs"][f"{a} − {b} | {sp}"]
            if p["n_scenes"] == 0:
                L.append(f"| {NAME[sp]} | 0/{p['of']} | " + " | ".join("—" for _ in MET) + " | — |")
                continue
            L.append(f"| {NAME[sp]} | {p['n_scenes']}/{p['of']} | " + " | ".join(fd(p[q]) for q in MET) +
                     f" | {p['auc100']['ep_wins']}/{p['auc100']['ep_losses']}；{p['final']['ep_wins']}/{p['final']['ep_losses']}（共 {p['final']['ep_n']}） |")
    p = S["pairs"]["V3-D1-H3-NORISK − V3-YBEST-H3 | all14"]
    L += ["", "## 3. D1 − YBEST-H3 逐场景差（All-14；场景均值差，正 = D1 更好）", "", "| 场景 | 分组 | " + " | ".join(MLAB[q] for q in MET) + " |", "|---|---|" + "---|" * len(MET)]
    for s in p["scenes"]:
        sp = next(x for x in ("train", "dev", "test") if s in SPLIT[x])
        L.append(f"| {s} | {sp} | " + " | ".join(f"{p[q]['per_scene'][s]:+.3f}" for q in MET) + " |")
    L += ["", "## 4. 逐起点差（D1 − YBEST-H3；每格为起点 0–4 的 AUC@100 差 / Final 差）", "", "| 场景 | s0 | s1 | s2 | s3 | s4 |", "|---|---|---|---|---|---|"]
    for s in p["scenes"]:
        L.append(f"| {s} | " + " | ".join(f"{E[('V3-D1-H3-NORISK', s, k)]['auc100'] - E[('V3-YBEST-H3', s, k)]['auc100']:+.3f} / {E[('V3-D1-H3-NORISK', s, k)]['final'] - E[('V3-YBEST-H3', s, k)]['final']:+.3f}" for k in range(5)) + " |")
    L += ["", "## 5. 集中度（优势是否由少数场景驱动）", "", "| 对比 | 分组 | 指标 | 均值 | 中位数 | 去掉一个场景后均值范围 | 去掉 |d| 最大两个场景后的均值（这两个场景） |", "|---|---|---|---|---|---|---|"]
    for key in (("V3-D1-H3-NORISK", "V3-YBEST-H3"), ("V3-D1-H3-NORISK", "V3-H3"), ("V3-YBEST-H3", "V3-H3"), ("V3-D1-H3-NORISK", "V3"), ("V3-YBEST-H3", "V3")):
        for sp in ("all14", "unseen8"):
            pp = S["pairs"][f"{key[0]} − {key[1]} | {sp}"]
            for q in ("auc20", "auc100", "final"):
                r = pp[q]
                if "loso_min" in r:
                    L.append(f"| {key[0]} − {key[1]} | {NAME[sp]} | {MLAB[q]} | {r['d']:+.4f} | {r['median']:+.4f} | [{r['loso_min']:+.4f}, {r['loso_max']:+.4f}]{' 符号会翻转' if r['loso_sign_flip'] else ''} | {r['without_top2_absd']:+.4f}（{', '.join(r['top2_scenes'])}） |")
    L += ["", "## 6. Final 退化事件与离散度", "", "- 退化事件：同起点 Final 比 H0（V3-T1 40K）低 0.05 以上。", "",
          "| 模型 | All-14 退化事件 / episode | Unseen-8 退化事件 | All-14 Final P10 | All-14 Final 最小 | 场景内起点间 Final 标准差（均值） | AUC@100 P10 |", "|---|---|---|---|---|---|---|"]
    for nm in MODELS:
        dg, ds = S["degradation"][nm], S["dispersion"][nm]
        if not ds["all14"]:
            continue
        L.append(f"| {nm} | {dg['all14']['events']}/{dg['all14']['n']} | {dg['unseen8']['events']}/{dg['unseen8']['n']} | {ds['all14']['final_p10']:.3f} | {ds['all14']['final_min']:.3f} | "
                 f"{f3(ds['all14']['final_within_scene_std_mean'])} | {ds['all14']['auc100_p10']:.3f} |")
    L += ["", "退化事件明细（All-14）：", ""]
    for nm in ("D1", "YBEST-H3", "H3", "YD (50K+λ0)"):
        L.append(f"- {nm}：" + ("；".join(f"{s} s{k} {d:+.3f}" for s, k, d in S["degradation"][nm]["all14"]["list"]) or "无"))
    L += ["", "## 7. 2 × 2：{40K, 50K} × {λ 0.25, λ 0}", "",
          "- 格子：H3 = 40K/λ0.25，D1 = 40K/λ0，YBEST-H3 = 50K/λ0.25，YD = 50K/λ0。",
          "- 微调效应 = 两个 λ 下 (50K − 40K) 的均值；去风险效应 = 两个权重下 (λ0 − λ0.25) 的均值；交互 = (YD − YBEST) − (D1 − H3)。", ""]
    for sp in ORDER:
        fa = S["factorial"][sp]
        L += [f"### {NAME[sp]}（完整场景 {fa['n_scenes']}/{fa['of']}{'' if fa['n_scenes'] == fa['of'] else '，partial'}）", ""]
        if fa["n_scenes"] < 2:
            L += ["数据不足。", ""]
            continue
        L += ["| 效应 | " + " | ".join(MLAB[q] for q in MET) + " |", "|---|" + "---|" * len(MET)]
        for e, lab in (("ft", "微调（平均）"), ("ft_at_l025", "微调 @λ0.25（YBEST − H3）"), ("ft_at_l0", "微调 @λ0（YD − D1）"), ("risk_removal", "去风险（平均）"),
                       ("risk_at_40K", "去风险 @40K（D1 − H3）"), ("risk_at_50K", "去风险 @50K（YD − YBEST）"), ("interaction", "交互")):
            L.append(f"| {lab} | " + " | ".join(fd(fa[q][e]) for q in MET) + " |")
        L.append("")
    open(os.path.join(EX, "results", "STABILITY_TABLES.md"), "w").write("\n".join(L) + "\n")

    # figures
    p = S["pairs"]["V3-D1-H3-NORISK − V3-YBEST-H3 | all14"]
    sc = p["scenes"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for i, q in enumerate(("auc20", "auc100", "final")):
        y = np.arange(len(sc))
        d = [p[q]["per_scene"][s] for s in sc]
        ax[i].barh(y, d, color=["#2f6db5" if x > 0 else "#c0504d" for x in d], height=0.6)
        for j, s in enumerate(sc):
            ax[i].scatter([E[("V3-D1-H3-NORISK", s, k)][q] - E[("V3-YBEST-H3", s, k)][q] for k in range(5)], [j] * 5, s=10, color="#333", zorder=3)
        ax[i].axvline(0, color="#888", lw=0.8)
        ax[i].set_title(f"{MLAB[q]}: D1 − YBEST-H3 = {p[q]['d']:+.3f}")
        ax[i].set_yticks(y)
        ax[i].set_yticklabels([f"{s} ({next(x for x in ('train', 'dev', 'test') if s in SPLIT[x])})" for s in sc], fontsize=8)
        ax[i].grid(axis="x", alpha=0.3)
    fig.suptitle("Per-scene difference (bar = scene mean, dots = the 5 starts); positive = D1 better", fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(EX, "FIGURES", f"stability_scene_diffs.{ext}"), dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    mods = [("V3", "H0 40K", "#888888"), ("V3-H3", "H3", "#e67e22"), ("V3-D1-H3-NORISK", "D1", "#2f6db5"), ("V3-YBEST-H3", "YBEST-H3", "#a61e4d")]
    for i, s in enumerate(SPLIT["all14"]):
        for j, (mid, lab, col) in enumerate(mods):
            v = [E[(mid, s, k)]["final"] for k in range(5) if (mid, s, k) in E]
            ax.scatter([i + (j - 1.5) * 0.18] * len(v), v, s=12, color=col, label=lab if i == 0 else None)
    ax.set_xticks(range(14))
    ax.set_xticklabels(SPLIT["all14"], rotation=30, fontsize=8)
    ax.set_ylabel("Final coverage (official)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(alpha=0.3)
    ax.set_title("Per-start Final coverage, All-14")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(EX, "FIGURES", f"stability_per_start_final.{ext}"), dpi=160)
    plt.close(fig)
    print("ok", len(E))


if __name__ == "__main__":
    main()
