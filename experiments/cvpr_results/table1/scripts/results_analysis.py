"""CVPR Table 1 -- reports/RESULTS_ANALYSIS.md + results/analysis_pairs[_mono].csv (CPU; generated, do not edit by hand).
Answers the Table 1 questions with numbers computed from the canonical episode records:
  Q1 early speed vs MAG-B (dCov@10, dCov@20, dAUC@20)   Q2 AUC@20 and AUC@100 both improve?
  Q3 train-good / non-train-bad (per-split deltas vs MAG-B; Train minus Dev+Test gap)
  Q4 B2 on full scenes (B2 vs MAG-B / B0 / B1)          Q5 V3 / V4 / V5 vs B2 and the single-factor pairs
  Q6 scene-specific gains (per-scene dAUC@100 / dAUC@20 vs MAG-B)   Q7 statistics (scene bootstrap CI, scene sign test)
Paired difference A - B: per scene, mean over the 5 starts of (A_k - B_k); only scenes where both are complete; scene-level
bootstrap 95% CI (10k, seed 0); wins / losses = scenes with mean diff > 0 / < 0; two-sided exact sign test over scenes.
Groups: train, dev, test, devtest (= Dev+Test, the 8 non-training scenes), all14. Primary = official; mono = appendix."""
import argparse
import csv
import glob
import json
import math
import os

import numpy as np

import aggregate as AG

T1 = AG.T1
GROUPS = ("train", "dev", "test", "devtest", "all14")
GN = {"train": "Train-6", "dev": "Dev-5", "test": "Test-3", "devtest": "Dev+Test（8 个非训练场景）", "all14": "All-14"}
METS = ("cov10", "cov20", "auc20", "auc50", "auc100", "final")
MN = {"cov10": "ΔCov@10", "cov20": "ΔCov@20", "auc20": "ΔAUC@20", "auc50": "ΔAUC@50", "auc100": "ΔAUC@100", "final": "ΔFinal"}
MAIN = ("V2O", "B0", "B2", "E2D", "V3", "V4", "V5")
PAIRS = [("B2", "B0", "同一权重：D100 推理 vs S10 推理（评分都是 S10）"), ("B1", "B2", "同一权重：D100 评分 vs S10 评分（推理都是 D100）"),
         ("MAG-B", "MAG-G", "MAGICIAN：Beam 10×10 vs Greedy"), ("B0", "V2O", "17.5k 混合微调 vs V2 原版"),
         ("E2D", "B2", "额外 D100 混合微调 5k（E2D）vs B2"), ("V4C", "V4", "V4 多训练 12k"), ("V5", "V4C", "同父模型、同预算：V5 解码器与多表面监督 vs V4C（复合干预）"),
         ("V3", "B2", "系统级：V3-T1 vs B2"), ("V4", "B2", "系统级：V4 vs B2"), ("V5", "B2", "系统级：V5 vs B2"), ("E2D", "B0", "系统级：E2D vs B0")]


def load(protocol):
    key = "coverage_official" if protocol == "official" else "coverage_mono"
    split, order, label, _ = AG.load_registries()
    M = {}
    for f in glob.glob(os.path.join(AG.EPI, "*", "*.json")):
        r = json.load(open(f))
        M.setdefault(r["model"], {}).setdefault(r["scene"], {})[r["start"]] = AG.metrics(r[key])
    return split, order, label, M


def scenes_of(split, g):
    if g == "all14":
        return sorted(AG.RUNNABLE)
    if g == "devtest":
        return sorted(s for s in split if split[s] in ("dev", "test"))
    return sorted(s for s in split if split[s] == g)


def complete(M, m, s):
    return all(k in M.get(m, {}).get(s, {}) for k in range(AG.N_START))


def sign_p(w, l):
    n = w + l
    if n == 0:
        return None
    k = min(w, l)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(M, split, a, b, g, met):
    sc = scenes_of(split, g)
    ok = [s for s in sc if complete(M, a, s) and complete(M, b, s)]
    if len(ok) != len(sc):
        return None
    d = {s: float(np.mean([M[a][s][k][met] - M[b][s][k][met] for k in range(AG.N_START)])) for s in ok}
    v = list(d.values())
    w, l = sum(x > 0 for x in v), sum(x < 0 for x in v)
    return {"a": a, "b": b, "group": g, "metric": met, "n_scenes": len(ok), "d": float(np.mean(v)), "ci": AG.boot(v), "wins": w, "losses": l,
            "sign_p": sign_p(w, l), "per_scene": d}


def level(M, split, m, g, met):
    sc = scenes_of(split, g)
    if not all(complete(M, m, s) for s in sc):
        return None
    return float(np.mean([np.mean([M[m][s][k][met] for k in range(AG.N_START)]) for s in sc]))


def fd(r, sig=True):
    if r is None:
        return "—（不完整）"
    star = " **✱**" if sig and r["ci"][0] is not None and (r["ci"][0] > 0 or r["ci"][1] < 0) else ""
    return f"{r['d']:+.3f} [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}] {r['wins']}/{r['losses']}{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="official", choices=["official", "mono"])
    a = ap.parse_args()
    split, order, label, M = load(a.protocol)
    present = [m for m in order if m in M]
    allrows = []
    out = [f"# RESULTS_ANALYSIS — Table 1 结果分析（{'主口径：MAGICIAN 官方覆盖' if a.protocol == 'official' else '辅口径：单调覆盖'}；自动生成，`scripts/results_analysis.py`）", "",
           "**读法**：",
           "- 每格为 `Δ [95% CI] 胜/负场景数`，Δ 为 A − B；",
           "- 每个场景先对 5 个起点的配对差取均值，再在场景层面 bootstrap（10k，seed 0）；",
           "- **✱** 表示 CI 不含 0；",
           "- `—（不完整）` 表示该组有场景缺 episode，不计算。", "",
           "**场景性质**：",
           "- Train-6 是我们模型训练时见过的场景（起点 0–3 在训练轨迹中）；",
           "- Test-3 已不是盲测；",
           "- Dev+Test 是 8 个非训练场景，由本脚本派生，不是协议中的独立划分。", ""]
    # levels
    out += ["## 0. 绝对值（场景等权均值）", "", "| 模型 | " + " | ".join(f"{GN[g]} AUC@20 / AUC@100 / Final" for g in ("train", "devtest", "all14")) + " |",
            "|---|" + "---|" * 3]
    for m in present:
        cells = []
        for g in ("train", "devtest", "all14"):
            v = [level(M, split, m, g, k) for k in ("auc20", "auc100", "final")]
            cells.append("—" if any(x is None for x in v) else " / ".join(f"{x:.3f}" for x in v))
        out.append(f"| {label[m]} | " + " | ".join(cells) + " |")
    # Q1 / Q2
    for title, mets in (("## Q1. 前 20 步是否比 MAGICIAN Beam（MAG-B）覆盖更快？", ("cov10", "cov20", "auc20")),
                        ("## Q2. AUC@20 与 AUC@100 是否同时提升？（含 Final）", ("auc20", "auc100", "final"))):
        out += ["", title, ""]
        for g in ("all14", "train", "devtest", "dev", "test"):
            out += [f"**{GN[g]}**", "", "| 模型 − MAG-B | " + " | ".join(MN[k] for k in mets) + " |", "|---|" + "---|" * len(mets)]
            for m in present:
                if m == "MAG-B":
                    continue
                rs = [paired(M, split, m, "MAG-B", g, k) for k in mets]
                allrows += [r for r in rs if r]
                out.append(f"| {label[m]} | " + " | ".join(fd(r) for r in rs) + " |")
            out.append("")
    # Q3
    out += ["## Q3. 是否存在训练场景好、非训练场景差的模型？", "",
            "- 下表给出每个模型相对 MAG-B 的 ΔAUC@100 与 ΔAUC@20，分 Train-6 与 Dev+Test 两组；",
            "- 最后一列为 (Train Δ) − (Dev+Test Δ)，正值越大，表示优势越集中在训练场景。", "",
            "| 模型 | Train ΔAUC@20 | Dev+Test ΔAUC@20 | Train ΔAUC@100 | Dev+Test ΔAUC@100 | Train − DevTest（AUC@100） |", "|---|---|---|---|---|---|"]
    for m in present:
        if m == "MAG-B":
            continue
        t20, d20 = paired(M, split, m, "MAG-B", "train", "auc20"), paired(M, split, m, "MAG-B", "devtest", "auc20")
        t100, d100 = paired(M, split, m, "MAG-B", "train", "auc100"), paired(M, split, m, "MAG-B", "devtest", "auc100")
        gap = f"{t100['d'] - d100['d']:+.3f}" if t100 and d100 else "—"
        out.append(f"| {label[m]} | {fd(t20)} | {fd(d20)} | {fd(t100)} | {fd(d100)} | {gap} |")
    # Q4 / Q5 pairs
    out += ["", "## Q4 / Q5. 单因素配对与系统级配对", "",
            "- B2 的优势是否在全场景成立：见 B2 − B0、B1 − B2 两行，以及 Q1 / Q2 中 B2 − MAG-B 的结果；",
            "- V3 / V4 / V5 是否带来真实系统收益：见对 B2 的系统级配对，以及 V4C − V4、V5 − V4C 两个单因素配对。", ""]
    for g in ("all14", "train", "devtest"):
        out += [f"**{GN[g]}**", "", "| A − B | 说明 | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|"]
        for x, y, why in PAIRS:
            if x not in M or y not in M:
                continue
            rs = [paired(M, split, x, y, g, k) for k in ("auc20", "auc100", "final")]
            allrows += [r for r in rs if r]
            out.append(f"| {x} − {y} | {why} | " + " | ".join(fd(r) for r in rs) + " |")
        out.append("")
    # Q6
    out += ["## Q6. 场景层面：哪些场景有收益？（主模型 − MAG-B 的 ΔAUC@100 / ΔAUC@20，5 个起点均值）", ""]
    for met in ("auc100", "auc20"):
        mm = [m for m in MAIN if m in M]
        out += [f"**{MN[met]}**", "", "| 场景 | 划分 | " + " | ".join(mm) + " |", "|---|---|" + "---|" * len(mm)]
        for s in sorted(AG.RUNNABLE, key=lambda s: (["train", "dev", "test"].index(split[s]), s)):
            cells = []
            for m in mm:
                if complete(M, m, s) and complete(M, "MAG-B", s):
                    cells.append(f"{np.mean([M[m][s][k][met] - M['MAG-B'][s][k][met] for k in range(AG.N_START)]):+.3f}")
                else:
                    cells.append("—")
            out.append(f"| {s} | {split[s]} | " + " | ".join(cells) + " |")
        out.append("")
    # Q7
    out += ["## Q7. 统计证据（All-14，模型 − MAG-B；场景级 bootstrap CI 与场景符号检验）", "",
            "| 模型 | ΔAUC@20 | 符号检验 p | ΔAUC@100 | 符号检验 p | ΔFinal | 符号检验 p |", "|---|---|---|---|---|---|---|"]
    for m in present:
        if m == "MAG-B":
            continue
        rs = [paired(M, split, m, "MAG-B", "all14", k) for k in ("auc20", "auc100", "final")]
        sp = ["—" if not r or r["sign_p"] is None else f"{r['sign_p']:.3g}" for r in rs]
        out.append(f"| {label[m]} | " + " | ".join(f"{fd(r)} | {p}" for r, p in zip(rs, sp)) + " |")
    out += ["", "- 场景只有 14 个，场景级 CI 较宽；**只有 CI 不含 0 才称显著**。", "- 未对多重比较做校正，13 个模型 × 多个指标的显著结果应保守解读。"]
    suf = "" if a.protocol == "official" else "_mono"
    with open(os.path.join(T1, "results", f"analysis_pairs{suf}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "group", "metric", "n_scenes", "d", "ci_lo", "ci_hi", "wins", "losses", "sign_p"])
        seen = set()
        for r in allrows:
            k = (r["a"], r["b"], r["group"], r["metric"])
            if k in seen:
                continue
            seen.add(k)
            w.writerow([r["a"], r["b"], r["group"], r["metric"], r["n_scenes"], f"{r['d']:.6f}", f"{r['ci'][0]:.6f}", f"{r['ci'][1]:.6f}", r["wins"], r["losses"],
                        "" if r["sign_p"] is None else f"{r['sign_p']:.6g}"])
    open(os.path.join(T1, "reports", f"RESULTS_ANALYSIS{suf}.md"), "w").write("\n".join(out) + "\n")
    print(f"written reports/RESULTS_ANALYSIS{suf}.md ({len(out)} lines)")


if __name__ == "__main__":
    main()
