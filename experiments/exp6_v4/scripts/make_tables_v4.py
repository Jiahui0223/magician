"""Tables A-D of the first stage (brief §8.3) from reports/g3_v4<suffix>.json (+ g12_v4_<tag>.json when present) ->
reports/V4_COMPARISON_TABLES.md (and exp6_v3.1/reports/V3_1_COMPARISON_TABLES.md as a pointer copy).
Usage: python make_tables_v4.py [--suffix _interim_v4] [--g12 g12_v4_s_final_v4]"""
import argparse
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E4 = os.path.join(R, "experiments", "exp6_v4")
GPU_H = {"MAG-G": "—（官方权重）", "MAG-B": "—（官方权重）", "V2O": "18.7（60k，4 卡 4.7 h）", "B0": "18.7 + 5.5（V2 Original + Mix 20k）", "B2": "18.7 + 5.5",
         "E2D": "18.7 + 5.5 + 3.6", "T0": "63.7", "T1": "65.0", "V31": None, "V4": "17.3"}
STAGE = {"MAG-G": "MAGICIAN 官方 e697009，greedy", "MAG-B": "MAGICIAN 官方，beam 10 × 10", "V2O": "Outdoor-only scratch 60k（稀疏 P1；B0 口径）",
         "B0": "Mixed FT 20k（V2 Original → V2-Mix；稀疏 / 稀疏）", "B2": "Mixed FT 20k（同一 V2-Mix；D100 / 稀疏）", "E2D": "Mixed FT 20k + 5k（D100）",
         "T0": "Mixed scratch 40k（D100）", "T1": "Mixed scratch 40k（D100）+ 瓶颈 X-Attn", "V31": "**Outdoor-only scratch 40k**（D100）", "V4": "**Outdoor-only scratch 40k**（D100）+ Ray-Aligned X-Attn"}
ZS = {"V2O": "是（zero-shot）", "V31": "是（zero-shot）", "V4": "是（zero-shot）", "MAG-G": "官方（MP3D 训练未知）", "MAG-B": "官方"}


def f(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def ci(b):
    return f"{b['mean']:.3f} [{b['ci95'][0]:.3f}, {b['ci95'][1]:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--g12", default=None)
    ap.add_argument("--v31-gpuh", default=None)
    a = ap.parse_args()
    g = json.load(open(os.path.join(E4, "reports", f"g3_v4{a.suffix}.json")))
    if a.v31_gpuh:
        GPU_H["V31"] = a.v31_gpuh
    GRP = g["groups"]
    pairs = g["pairs"]

    def delta(hi, lo, d, k):
        key = f"{hi}-{lo}"
        if key in pairs:
            p = pairs[key][d]["paired"][k]
            return f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]（{p['wins']}/{p['losses']}）"
        key = f"{lo}-{hi}"
        if key in pairs:
            p = pairs[key][d]["paired"][k]
            return f"{-p['mean']:+.3f} [{-p['ci95'][1]:+.3f}, {-p['ci95'][0]:+.3f}]（{p['losses']}/{p['wins']}）"
        return "—"
    L = [f"# EXP6 V4 / V3.1 — 统一对比表（Table A–D；自动生成 `make_tables_v4.py{' ' + a.suffix if a.suffix else ''}`；数据 `reports/g3_v4{a.suffix}.json`）", "",
         "均值 = 场景 / 楼层均值的平均，[CI] = 场景级 bootstrap 95 %（MP3D 5 楼层、Macarons++ 6 场景为独立簇；10 / 30 条 episode 的步不是独立样本）；配对差按 (场景, 起点)，括号 = 胜 / 负场景数。历史组不重跑；训练来源与 P1 口径逐行标注，V2 Original 为稀疏口径（与其余 D100 口径不同，单独看）。每步时延为各自运行时记录（共享负载，不可直接比较）。", ""]
    for d, title, nm in (("outdoor", "Table A — Macarons++（Dev-3 + Test-3，30 ep × 100 步）", "AUC@100 / Final@100 / Cov@20"), ("indoor", "Table B — MP3D zero-shot（5 楼层 × 2 起点，150 步）", "AUC@150 / Final@150 / Cov@20")):
        M = g["means"][d]
        L += [f"## {title}", "", f"| 模型 | 训练数据 / 阶段 | {'真正 zero-shot？ | ' if d == 'indoor' else ''}AUC | Final | Cov@20 | Δ vs MAG-G（AUC / Final） | Δ vs MAG-B | Δ vs V2 Original | Δ vs B2 | 每步时延 s | 峰值显存 GB | 训练 GPU-h |",
              "|---|---|" + ("---|" if d == "indoor" else "") + "---|---|---|---|---|---|---|---|---|---|"]
        for gname in GRP:
            zs = (ZS.get(gname, "否（历史 Mixed）") + " | ") if d == "indoor" else ""
            row = f"| {gname} | {STAGE[gname]} | {zs}{ci(M['auc'][gname])} | {ci(M['final'][gname])} | {ci(M['cov20'][gname])} | "
            if gname in ("V4", "V31"):
                row += " | ".join(f"{delta(gname, ref, d, 'auc')} / {delta(gname, ref, d, 'final')}" for ref in ("MAG-G", "MAG-B", "V2O", "B2"))
            else:
                row += " | ".join("—" for _ in range(4))
            row += f" | {M['time_per_move'][gname]['mean']:.2f} | {M['peak_gb'][gname]['mean']:.1f} | {GPU_H.get(gname) or '待填'} |"
            L.append(row)
        if d == "indoor":
            st = g["stall_mp3d"]
            L += ["", "MP3D 中期停滞 episode 数（/ 10，t ∈ [30, 120] 内 ≥ 30 步覆盖增长 < 0.01）：" + "，".join(f"{k} {v['episodes_with_stall']}" for k, v in st.items())]
        L.append("")
    L += ["## 配对差与判定（Task 3 §4：有意义 = ΔAUC ≥ 0.010 且 ΔFinal ≥ 0.010 且胜 ≥ 4 且无场景 < −0.030；退化 = ΔAUC ≤ −0.010 或 ΔFinal ≤ −0.010）", "",
          "| 配对 | 数据集 | ΔAUC [CI]（胜/负） | ΔFinal [CI]（胜/负） | ΔCov@20 | 判定 |", "|---|---|---|---|---|---|"]
    for k, v in pairs.items():
        for d in ("indoor", "outdoor"):
            p, c = v[d]["paired"], v[d]["criteria"]
            verdict = "有意义" if c["meaningful"] else ("退化" if c["degraded"] else "无差异")
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {p['auc']['mean']:+.3f} [{p['auc']['ci95'][0]:+.3f}, {p['auc']['ci95'][1]:+.3f}]（{p['auc']['wins']}/{p['auc']['losses']}） | "
                     f"{p['final']['mean']:+.3f} [{p['final']['ci95'][0]:+.3f}, {p['final']['ci95'][1]:+.3f}]（{p['final']['wins']}/{p['final']['losses']}） | {p['cov20']['mean']:+.3f} | {verdict} |")
    L.append("")
    if a.g12 and os.path.exists(os.path.join(E4, "reports", f"{a.g12}.json")):
        G = json.load(open(os.path.join(E4, "reports", f"{a.g12}.json")))
        S = G["summary"]
        models = [m for m in ("B2", "E2D", "T0", "T1", "V31", "V4") if f"{m}_gain" in S["dev3"]["all"]["means"]]
        L += ["## Table C — G1 / G2（开发集固定状态：MP3D VALSEL 90 态 / Macarons++ Dev-3 275 态；GT 只在评估侧）", "",
              "主指标 = 被选动作的真实一步覆盖增益 pp（越大越好）与 Top-1；regret 只在 `g12_v4_*.md` 附录。", "",
              "| 集 | 指标 | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        for d in ("valsel", "dev3"):
            for met, lab in (("gain", "被选动作真实增益 pp"), ("oracle", "oracle 上限 pp"), ("top1", "Top-1"), ("mae_unknown", "未知射线深度 MAE"), ("mae_new", "真正新表面 MAE"), ("mae_known", "已知射线 MAE"), ("hit_agree", "HIT 一致率"), ("rel_std", "K=4 相对离散度")):
                L.append(f"| {'MP3D VALSEL' if d == 'valsel' else 'Macarons++ Dev-3'} | {lab} | " + " | ".join(f(S[d]['all']['means'].get(f'{m}_{met}', {}).get('mean')) for m in models) + " |")
        L.append("")
    L += ["## Table D — 逐场景 / 楼层（AUC / Final / Cov@20）与逐 episode", "",
          "| 数据集 | 场景 / 楼层 | " + " | ".join(GRP) + " |", "|---|---|" + "---|" * len(GRP)]
    for d in ("indoor", "outdoor"):
        for s, v in g["per_scene"][d].items():
            L.append(f"| {'MP3D' if d == 'indoor' else 'Macarons++'} | {s} | " + " | ".join(f"{v[x]['auc']:.3f} / {v[x].get('final', float('nan')):.3f}" for x in GRP) + " |")
    L += ["", "逐 episode（AUC / Final / Cov@20；MP3D 附停滞窗口长度）：", "", "| 数据集 | 场景 | 起点 | " + " | ".join(GRP) + " |", "|---|---|---|" + "---|" * len(GRP)]
    for d in ("indoor", "outdoor"):
        for e in g["per_episode"][d]:
            L.append(f"| {'MP3D' if d == 'indoor' else 'Macarons++'} | {e['scene']} | {e['start']} | " + " | ".join(f"{e[x]['auc']:.3f} / {e[x]['final']:.3f} / {e[x]['cov20']:.3f}" + (f" (stall {e[x]['stall']})" if d == "indoor" else "") for x in GRP) + " |")
    L += ["", f"覆盖率曲线：`reports/figures/g3_v4{a.suffix}_coverage_curves.png`。", ""]
    out = os.path.join(E4, "reports", f"V4_COMPARISON_TABLES{a.suffix}.md")
    open(out, "w").write("\n".join(L))
    print("written", out)


if __name__ == "__main__":
    main()
