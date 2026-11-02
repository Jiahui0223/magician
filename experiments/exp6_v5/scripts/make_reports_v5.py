"""EXP6 V5 -- per-gate reports from the measured analyses (CPU): reports/g3_v5.json -> G3_OUTDOOR.md, G3_INDOOR_ZEROSHOT.md;
reports/g12_v5_s_final.{json,md} + belief_metrics_eval.json -> G1_G2.md. Usage: python make_reports_v5.py"""
import json
import os

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E = f"{R}/experiments/exp6_v5/reports"
MAIN = ("V4", "V4C", "V5")
REF = ("MAG-G", "MAG-B", "B2", "E2D", "V31")


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(p):
    return f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"


def g3_report(G, d, title, n_ep, steps):
    M = G["means"][d]
    L = [f"# {title}", "", f"冻结规划器（候选生成器、K = 4、10 步 Euler、n_win = 5、D100 输入 / S10 稀疏 DistP1 评分）；{n_ep} 个 episode × {steps} 步；同起点同种子同动作规则；"
         "评估场景已多次使用，**非盲测**；数字全部来自 `scripts/eval_g3.py` → `g3_v5.json`。", "",
         "| 组 | AUC [CI] | Final [CI] | Cov@20 | Cov@50 | 每步时间 s* | 峰值显存 GB |", "|---|---|---|---|---|---|---|"]
    for g in REF + MAIN + (("A1",) if "A1" in G["groups"] else ()):
        if g not in G["groups"]:
            continue
        a, fi = M["auc"][g], M["final"][g]
        L.append(f"| {g} | {a['mean']:.3f} [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] | {fi['mean']:.3f} [{fi['ci95'][0]:.3f}, {fi['ci95'][1]:.3f}] | {f3(M['cov20'][g]['mean'])} | {f3(M['cov50'][g]['mean'])} | "
                 f"{M['time_per_move'][g]['mean']:.2f} | {M['peak_gb'][g]['mean']:.1f} |")
    L += ["", "\\* 共享负载下的记录，只作量级参考。", "", "| 配对 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 [CI] | 有意义 | 退化 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in G["pairs"].items():
        p, c = v[d]["paired"], v[d]["criteria"]
        L.append(f"| {k} | {ci(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {ci(p['final'])} | {p['final']['wins']}/{p['final']['losses']} | {ci(p['cov20'])} | {c['meaningful']} | {c['degraded']} |")
    L += ["", "逐场景 / 楼层（AUC / Final；逐 episode 的 Cov@20 在 `g3_v5.json` 的 per_episode）：", "", "| 场景 | " + " | ".join(MAIN + ("B2", "E2D", "MAG-B")) + " |", "|---|" + "---|" * (len(MAIN) + 3)]
    for s, v in G["per_scene"][d].items():
        L.append(f"| {s} | " + " | ".join(f"{v[g]['auc']:.3f} / {v[g].get('final', float('nan')):.3f}" for g in MAIN + ("B2", "E2D", "MAG-B") if g in v) + " |")
    if d == "indoor":
        L += ["", "MP3D 中期停滞 episode 数（/ 10）：" + "，".join(f"{g} {G['stall_mp3d'][g]['episodes_with_stall']}" for g in G["groups"])]
    return "\n".join(L) + "\n"


def main():
    G = json.load(open(f"{E}/g3_v5.json"))
    open(f"{E}/G3_OUTDOOR.md", "w").write(g3_report(G, "outdoor", "G3 —— Macarons++（6 场景 × 5 起点 × 100 步 = 30 ep）", 30, 100))
    open(f"{E}/G3_INDOOR_ZEROSHOT.md", "w").write(g3_report(G, "indoor", "G3 —— MP3D zero-shot（5 楼层 × 2 起点 × 150 步 = 10 ep）", 10, 150))
    L = ["# G1 / G2 —— 开发集固定状态（365 态：MP3D VALSEL 90、Macarons++ Dev-3 B0 150 + B2 125）", "",
         "同状态、同候选、同噪声；B2 回放与 E2 行逐位相同（不同则 STOP）。主指标：被选候选的真实一步覆盖增益（pp）、Top-1；regret 只在附录。", ""]
    if os.path.exists(f"{E}/g12_v5_s_final.md"):
        L += open(f"{E}/g12_v5_s_final.md").read().splitlines()[1:]
    if os.path.exists(f"{E}/belief_metrics_eval.json"):
        B = json.load(open(f"{E}/belief_metrics_eval.json"))
        L += ["", "## G1-belief：V5 多表面 Surface Belief（评估端网格 GT，只用于指标）", "",
              f"每个评分状态的每个合法候选视角 256 条射线（UNKNOWN 优先）；查询 = 每个 GT 表面深度 + [t_lo, t_hi] 内 64 个对数均匀深度；Δ = {B['delta_model_units']:.3f} 模型单位（8 VS），阈值 0.5。", "",
              "| 集 | 射线组 | 射线 | ≥ 2 表面 | 首表面召回 | 第二+ 召回 | <8 VS | 8–32 | 32–128 | ≥128 | 近 <64 VS | 64–256 | ≥256 | 网格 precision | recall | 正例率 | Brier | ECE | 首表面后虚假表面率 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in B["metrics"].items():
            for sub in ("all", "unknown_rays"):
                x = v[sub]
                b, dd = x["recall_second_plus_by_gap"], x["recall_second_plus_by_depth"]
                L.append(f"| {k.split('eval_fixed/')[-1]} | {sub} | {x['rays']:,} | {f3(x['rays_ge2_surfaces'])} | {f3(x['recall_first'])} | {f3(x['recall_second_plus'])} | "
                         + " | ".join(f"{f3(b[q]['recall'])}" for q in ("gap<8vox", "gap8-32vox", "gap32-128vox", "gap>=128vox")) + " | "
                         + " | ".join(f"{f3(dd[q]['recall'])}" for q in ("t<64vox", "t64-256vox", "t>=256vox")) + " | "
                         f"{f3(x['grid_precision'])} | {f3(x['grid_recall'])} | {f3(x['grid_pos_rate'])} | {f3(x['grid_brier'])} | {f3(x['grid_ece'])} | {f3(x['false_surface_rate_behind_first'])} |")
        L += ["", "对照：V4 / V4C / B2 / E2-D100 的输出是 K = 4 个首中深度，没有第二表面，第二+ 召回按构造为 0；它们的首中质量见上表的 MAE / 命中一致。"]
    open(f"{E}/G1_G2.md", "w").write("\n".join(L) + "\n")
    print("written G3_OUTDOOR.md, G3_INDOOR_ZEROSHOT.md, G1_G2.md")


if __name__ == "__main__":
    main()
