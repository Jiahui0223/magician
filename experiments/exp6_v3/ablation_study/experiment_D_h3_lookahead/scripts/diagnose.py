"""EXP6 V3 ablation D -- colosseum / pisa late-coverage diagnosis (CPU; generated; evaluation side only, after execution).
Policies: H0 (Table 1 V3), H3 (V3-H3), D1 / D2 / D3 / D4 (Table 1 records when present). Per episode: official coverage curve,
Cov@20 / Cov@50 / Final, first step whose executed pose differs from H0, plateau step (first t >= 20 after which the
coverage gain over the next 20 steps is < 1 pp), unique camera positions / orientations visited, extent of the visited
positions on the settings grid, revisits, fallback steps, Spearman between the chosen action's planner score and its
realised one-step official coverage gain, D search nodes / predicted views per step.
Outputs reports/COLOSSEUM_PISA_DIAGNOSIS.md, results/diagnosis_episodes.csv, figures/{colosseum,pisa}_{0_100,0_20,20_50}.png."""
import csv
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(D))))
E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
POL = {"H0": "V3", "H3": "V3-H3", "D1": "V3-D1-H3-NORISK", "D2": "V3-D2-DEPTH3-B3", "D3": "V3-D3-DEPTH3-B5", "D4": "V3-D4-DEPTH4-B3"}
COL = {"H0": "#333333", "H3": "#d6336c", "D1": "#2a78d6", "D2": "#1baf7a", "D3": "#c9a227", "D4": "#7b4fd6"}
SCENES = ("colosseum", "pisa")


def steps_of(rec, s, k):
    ep = os.path.join(R, rec["episode_file"])
    pl = glob.glob(os.path.join(os.path.dirname(os.path.dirname(ep)), "planner_logs", f"pl_V2M_{s}_s{k}_r0.steps.jsonl"))
    return [json.loads(l) for l in open(pl[0])] if pl else []


def keyv(k):
    return [int(x) for x in str(k).strip("[]").split(",")]


def main():
    rows, curves = [], {}
    for s in SCENES:
        for k in range(5):
            h0 = os.path.join(E, "V3", f"{s}_s{k}.json")
            h0k = json.load(open(h0))["executed_keys"] if os.path.exists(h0) else None
            for p, mid in POL.items():
                f = os.path.join(E, mid, f"{s}_s{k}.json")
                if not os.path.exists(f):
                    continue
                r = json.load(open(f))
                c = np.asarray(r["coverage_official"])
                curves.setdefault((s, p), {})[k] = c
                keys = [keyv(x) for x in r["executed_keys"]]
                pos = {tuple(x[:3]) for x in keys}
                ori = {tuple(x[3:]) for x in keys}
                P = np.asarray([x[:3] for x in keys])
                div = next((t for t, (a, b) in enumerate(zip(r["executed_keys"], h0k)) if a != b), None) if h0k else None
                plateau = next((t for t in range(20, 81) if c[t + 20] - c[t] < 0.01), None)
                st = steps_of(r, s, k)
                fb = sum(1 for x in st if (x.get("choice") or {}).get("fallback"))
                sel, gain = [], []
                for x in st:
                    t, pk = x["t"], (x.get("choice") or {}).get("pick")
                    if pk is not None and x.get("scores") and t + 1 <= 100:
                        sel.append(x["scores"][pk]); gain.append(c[t + 1] - c[t])
                rho = spearmanr(sel, gain).correlation if len(sel) > 5 and np.std(sel) > 0 else np.nan
                dd = [x["all_scores"].get("d") or x["all_scores"].get("h3") for x in st if x.get("all_scores")]
                nodes = [sum(z["nodes"] for z in d_["per_root"]) for d_ in dd if d_ and "per_root" in d_ and d_["per_root"] and "nodes" in d_["per_root"][0]]
                views = [d_.get("predicted_views", d_.get("n_c2_unique")) for d_ in dd if d_]
                rows.append({"scene": s, "start": k, "policy": p, "cov20": c[20], "cov50": c[50], "final": c[100], "auc100": float(np.trapz(c) / 100),
                             "first_divergence_vs_h0": div, "plateau_step": plateau, "unique_positions": len(pos), "unique_orientations": len(ori),
                             "extent_xyz": (P.max(0) - P.min(0)).tolist(), "revisits": sum(1 for i in range(1, len(keys)) if keys[i] in keys[:i]),
                             "fallback_steps": fb, "rho_score_vs_gain": float(rho), "nodes_per_step": float(np.mean(nodes)) if nodes else None,
                             "views_per_step": float(np.mean([v for v in views if v is not None])) if views else None})
    os.makedirs(os.path.join(D, "results"), exist_ok=True)
    with open(os.path.join(D, "results", "diagnosis_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.makedirs(os.path.join(D, "figures"), exist_ok=True)
    for s in SCENES:
        for lo, hi, nm in ((0, 100, "0_100"), (0, 20, "0_20"), (20, 50, "20_50")):
            fig, ax = plt.subplots(figsize=(6.8, 3.6))
            for p in POL:
                d = curves.get((s, p), {})
                if not d:
                    continue
                M = np.asarray([d[k] for k in sorted(d)])
                x = np.arange(lo, hi + 1)
                for row in M:
                    ax.plot(x, row[lo:hi + 1], color=COL[p], lw=0.6, alpha=0.35)
                ax.plot(x, M.mean(0)[lo:hi + 1], color=COL[p], lw=2.0, label=f"{p} ({len(d)} starts)")
            ax.set_xlim(lo, hi)
            ax.set_xlabel("exploration step")
            ax.set_ylabel("coverage (MAGICIAN official)")
            ax.set_title(f"{s}: steps {lo}-{hi} (thin = individual starts)", loc="left", fontsize=9)
            ax.legend(fontsize=7.5, loc="lower right")
            ax.grid(alpha=0.25)
            fig.savefig(os.path.join(D, "figures", f"{s}_{nm}.png"), dpi=200, bbox_inches="tight")
            fig.savefig(os.path.join(D, "figures", f"{s}_{nm}.pdf"), bbox_inches="tight")
            plt.close(fig)
    L = ["# COLOSSEUM_PISA_DIAGNOSIS — 后期覆盖退化诊断（自动生成，`scripts/diagnose.py`）", "",
         "- **评价**：所有指标都在动作执行之后用官方覆盖计算，规划器不读取 GT。",
         "- **平台步**：t ≥ 20 之后，第一个“接下来 20 步覆盖增益 < 1 pp”的步数。",
         "- **ρ**：所选动作的规划器分数与它实际单步覆盖增益的 Spearman 相关。", "",
         "| 场景 | 策略 | 起点数 | Cov@20 | Cov@50 | Final | 首次与 H0 分叉的步数（中位数） | 平台步（中位数） | 不同位置数 | 不同朝向数 | 位置范围（网格 x / y / z） | 重复访问 | 回退步 | ρ（分数 vs 实际增益） | 每步搜索节点 / 预测视角 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in SCENES:
        for p in POL:
            rr = [r for r in rows if r["scene"] == s and r["policy"] == p]
            if not rr:
                continue
            med = lambda k: np.nanmedian([np.nan if r[k] is None else r[k] for r in rr])  # noqa: E731
            ext = np.mean([r["extent_xyz"] for r in rr], 0)
            nv = [r["nodes_per_step"] for r in rr if r["nodes_per_step"] is not None]
            vv = [r["views_per_step"] for r in rr if r["views_per_step"] is not None]
            L.append(f"| {s} | {p} | {len(rr)} | {np.mean([r['cov20'] for r in rr]):.3f} | {np.mean([r['cov50'] for r in rr]):.3f} | {np.mean([r['final'] for r in rr]):.3f} | "
                     f"{med('first_divergence_vs_h0'):.0f} | {med('plateau_step'):.0f} | {np.mean([r['unique_positions'] for r in rr]):.1f} | {np.mean([r['unique_orientations'] for r in rr]):.1f} | "
                     f"{ext[0]:.1f} / {ext[1]:.1f} / {ext[2]:.1f} | {np.mean([r['revisits'] for r in rr]):.1f} | {np.mean([r['fallback_steps'] for r in rr]):.1f} | "
                     f"{np.nanmean([r['rho_score_vs_gain'] for r in rr]):+.2f} | {np.mean(nv) if nv else float('nan'):.0f} / {np.mean(vv) if vv else float('nan'):.0f} |")
    L += ["", "**逐起点**（Cov@50 / Final / 不同位置数）：", "", "| 场景 | 起点 | " + " | ".join(POL) + " |", "|---|---|" + "---|" * len(POL)]
    for s in SCENES:
        for k in range(5):
            cells = []
            for p in POL:
                r = next((r for r in rows if r["scene"] == s and r["start"] == k and r["policy"] == p), None)
                cells.append("—" if r is None else f"{r['cov50']:.3f} / {r['final']:.3f} / {r['unique_positions']}")
            L.append(f"| {s} | {k} | " + " | ".join(cells) + " |")
    L += ["", "图：`figures/{colosseum,pisa}_{0_100,0_20,20_50}.png`。"]
    open(os.path.join(D, "reports", "COLOSSEUM_PISA_DIAGNOSIS.md"), "w").write("\n".join(L) + "\n")
    print("episodes", len(rows))


if __name__ == "__main__":
    main()
