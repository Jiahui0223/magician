"""EXP6 V3 ablation D -- reports/SEARCH_AUDIT.md (CPU; generated). Search definition (from d_policy.py / d_config.yaml) and
measured search statistics from the planner step logs of every finished D episode (Table 1 g3 roots) and of the barts_s0
smokes: legal first-layer candidates, search nodes (path evaluations) per step, predicted (new) views per step, cache reuse
= 1 - predicted views / nodes beyond layer 1, blocked expansions, per-step decision time P50 / P90 / max, planner peak memory,
and the H3 reference (V3-H3 step logs: second-layer unique views)."""
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(D))))
G3 = os.path.join(R, "work_dirs", "cvpr_results", "table1", "g3", "closed_loop", "outdoor")
SM = os.path.join(R, "work_dirs", "exp6_v3", "ablation_study", "experiment_D_h3_lookahead", "smoke")
MODELS = {"H3 (V3-H3)": "V3-H3", "D1": "V3-D1-H3-NORISK", "D2": "V3-D2-DEPTH3-B3", "D3": "V3-D3-DEPTH3-B5", "D4": "V3-D4-DEPTH4-B3"}


def stats(files):
    n_leg, nodes, views, blocked, dec, mem, eps = [], [], [], [], [], [], 0
    for f in files:
        eps += 1
        for l in open(f):
            x = json.loads(l)
            a = x.get("all_scores") or {}
            d = a.get("d") or a.get("h3")
            if not x.get("scores") or not d:
                continue
            n_leg.append(x["n"])
            dec.append(x["decision_s"])
            mem.append(x["peak_mem_gb"])
            if "per_root" in d and d["per_root"] and "nodes" in d["per_root"][0]:
                nodes.append(sum(r["nodes"] for r in d["per_root"]))
                views.append(d.get("predicted_views", 0))
                blocked.append(d.get("blocked", 0))
            else:                                        # H3 log format
                views.append(d.get("n_c2_unique", 0))
                nodes.append(sum(r.get("n_c2", 0) for r in d.get("per_c1", [])))
                blocked.append(d.get("blocked", 0))
    if not dec:
        return None
    nodes, views = np.asarray(nodes, float), np.asarray(views, float)
    return {"episodes": eps, "steps": len(dec), "legal": float(np.mean(n_leg)), "nodes": float(nodes.mean()), "views": float(views.mean()),
            "reuse": float(1 - views.sum() / max(1.0, nodes.sum())), "blocked": float(np.mean(blocked)), "dec_p50": float(np.median(dec)),
            "dec_p90": float(np.quantile(dec, 0.9)), "dec_max": float(np.max(dec)), "mem": float(np.max(mem))}


def main():
    L = ["# SEARCH_AUDIT — D 的搜索定义与实测统计（自动生成，`scripts/search_audit.py`）", "",
         "## 定义（`scripts/d_policy.py`，`configs/d_config.yaml`，D-v1）",
         "- **第 1 层**：模拟器给出的全部合法候选 c1（与 H0–H3 相同，不剪枝）。",
         "- **第 2 层**：每个 c1 的设置网格邻居（`get_neighboring_poses`：6 个平移 + 4 个旋转，边界截断，方位角取模）。",
         "  - 未访问优先：历史位姿与路径上的位姿都视为已访问；",
         "  - 排除路径上已有的位姿；",
         "  - 线段穿过当前 D100 P1 已知表面带的视为阻挡并丢弃；",
         "  - 不读取 GT 碰撞；**穷举**。这与 H3 相同，所以 D1 只差 λ。",
         "- **第 3 / 4 层**（D2–D4）：每个第一步候选（per-root）把上一层的路径按 Q 排序（稳定排序，并列保持生成顺序），只保留前 B 条，再用同一邻居规则展开。",
         "  - beam 作用在**每个根**内部，而不是全局，所以每个 c1 获得相同深度的前瞻；",
         "  - depth = 2 时没有 beam（最后一层不剪枝），所以只改 width 而不加深度对第一步决策没有意义，D2 / D3 必须用 depth 3 才能比较 width。",
         "- **路径价值**：每个样本 k 取路径上所有位姿新格子的并集（每格取最大权重；新格子 = 与已观测 S10 点相距 > 1.0 的预测命中，按 1.0 的格子去重），Q = mean_k − λ·std_k（ddof 0）。",
         "- **c1 的分数**：它在最深一层所有路径中 Q 的最大值（并列时取第一个）；没有任何第 2 层子节点时用单步 Q。",
         "- **选择**：`m_score.choose`（取最大、并列取最小索引、全零走回退）；只执行 c1，真实观测后重新规划。",
         "- **预测缓存**：同一步内每个位姿只预测一次，使用同一组 V3 特征 h 与同一组 K 个噪声图；所有路径复用。",
         "  - 第 2 层的预测请求顺序和批次（每批 16 个）与 H3 完全相同，所以 d0ref（深度 2、λ = 0.25）在 barts_s0 上与 V3-H3 逐位一致；",
         "  - 不跨步缓存，因此不存在跨状态误用。",
         "- **样本配对**：不同位姿的第 k 个样本共用第 k 个噪声图，跨视角配对只是蒙特卡洛近似。",
         "- **虚拟集合**：只用于打分，不写回真实 P1，也不重新编码。", "",
         "## 实测（正式 episode 的规划器步日志；正式运行时 GPU 共享，耗时偏大）", "",
         "| 策略 | episodes / 步数 | 合法 c1 | 每步搜索节点（c1 之外） | 每步新预测视角 | 缓存复用比例 | 每步阻挡展开 | 每步决策 P50 / P90 / 最大（s） | 峰值显存（GB） |",
         "|---|---|---|---|---|---|---|---|---|"]
    for nm, mid in MODELS.items():
        s = stats(sorted(glob.glob(os.path.join(G3, mid, "V2M", "planner_logs", "*.steps.jsonl"))))
        if s:
            L.append(f"| {nm} | {s['episodes']} / {s['steps']} | {s['legal']:.1f} | {s['nodes']:.0f} | {s['views']:.0f} | {s['reuse']:.2f} | {s['blocked']:.1f} | "
                     f"{s['dec_p50']:.1f} / {s['dec_p90']:.1f} / {s['dec_max']:.1f} | {s['mem']:.1f} |")
        else:
            L.append(f"| {nm} | 尚无完成的 episode | | | | | | | |")
    L += ["", "## 冒烟（barts_s0，D 队列启动前每张卡只跑 1 条；d2 / d4 的后半段与队列共用 GPU）", "",
          "| 冒烟 | 步数 | 合法 c1 | 每步搜索节点 | 每步新预测视角 | 缓存复用比例 | 每步决策 P50 / P90（s） |", "|---|---|---|---|---|---|---|"]
    for p in ("d0ref", "d1", "d2", "d4"):
        s = stats(sorted(glob.glob(os.path.join(SM, p, "V2M", "planner_logs", "*.steps.jsonl"))))
        if s:
            L.append(f"| {p} | {s['steps']} | {s['legal']:.1f} | {s['nodes']:.0f} | {s['views']:.0f} | {s['reuse']:.2f} | {s['dec_p50']:.1f} / {s['dec_p90']:.1f} |")
    L += ["", "- **回归**：d0ref 在 barts_s0 上与 Table 1 V3-H3 的执行动作和官方覆盖逐位一致。"]
    open(os.path.join(D, "reports", "SEARCH_AUDIT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-14:]))


if __name__ == "__main__":
    main()
