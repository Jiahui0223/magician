"""CVPR overnight -- report/RESULTS.md, report/MISSING.md, report/RUN_LOG.md from the files on disk (CPU). python make_reports.py
Numbers are read from the generated CSVs / the manifest / the queue logs, never typed by hand. The hand-written part of the run log
(decisions, incidents) lives in report/RUN_LOG_events.md and is included verbatim."""
import csv
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvpr_lib as L  # noqa: E402

rc = lambda p: list(csv.DictReader(open(p))) if os.path.exists(p) else []  # noqa: E731
f3 = lambda x, s=False: "—" if x in (None, "") else (f"{float(x):+.3f}" if s else f"{float(x):.3f}")  # noqa: E731
QUEUE = {"V3-T0-K4D1": "t3", "V3-T0-K1D2": "t3", "V3-T0-K1D1": "t3", "RANDOM": "random", "MACARONS": "macarons"}
for _d in (2, 1):
    for _j in (1, 2, 3):
        QUEUE[f"V3-T0-K1D{_d}-N{_j}"] = "t3n"


def queue_stats():
    """per method: successful episodes, walltime stats, non-zero exits, failed tries, gave up (from the Table 1 queue logs of 2026-11-10)."""
    st = {}
    for f in glob.glob(os.path.join(L.T1W, "logs", "queue_gpu114_*.log")):
        for line in open(f):
            m = re.match(r"exit (\d+) ([^:]+):([^:]+):(\d+) (\S+) gpu(\d) (\d+)s (\S+ \S+)", line)
            if not m or m.group(2) not in QUEUE or not m.group(8).startswith("2026-11-10"):
                continue
            d = st.setdefault(m.group(2), {"ok": [], "bad": [], "t": []})
            (d["ok"] if m.group(1) == "0" else d["bad"]).append(int(m.group(7)))
            d["t"].append(m.group(8))
    for m in st:
        st[m]["failed_tries"] = len(glob.glob(os.path.join(L.T1W, "failed", f"{m}_*_try*")))
    gu = os.path.join(L.T1W, "logs", "gave_up.log")
    for m in st:
        st[m]["gave_up"] = sum(1 for l in open(gu) if l.split()[1].startswith(m + ":")) if os.path.exists(gu) else 0
    return st


def main():
    man = rc(os.path.join(L.MAN, "episodes.csv"))
    cnt = {}
    for r in man:
        c = cnt.setdefault(r["method_id"], {"COMPLETE": 0, "INCOMPLETE": 0, "FAILED": 0, "BLOCKED": 0, "missing": []})
        c[r["run_status"]] += 1
        if r["run_status"] in ("INCOMPLETE", "FAILED"):
            c["missing"].append(f"{r['scene_id']}:{r['start_id']}")
    now = time.strftime("%F %T")
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=L.REPO).stdout.strip()[:12]
    # ------------------------------------------------------------------------------------------------ RESULTS
    R = [f"# RESULTS — CVPR overnight 2026-11-10（生成于 {now}，git {head}）", "",
         "由 `scripts/make_reports.py` 从生成的 CSV / manifest 自动汇总。所有数值 = 冻结协议（14 场景 × 5 官方起点 × 100 步，MAGICIAN 官方覆盖，场景等权）下的真实 episode；"
         "未跑满 70 条的方法不出现在 All-14 数值里。CI = 场景级配对 bootstrap 95%（重抽样单位 = 场景）。", ""]
    t1 = rc(os.path.join(L.FIG, "table1", "table1_all14_full.csv"))
    R += ["## Table 1 — Main benchmark（All-14）", "", "路径：`fig/table1/{table1.pdf, table1.png, table1.tex, table1_paper.csv, table1_all14_full.csv, table1_auc_cov_long.csv, table1_auc_wide.csv, table1_cov_wide.csv, table1_paired_ours_minus_baseline.csv, provenance.csv}`", "",
          "| 方法 | 来源 | 状态 | episode | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t1:
        full = r["status"] == "COMPLETE"
        R.append(f"| {r['method']} | {r['source']} | {r['status']} | {r['n_completed']}/70 | " + " | ".join(f3(r[n]) if full else "—" for n in ("AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final")) + " |")
    pr = rc(os.path.join(L.FIG, "table1", "table1_paired_ours_minus_baseline.csv"))
    if pr:
        R += ["", "Ours − 基线（场景配对差，[95% CI]，胜 / 负场景数）：", "", "| 基线 | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final |", "|---|---|---|---|---|---|"]
        for b in dict.fromkeys(r["method_b"] for r in pr):
            d = {r["metric"]: r for r in pr if r["method_b"] == b}
            R.append(f"| {L.METHODS[b]['display']} | " + " | ".join(f"{f3(d[n]['mean_diff'], 1)} [{f3(d[n]['ci95_lo'], 1)}, {f3(d[n]['ci95_hi'], 1)}] {d[n]['scene_wins']}/{d[n]['scene_losses']}" for n in ("AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final")) + " |")
    R += ["", "- SCONE、FisherRF：**BLOCKED**（仓库内没有可在冻结协议下运行的实现，见 `MISSING.md`）；表中留空，未填入任何论文数字。",
          "- Random Walk、MACARONS 标 †（adapted）：在同一模拟器 / 起点 / 合法动作 / 官方覆盖下由 EXP3b 的冻结运行器运行，不是 MAGICIAN 论文作者的实现；MACARONS 与 MAGICIAN 使用同一份作者权重（在 14 个评测场景中的 8 个上训练过）。",
          "- 论文报告值（不同实现、论文只有 AUC@100 / Final）只在 `provenance.csv` 的参考列里。", ""]
    # Table 2
    t2 = rc(os.path.join(L.FIG, "table2", "table2_splits.csv"))
    R += ["## Table 2 — T0 留出场景上的比较", "", "路径：`fig/table2/{table2.pdf, table2.png, table2.tex, table2_splits.csv, split_manifest.csv}`", "",
          "| 划分 | 方法 | AUC@20 | AUC@50 | AUC@100 | Final |", "|---|---|---|---|---|---|"]
    for g in ("Train-6", "Dev-5", "Test-3", "Dev+Test-8"):
        for m in ("MAG-B", "V3-T0-D1"):
            v = [r for r in t2 if r["split"] == g and r["method_id"] == m and r["row_kind_or_metric"] == "value"]
            if v:
                R.append(f"| {g} | {L.METHODS[m]['display']} | " + " | ".join(f3(v[0][n]) for n in ("AUC@20", "AUC@50", "AUC@100", "Final")) + " |")
        d = {r["row_kind_or_metric"]: r for r in t2 if r["split"] == g and r["method_id"] == "V3-T0-D1 minus MAG-B"}
        if d:
            R.append(f"| {g} | Δ Ours − MAGICIAN [95% CI] | " + " | ".join(f"{f3(d[n]['AUC@20'], 1)} [{f3(d[n]['Cov@20'], 1)}, {f3(d[n]['AUC@50'], 1)}]" for n in ("AUC@20", "AUC@50", "AUC@100", "Final")) + " |")
    R += ["", "- 这些是 **T0 的留出场景**（T0 的训练与选模从未使用 Dev-5 / Test-3），不是\"两种方法都没见过的场景\"：MAGICIAN 用的 MACARONS 权重在 14 个场景中的 8 个上训练过（`split_manifest.csv` 逐场景列出）。",
          "- Test-3 对系统设计不是盲测：此前的规划器变体在 All-14（含 Test-3）上评过。Test-3 只有 3 个场景，CI 很宽。", ""]
    # Table 3
    t3 = rc(os.path.join(L.FIG, "table3", "table3_ablation.csv"))
    ef = rc(os.path.join(L.FIG, "table3", "table3_effects.csv"))
    nz = rc(os.path.join(L.FIG, "table3", "table3_k1_noise_replicates.csv"))
    lab = t3[0]["scenes_used"] if t3 else "—"
    R += ["## Table 3 — K × 规划深度（冻结 V3-T0）", "", "路径：`fig/table3/{table3.pdf, table3.png, table3.tex, table3_ablation.csv, table3_effects.csv, table3_k1_noise_replicates.csv, ablation_manifest.csv}`", "",
          f"使用的场景：**{lab}**。", "", "| K | 深度 | episode | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final | 规划器每步决策秒数（共享 GPU 下实测） |", "|---|---|---|---|---|---|---|---|---|"]
    for r in t3:
        R.append(f"| {r['K']} | {r['planning_depth']}{'（主方法）' if r['method_id'] == 'V3-T0-D1' else ''} | {r['n_completed']}/70 | " + " | ".join(f3(r[n]) for n in ("AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final"))
                 + f" | {f3(r['planner_decision_s_per_step_shared_gpu'])[:5] if r['planner_decision_s_per_step_shared_gpu'] else '—'} |")
    if ef:
        R += ["", "效应（场景配对差 [95% CI]，胜 / 负场景数）：", "", "| 对比 | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final |", "|---|---|---|---|---|---|"]
        for c in dict.fromkeys(r["contrast"] for r in ef):
            d = {r["metric"]: r for r in ef if r["contrast"] == c}
            R.append(f"| {c} | " + " | ".join(f"{f3(d[n]['mean_diff'], 1)} [{f3(d[n]['ci95_lo'], 1)}, {f3(d[n]['ci95_hi'], 1)}] {d[n]['scene_wins']}/{d[n]['scene_losses']}" for n in ("AUC@20", "AUC@50", "AUC@100", "Cov@20", "Final")) + " |")
    if nz:
        R += ["", "K = 1 噪声复核（K = 4 噪声组的第 0–3 张噪声图各自单独作为 K = 1 的噪声；只统计跑满 70 条的噪声图）：", "",
              "| 深度 | 指标 | 完成的噪声图 | 各噪声图的 All-14 值 | 均值 | 标准差 | K = 4 的值 | K4 − mean(K1) [95% CI] |", "|---|---|---|---|---|---|---|---|"]
        for r in nz:
            if r["metric"] in ("AUC@20", "AUC@100", "Final"):
                R.append(f"| {r['depth']} | {r['metric']} | {r['noise_maps']} | {' / '.join(f3(x) for x in r['all14_value_per_noise_map'].split(';'))} | {f3(r['mean_over_noise_maps'])} | {float(r['sd_over_noise_maps(ddof1)']):.4f} | {f3(r['K4_value'])} | "
                         f"{f3(r['K4_minus_mean_K1(scene-paired)'], 1)} [{f3(r['ci95_lo'], 1)}, {f3(r['ci95_hi'], 1)}] |")
    R += ["", "- 四格使用同一冻结 T0 checkpoint、模拟器、候选、噪声键、value_cov（eps 1.0、cell 1.0）、λ = 0、R10、Euler 10；深度 1 的分数是 value_cov 的样本均值，不是 DistP1。",
          "- K = 1 用 K = 4 噪声组的第 0 张噪声图（同种子；bf16 批大小不同导致非逐位相同，t = 0 的相对差 ≤ 2.5%）。", ""]
    # Figures
    cm = rc(os.path.join(L.FIG, "fig3", "curve_methods.csv"))
    R += ["## Fig 3 — 覆盖曲线与累计 AUC", "", "路径：`fig/fig3/{fig3_combined.pdf, fig3_combined.png, coverage_curve.pdf, auc_curve.pdf, fig3_combined_ci.pdf, fig3_data.csv, curve_methods.csv}`", "",
          "画出的方法（跑满 70 条的才画）：" + "、".join(f"{r['method']}（{r['n_completed']}/70{'，已画' if r['drawn'] == '1' else '，未画：' + r['status']}）" for r in cm) + "。",
          "曲线是官方覆盖的场景等权平均，未做 cummax 或平滑；(b) 是 T = 1..100 的 AUC@T。`fig3_combined_ci` 带场景 bootstrap 95% 阴影（补充材料用）。", ""]
    sel = os.path.join(L.FIG, "fig4", "selection.json")
    R += ["## Fig 4 — 真实多假设深度", "", "路径：`fig/fig4/{hypothesis_depth.pdf, hypothesis_depth.png, hypothesis_depth_all.pdf, panels/, visualization_manifest.csv, selection.json}`", ""]
    if os.path.exists(sel):
        s = json.load(open(sel))
        R += [f"候选池 {s['n_states']} 个真实决策状态（每个划分 2 个场景 × 5 个起点 × 步 0 / 10 / … / 50），全部通过精确性校验（4× 网格哈希与每个样本的一步价值都等于规划器日志）；符合条件 {s['n_eligible']} 个。"
              "按预先写在脚本里的规则选出：", "", "| 划分 / 类型 | 状态 | UNKNOWN 射线占比 | 样本间命中不一致占比 | 命中一致率（对真实深度） | 深度中位绝对误差 |", "|---|---|---|---|---|---|"]
        for k, v in s["selected"].items():
            R.append(f"| {k} | {v['scene']} 起点 {v['start']} 步 {v['t']} | {v['unknown_share']:.2f} | {v['flip_share_on_unknown']:.3f} | {v['hit_agreement_on_unknown']:.3f} | {v['median_abs_depth_err_on_unknown']:.2f} |")
        R += ["", f"主图 `hypothesis_depth` = {'；'.join(s['main_figure'])}；全部 6 个例子在 `hypothesis_depth_all` 和 `panels/`。",
              "k = 1..4 是同一模型的 4 个噪声样本（不是四种模型配置）；每个样本是沿候选视角射线解码出的首次命中深度，不是一个完整一致的 3D 世界。真实深度是执行后下一步的观测，只用于显示。", ""]
    ps = rc(os.path.join(L.FIG, "fig5", "per_scene_auc20.csv"))
    R += ["## Fig 5 — 逐场景 AUC@20", "", "路径：`fig/fig5/{per_scene_radar.pdf, per_scene_radar.png, per_scene_delta.pdf, per_scene_delta.png, per_scene_auc20.csv}`", ""]
    if ps:
        cols = [c for c in ps[0] if c.startswith("AUC@20_")]
        R += ["| 场景 | 划分 | " + " | ".join(L.METHODS[c[7:]]["display"] for c in cols) + " | Δ Ours − MAGICIAN（± 5 个配对起点的 SE） | 起点胜数 |", "|---|---|" + "---|" * (len(cols) + 2)]
        for r in ps:
            R.append(f"| {r['official_name']} | {r['split']} | " + " | ".join(f3(r[c]) for c in cols) + f" | {f3(r['delta_ours_minus_magician'], 1)} ± {float(r['delta_se_over_5_paired_starts']):.3f} | {r['starts_won_of_5']}/5 |")
        lose = [r["official_name"] for r in ps if float(r["delta_ours_minus_magician"]) < 0]
        R += ["", f"14 个轴统一 0–1 标尺，按冻结场景顺序；Ours 低于 MAGICIAN 的场景：{('、'.join(lose)) if lose else '无'}（没有隐藏）。", ""]
    open(os.path.join(L.REP, "RESULTS.md"), "w").write("\n".join(R) + "\n")
    # ------------------------------------------------------------------------------------------------ MISSING
    M = [f"# MISSING — 尚缺的 episode 与可直接执行的下一步（生成于 {now}）", "", "| 方法 / 配置 | 完成 | 缺失 | 失败（放弃） | 状态 |", "|---|---|---|---|---|"]
    for m, d in L.METHODS.items():
        c = cnt.get(m, {})
        st = "BLOCKED" if d.get("blocked") else ("完成" if c.get("COMPLETE") == 70 else ("未开始" if c.get("COMPLETE", 0) == 0 else "部分完成"))
        M.append(f"| {m}（{d['display']}） | {c.get('COMPLETE', 0)}/70 | {c.get('INCOMPLETE', 0)} | {c.get('FAILED', 0)} | {st} |")
    M += ["", "## 逐项缺口与下一步命令", ""]
    for m, d in L.METHODS.items():
        c = cnt.get(m, {})
        if d.get("blocked"):
            M += [f"### {m} — BLOCKED", "", f"- 原因：{d['blocked']}", "- 需要的工作（未做）："
                  + ("为 SCONE 写 Macarons++ 场景级规划器适配（把 `MACARONS/demos/scone_demo.py` 的占据 + coverage-gain 网络接到冻结 EpisodeSim 的合法动作列表上，参照 `experiments/exp3b_macaronspp_transfer/m_t7_macarons.py` 的 R1–R8 修复），再按 `table1/README.md` 的“加入新模型”流程登记并跑 70 条。" if m == "SCONE"
                     else "引入 FisherRF 官方代码与 3DGS 依赖（用户负责环境安装），写冻结模拟器适配器（每步在线训练 3DGS、对合法动作列表算 Fisher 信息增益），再登记并跑 70 条。"), ""]
        elif c.get("COMPLETE", 0) < 70:
            q = QUEUE.get(m)
            miss = c.get("missing", [])
            M += [f"### {m} — 缺 {len(miss)} 条", "", "- 缺失：" + "、".join(miss[:70]), "- 补跑（在一个 L40S 分配内，已完成的会自动跳过）：", "```bash",
                  "cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene",
                  (f"python - <<'EOF'\nopen('work_dirs/cvpr_results/overnight_20261110/jobs/fill_{m}.txt','w').write(''.join(f'{m}:{{x}}\\n' for x in {miss!r}))\nEOF\n"
                   f"for g in 0 1 2 3; do setsid nohup bash experiments/cvpr_results/table1/scripts/t1_queue.sh $g $PWD/work_dirs/cvpr_results/overnight_20261110/jobs/fill_{m}.txt > /dev/null 2>&1 < /dev/null & done"
                   + ("   # dunnottar 需独占一张卡" if any(x.startswith("dunnottar") for x in miss) and m != "RANDOM" else "")),
                  "bash experiments/cvpr_results/scripts/run_all.sh && /bsuscratch/jiahuizhang/envs/seen2scene/bin/python experiments/cvpr_results/scripts/make_reports.py", "```", ""]
    M += ["## 本轮未安排的项目（计划书 P2）", "", "- Ray Encoder 消融：未做（需要重新训练，不在 8 小时窗口内）。",
          "- Runtime 对照（独占 GPU 的逐方法计时）：未做；Table 3 的 CSV 只给出共享 GPU 下的规划器每步决策秒数，不能当作正式 runtime。",
          "- sofia_church（第 15 个场景）：冻结协议 v1.0 下所有方法都不可运行，始终报告 All-14。", ""]
    open(os.path.join(L.REP, "MISSING.md"), "w").write("\n".join(M) + "\n")
    # ------------------------------------------------------------------------------------------------ RUN_LOG
    qs = queue_stats()
    ev = os.path.join(L.REP, "RUN_LOG_events.md")
    G = [f"# RUN_LOG — CVPR overnight 2026-11-10（生成于 {now}，git {head}）", "", "## 环境", "",
         "| 项 | 值 |", "|---|---|", "| 节点 / Slurm | gpu114，job **3489889**（vscode-dev 交互分配，4 × NVIDIA L40S 46 GB，60 CPU，500 GB RAM，2026-11-09 20:35 → 11-14 20:35） |",
         "| 运行方式 | 分配内的 `setsid nohup` 进程（没有提交新的 sbatch 作业：所有 L40 均已分配，新作业只能靠抢占他人作业启动，见 `OVERNIGHT_AUDIT.md` §5） |",
         "| 环境 | 模拟器 / 官方重放 / Random / MACARONS：`/bsuscratch/jiahuizhang/envs/magician`；规划器 / Fig 4 前向 / 汇总：`/bsuscratch/jiahuizhang/envs/seen2scene` |",
         "| 队列 | `experiments/cvpr_results/table1/scripts/t1_queue.sh GPU JOBFILE`：闭环 → 官方覆盖重放（不逐位相同即失败）→ 规范记录；noclobber claim；失败释放重试 ≤ 3 次；已完成的跳过（可断点续跑） |", "",
         "## 各队列的执行统计（来自 `work_dirs/cvpr_results/table1/g3/logs/queue_gpu114_*.log`，仅 2026-11-10）", "",
         "| 方法 | 成功 episode | 单条墙钟（均值 / 最大，秒，含官方重放） | 首条 → 末条完成时间 | 非零退出 | 失败重试目录 | 放弃 |", "|---|---|---|---|---|---|---|"]
    for m in QUEUE:
        if m in qs:
            d = qs[m]
            G.append(f"| {m} | {len(d['ok'])} | {np.mean(d['ok']):.0f} / {max(d['ok'])} | {min(d['t'])[11:]} → {max(d['t'])[11:]} | {len(d['bad'])} | {d['failed_tries']} | {d['gave_up']} |" if d["ok"] else f"| {m} | 0 | — | — | {len(d['bad'])} | {d['failed_tries']} | {d['gave_up']} |")
        else:
            G.append(f"| {m} | 0（未启动） | — | — | — | — | — |")
    for name, p in (("回归 / 冒烟", "regression.log"), ("回归第二轮（终版封装）", "regression2.log"), ("Fig 4 扫描与背面检查", "fig4_scan.log"), ("阶段标记", "table3_chain.log")):
        fp = os.path.join(L.OV, "logs", p)
        if os.path.exists(fp):
            G += ["", f"### {name}（`work_dirs/cvpr_results/overnight_20261110/logs/{p}`）", "", "```"] + [l.rstrip() for l in open(fp)] + ["```"]
    fp = os.path.join(L.OV, "logs", "regression_check.txt")
    if os.path.exists(fp):
        G += ["", "### 回归比对结果（`logs/regression_check.txt`）", "", "```"] + [l.rstrip() for l in open(fp)] + ["```"]
    if os.path.exists(ev):
        G += ["", open(ev).read().rstrip()]
    open(os.path.join(L.REP, "RUN_LOG.md"), "w").write("\n".join(G) + "\n")
    print("reports written:", {m: (c["COMPLETE"], c["INCOMPLETE"], c["FAILED"]) for m, c in cnt.items() if not L.METHODS[m].get("blocked")})


if __name__ == "__main__":
    main()
