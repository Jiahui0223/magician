# Experiment U — Generative Disagreement for NBV

任务书原文：`TASK_BOOK.md`（逐字保存）。问题：在同一个 V3-T0@40K 模型、同一套候选与模拟器下，把规划打分从“预期新增覆盖”换成“K = 4 个生成首次命中假设之间的分歧”，主动建图（尤其前 20 步）会不会更快？

V3-T0 生成的是**视角条件化的首次命中深度假设**，不是完整三维场景；所有分歧都在这些深度及其反投影点上计算。分歧不是经过校准的信息增益。

## 当前状态（2026-11-13）
| 阶段 | 状态 | 文件 |
|---|---|---|
| A 审计 | ✅ | `reports/U0_PIPELINE_AUDIT.md` |
| B / C 设计（冻结 U-v1） | ✅ `configs/u_config.yaml` sha256 `8a62ffec…`，在任何模型运行之前写入 | `reports/U0_PIPELINE_AUDIT.md` §B–§C |
| C 实现 | ✅ 规划器适配层 + 冒烟 | `scripts/u_policy.py`、`u_metrics.py`、`u_out_planner.py`、`sim_out_u.py`、`u_out_cov.py` |
| D 固定状态诊断 | ✅ 755 状态，回归 755/755 逐位；U-best = U2；U4 规则 FAIL（未定义）；三个 U 都显著不如 C1 | `reports/U1_FIXED_STATE_ANALYSIS.md`；`scripts/u_fixed.py`、`u_analyze_fixed.py`、`u_fig_cases.py`；结果 `results/fixed/`，图 `figures/` |
| 冒烟 | ✅ C1 / U1 / U2 / U3 各 15 步（liberty s0），C1 与归档 V3-T0-K4D1 逐位一致 | `work_dirs/exp6_v3/experiment_U/smoke/*/result.json` |
| E 全闭环 | ✅ U1 / U2 / U3 各 70/70（05:57–07:36，0 失败，回放 210/210 逐位一致）；三个 U 在 AUC@20 / AUC@100 上都 0/14 场景优于 C1 | `results/closed_loop/`、`figures/U1_coverage_curves.png`、`U1b_coverage_curves_with_random.png` |
| 最终报告 | ✅ | `reports/EXPERIMENT_U_FINAL_REPORT.md` |

## 目录
| 路径 | 内容 |
|---|---|
| `configs/u_config.yaml` | 冻结协议 U-v1：指标定义、U3 归一化、U-best 规则、U4 规则 |
| `scripts/u_metrics.py` | U1 / U2 / U3（+ 离线诊断、跨视角编号一致度）纯函数 |
| `scripts/u_policy.py` | 规划器适配层：冻结 d_policy + 一步前瞻 + 只换分数 |
| `scripts/u_out_planner.py`、`sim_out_u.py` | 闭环入口（magician 环境模拟器 + seen2scene 规划器子进程） |
| `scripts/u_out_cov.py` | 官方覆盖回放（t25_out_cov 原样） |
| `scripts/u_smoke.sh` | 冒烟 / 回归任务 |
| `scripts/u_fixed.py`、`u_fixed_worker.sh` | Phase D 固定状态回放（复用 CT / F 回放链，V3-T0 + D1，离线 GT 归因） |
| `scripts/u_analyze_fixed.py`、`u_fig_cases.py` | Phase D 统计、Table U3、图 |
| `scripts/u_episode.sh`、`make_queue_tasks.py`、`u_collect.py` | Phase E：单 episode 任务（幂等、跳过已完成、原子认领）、入队生成器、汇总与统计 |
| `work_dirs/exp6_v3/experiment_U/` | 大文件：`smoke/`、`fixed/{states,arrays,logs}`、`closed_loop/<policy>/<scene>_s<k>/`、`staging/` |

## 公平性约束（与 D1 相同的部分）
V3-T0@40K 权重（sha `7ed0ddd7…`）、VAE / 射线编码器 / U-Net、K = 4、Euler 10、R10、P1 / TSDF 更新（b2）、模拟器合法候选与 GT 碰撞检测（只在模拟器侧）、共享噪声键、`m_score.choose`（含全 0 回退）、官方覆盖与逐位回放、每场景 5 个官方起点、100 步。只有分数（和一步 / 两步）不同。GT 只用于离线标签和最终覆盖评价。

## 变更日志
- 2026-11-13 02:37 目录与任务书建立（用户）。
- 2026-11-13 02:45 冻结 `configs/u_config.yaml`（U-v1）；写 `u_metrics.py`、`u_policy.py`、闭环入口。
- 2026-11-13 （API 认证中断约 2.5 h，期间无任何进程运行，无文件改动）。
- 2026-11-13 05:23 冒烟 `expU_a_smoke_{c1,u1,u2,u3}_liberty_s0`（15 步）全部通过；C1 与 V3-T0-K4D1 逐位一致。
- 2026-11-13 05:24 Phase D 测试（conf_liberty_s0 t ≤ 5）通过，D1 分数与 EXP9 Table 4 V2 T0 行逐位一致。
- 2026-11-13 05:26 Phase D 第 1 次正式运行（8 个队列任务 `expU_c_fixed_w*`）；05:28 按 PID 中止，为了加入**离线 GT 归因**（回答任务书 §5 问题 3；只写入诊断字段，不改任何分数、指标定义或协议）。已认领文件清除，未保留任何部分结果。
- 2026-11-13 05:28 `u_episode.sh` 测试（10 步、测试目录 `test_phaseE/`）通过，第二次运行正确跳过。
- 2026-11-13 05:30 Phase D 第 2 次正式运行（`expU_c_fixed2_w1..8`）；05:53 全部结束（54/54 轨迹，755 状态，0 失败，≈ 3.1 GPU-h）。
- 2026-11-13 05:55 Phase D 汇总：`u_analyze_fixed.py`、`u_fig_cases.py`；图例选例规则 C 在第一稿选中全 0 退化状态后加了“oracle 最大值 > 0 且 U-best 非全 0”的条件（只影响图，不影响任何数字）。U-best = U2（预注册规则）；U4 规则 FAIL（同 / 不同编号一致度 0.349 / 0.318，比值 1.10 < 1.25）→ U4 未定义。
- GPU 用量（Phase D + 冒烟 + 测试，含中止的第 1 次运行）≈ 3.5 GPU-h，全部经共享队列（任务名 `expU_*`）。

## Phase E 启动方式（2026-11-13 05:58 已由协调者按此执行）
```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
PY=/bsuscratch/jiahuizhang/envs/seen2scene/bin/python
S=experiments/exp6_v3/ablation_study/experiment_U_generative_disagreement/scripts
$PY $S/make_queue_tasks.py --policies u1,u2,u3            # 预演：只写 work_dirs/exp6_v3/experiment_U/staging/phaseE/，不入队
$PY $S/make_queue_tasks.py --policies <U-best> --enqueue  # 先入队 U-best（70 个任务 expU_e_<p>_<NN>_<scene>_s<k>）
$PY $S/make_queue_tasks.py --policies u1,u2,u3 --enqueue  # 再入队其余（已入队 / 已完成的自动跳过）
# 可选：C1 用本适配层重跑（不需要：C1 = 归档 V3-T0-K4D1，适配层已逐位复现）
# $PY $S/make_queue_tasks.py --policies c1 --enqueue
$PY $S/u_collect.py --policies u1,u2,u3                    # 任何时候都可运行：Table U1 / U2、配对 bootstrap、逐场景胜负、完整曲线、Figure U1
```
- 一个任务 = 一条 100 步 episode + 官方覆盖回放；已完成（回放逐位一致、101 个值）的 episode 跳过；`mkdir` 原子认领；未完成的部分结果移到 `_failed/<policy>_<scene>_s<k>_try<n>` 后重跑（最多 3 次）。
- 成本估计：一步策略每条 ≈ 5.6 min 闭环（归档 V3-T0-K4D1 在共享 GPU 上的实测均值）+ 1.3 min 回放 ≈ 6.9 min → 每个策略 70 条 ≈ 8 GPU-h；U1 + U2 + U3 ≈ 24 GPU-h，8 张卡约 3 h。dunnottar 规划器峰值约 30 GB，队列本来就是一卡一任务。

## 变更日志（续）
- 2026-11-13 05:58 协调者用 `make_queue_tasks.py` 入队 Phase E（U1 / U2 / U3 × All-14 × 5 起点 = 210 条）；07:36 全部完成，0 失败；代码为提交 `a4ed5a8`（之后 `scripts/`、`configs/` 无改动；210 条规划器日志的 config / u_policy / u_metrics sha 一致）。
- 2026-11-13 协调者运行 `u_collect.py --policies u1,u2,u3`（rc 0，问题 0）→ `results/closed_loop/`、`figures/U1_coverage_curves.png`。
- 2026-11-13 新增 `scripts/u_final_extras.py`（只读 CPU 分析：Random 参照、逐场景表、规划器行为、早期曲线、Figure U1b）→ `results/closed_loop/final_extras.json`、`per_scene_with_random.csv`、`figures/U1b_coverage_curves_with_random.png`；不改动任何已有结果。
- 2026-11-13 写 `reports/EXPERIMENT_U_FINAL_REPORT.md`（任务书 §10 的 12 项）。结论：分歧不优于覆盖；U-best（U2）− C1：AUC@20 −0.141、AUC@100 −0.168、Final −0.109（场景 CI 均不含 0）；固定状态诊断准确预测了闭环排序。

