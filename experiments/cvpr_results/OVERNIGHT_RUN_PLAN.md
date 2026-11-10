# OVERNIGHT_RUN_PLAN — 2026-11-10 的执行计划（写于 05:12，随运行更新见 `report/RUN_LOG.md`）

冻结协议不变：模拟器、起点、合法动作、官方覆盖、D1 策略文件、T0 权重都不改；新增的只有 `experiments/cvpr_results/scripts/`（Table 3 的薄封装、汇总与出图脚本）、Table 1 注册表的新条目和队列分支。

## 1. 任务与优先级

| 优先级 | 任务 | episode 数 | 运行方式 | 状态（05:12） |
|---|---|---|---|---|
| P0 | Table 1 / 2、Fig 3 / 5 的汇总与出图（T0 + D1、MAGICIAN 已齐） | 0（只汇总） | CPU，`scripts/make_tables.py`、`make_figures.py` | 首版已生成，随新记录重跑 |
| P0 | Table 3：V3-T0-K4D1、V3-T0-K1D2（先）、V3-T0-K1D1（后） | 3 × 70 = 210 | Table 1 队列，job 3489889 内 10 个 worker | 运行中（04:43:42 起） |
| P0 | Fig 4：候选池单状态前向 + 背面检查 | 6 场景 × 5 起点 × 6 步 = 180 个状态 | `scripts/fig4/fig4_scan.py`、`fig4_backface.py` | 运行中（05:05 起） |
| P1 | Table 1 基线 Random Walk（adapted） | 70 | Table 1 队列，GPU 0 上 2 个 worker | 运行中（04:51 起） |
| P1 | Table 1 基线 MACARONS（adapted） | 70 | Table 1 队列，Table 3 第一阶段结束后启动，每卡 ≤ 2 个 worker（单进程显存 23 GB） | 待启动 |
| P2 | K = 1 噪声复核：噪声图 1 / 2 / 3 × depth {2, 1} | 6 × 70 = 420 | 同上；顺序 N1(depth 2, 1) → N2 → N3，跑到时间窗口结束为止 | 待启动 |
| — | SCONE、FisherRF | — | — | BLOCKED（`OVERNIGHT_AUDIT.md` §4） |
| — | Ray Encoder 消融、runtime 对照 | — | — | 不在本轮（P2 之后） |

## 2. Table 3 的实现与回归（质量关口，正式运行前完成）

- 封装：`scripts/table3/{sim_out_kd.py, kd_out_planner.py, kd_policy.py, kd_config.yaml}`（KD-v1）。冻结的 `d_policy.py` / `h_policy.py` 只读导入。
  - depth 2：直接安装冻结的 D-v1 策略 `d1`（冻结的 `d_config.yaml`，握手里的 config / policy sha 与 Table 1 的 V3-T0-D1 行相同）。
  - depth 1：策略 `depth1`（KD-v1，仅 `depth` 与 d1 不同）。分数 = `d_policy` 的一步值 `s_one` = 对 K 个样本的 `value_cov` 新覆盖值取平均（lambda 0）；不预测任何第二层视角；**不是** H0 的 DistP1。
  - K：规划器从 arm 表读 K；噪声图按 k 独立播种（`seed_key(scene, start, rep, t_r, k)`），所以 K = 1 用的就是 K = 4 噪声组的第 0 张图。`--noise j`（仅 K = 1）改用第 j 张图。
- 回归（liberty 起点 0，`work_dirs/cvpr_results/overnight_20261110/regression{,2}/`，`jobs/check_regression.py`）：

| 运行 | 对照的 Table 1 记录 | 执行位姿 | 官方覆盖曲线 |
|---|---|---|---|
| T0，K4 / depth 2（两轮：封装初版与终版） | V3-T0-D1 liberty_s0 | 逐位相同 | 逐位相同 |
| T1 权重，K4 / depth 1（两轮） | V3-H1 liberty_s0（一步 value_cov 贪心） | 逐位相同 | 逐位相同 |
| T0，K1 / depth 1、K1 / depth 2（冒烟） | — | 跑通 100 步 | t = 0 各候选的样本值与 K4 的第 0 个样本相对差 ≤ 2.5%（K4 四个样本之间的离散度 12–75%） |
| T0，K1 噪声图 1、3（冒烟） | — | 跑通 | t = 0 的样本值最接近 K4 的第 1 / 第 3 个样本（相对差 0.021 / 0.013，对其他样本 0.65–8.1） |

  说明：K = 1 与 K = 4 第 0 个样本**同一噪声种子但不是逐位相同**，因为 bf16 自动混合精度下批大小不同（每视角 1 个 vs 4 个样本）产生舍入级差异，个别临界射线的命中 / 不命中会翻转。这是真实的 K = 1 推理路径，没有做任何补偿。

## 3. 资源布局（分配 3489889，gpu114）

| GPU | 第一阶段 | 之后 |
|---|---|---|
| 0 | dunnottar 链（15 条，独占：规划器峰值 30.5 GB）+ 2 个 Random worker（各 1.5 GB） | dunnottar 结束后转入其余场景；Random 结束后给 MACARONS |
| 1–3 | 各 3 个 Table 3 worker（非 dunnottar 的 195 条）；Fig 4 扫描挤在 GPU 1 / 2 的空余显存里 | MACARONS（每卡 ≤ 2）→ K = 1 噪声复核 |

- 每条 episode：闭环 → `t1_out_cov.py` 官方覆盖重放（不逐位相同即 STOP）→ `import_episodes.py --new` 写规范记录；noclobber claim，失败自动释放重试（≤ 3 次），已完成的记录被跳过，所以任何时候重跑启动脚本只会补缺。
- 实测速率：约 0.8–1 条 / 分钟（模拟器渲染与重放占主导，与 K、深度关系不大）。排期估算：Table 3 三格 ≈ 4–4.5 h（约 09:00 完成）；Random ≈ 06:20；MACARONS ≈ 1.5–2 h；噪声复核 420 条按此速率需 8–9 h，**窗口内只能完成一部分**，完成多少报多少。
- 不承诺 8 小时内全部完成；每完成一批即重跑 `scripts/run_all.sh` 更新图表，所有图表都带 n_completed / n_expected。

## 4. 停止 / 标注规则

- 任一 episode 的官方重放与存档曲线不同 → 该条不导入（FAILED），不参与任何统计。
- Table 3 四格若不是同样的 14 个完整场景 → 表格标 "preliminary subset"，只在四格共同完成的场景上做配对。
- 方法未跑满 70 条 → Table 1 / Fig 3 / Fig 5 中不画、不写数值，只在 CSV 里给出部分均值与完成数。
- 论文表的列（AUC@20 / AUC@50 / AUC@100 / Cov@20 / Final）由计划书预先指定，不按结果调整。
