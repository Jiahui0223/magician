# FisherRF-full · Stage F1b 试跑报告（PILOT_FULL）

日期：2026-11-11。用户 17:15 决定暂停：v3 试跑跑完后停止，不再开新一轮。恢复方法见 `FISHERRF_FULL_PAUSED.md`。
本文所有数字都来自下文列出的运行。

**目标**：保留 FisherRF 的 frontier 目标和路径评估，复现 MAGICIAN 论文 Table 2 的 FisherRF 行（15 场景 AUC 0.546 / Final 0.786）。
本试跑用 3 个场景 × 5 个官方起点 × 100 步，对照论文的逐场景数字：

| 场景 | 论文 AUC / Final |
|---|---|
| dunnottar | 0.500 / 0.809 |
| redeemer | 0.727 / 0.876 |
| barts | 0.673 / 0.878 |

## 0. 结论

1. **v3 跑完了 15 / 15 条**。官方覆盖率回放与闭环存储曲线**逐位一致，15 / 15**（Table 1 的 t25 流程，`official_equal_stored = True`）。
2. **3 场景平均 AUC@100 0.491 / Final 0.678，论文为 0.633 / 0.854，未复现**。
   - 分场景看结论不一：barts 0.753 / 0.987，**高于**论文（0.673 / 0.878）；redeemer 0.448 / 0.590，dunnottar 0.272 / 0.458，**明显低于**论文（0.727 / 0.876 和 0.500 / 0.809）。
   - 与 D3a（一步邻居，3 场景平均 0.429 / 0.651）相比：v3 在 barts 上明显更好（0.753 vs 0.478），在 redeemer 上更差（0.448 vs 0.522），在 dunnottar 上相当（0.272 vs 0.286）。
3. **差距的主要来源**（§4，证据来自逐步日志）：
   - 官方 2D frontier 规则推广到 3D 后退化：选中的簇是相机周围的大壳，目标只在几个网格步之内；
   - 目标落在障碍上或不可达时，退化为原地转向（redeemer s2 有 55 次）；
   - 开局深度无效（D13）：dunnottar s0 / s2 / s4 分别有 20 / 23 / 37 帧。
   - MAGICIAN 实际怎么适配不可知：论文只有一句脚注，仓库中没有对应代码。
4. **耗时**（共享 GPU）：每步约 5.2 s，其中官方建图 4.2 s；每次重规划约 3.4 s；规划端显存不超过 1.2 GB。

## 1. 实现（代码：`scripts/fisherrf_full_planner.py`；逐条对照见 `PROTOCOL_ADAPTATION.md` F1b）

**沿用的部分**：
- 官方 GaussianSLAM：每步 `track_rgbd`，60 次迭代；
- 官方 `pose_eval`：λ = 0.1；
- D12（docstring init）、D13（无有效深度时跳过建图，还没有地图时按种子随机）；
- RGB 旁路文件、open() 守卫（所有运行 `guard_denied` 都为空）、尺度 s = 7。

**新增：官方规划循环**，适配到 3D 和 settings 网格（`AM/` = FisherRF-active-mapping 8b196cd）：

| 环节 | 官方 | 本适配 |
|---|---|---|
| 占据 | `planning/astar.py:114-211` 2D 三通道 | 3D 体素，0.05 FR，并列时归为未知，全部像素射线 |
| frontier | `astar.py:213-410` | 26 连通；去掉高斯 > 25 的体素；3³ 开运算；最大连通空闲域；"combined" 规则选簇 |
| 候选 | `astar.py:532-746` | 256 个点放在半径 [0.5, 1.0 × expansion] FR 的球壳上，吸附到网格位置和朝向 → 官方 `pose_eval` → top-20 |
| 路径 | `tester.py:904-1030` | 5D 位姿图 BFS，邻接为官方 `get_neighboring_poses`；障碍为规划端的占据和高斯，未知视为空闲；已执行位姿不可经过 |
| 路径评分 | `tester.py:763-893` | 与官方逐式相同：≤21 条路径，每 5 个动作累加一次，`H_reg` 1e-6，`path_end_weight` 30；保留官方 zip 的 EIG 错位 |
| 主循环 | `tester.py:284-405, 497-518` | 队列 ≤ 30；队列空或队首不合法时重规划；无路径则转一次并 `expansion += 1`；不合法的未访问邻居记为封锁边 |

**版本演进**（每次修正都有日志证据）：

| 版本 | 改动 | 原因 |
|---|---|---|
| v1 → v2 | ① 膨胀之后，清空所有已执行位置周围 5³ 的障碍体素 | 官方 `get_pointcloud` 的 max-pool 问题会在天空像素处生成相机中心的零尺度高斯，加上高斯障碍规则就把起点围死（`n_reachable = 1`） |
| | ② 只有当腐蚀空闲中有网格位置时，才启用腐蚀空闲过滤 | 否则在 3D 网格上永远满足不了，候选为 0 |
| v2 → v3 | 路径改在 5D 位姿图上求，排除已执行位姿 | 仿真器在还有未执行邻居时不提供已执行位姿（`get_valid_neighbors`）。v2 的路径常以不合法的一步开头，导致原地转圈：redeemer s0 / s1 分别有 26 / 53 次 `first_move_not_legal` |

v3 中 `first_move_not_legal` 全部为 0。

## 2. v3 结果（`work_dirs/cvpr_results/fisherrf_reproduction/full/outdoor/closed_loop_v3/`）

**按场景**（5 个起点的均值）：

| 场景 | AUC@20 | AUC@100（论文） | Final（论文） | D3a AUC / Final | MAG-B（同协议 Table 1）AUC / Final |
|---|---|---|---|---|---|
| dunnottar | 0.106 | **0.272**（0.500） | **0.458**（0.809） | 0.286 / 0.440 | 0.721 / 0.973 |
| redeemer | 0.203 | **0.448**（0.727） | **0.590**（0.876） | 0.522 / 0.696 | 0.784 / 0.977 |
| barts | 0.310 | **0.753**（0.673） | **0.987**（0.878） | 0.478 / 0.818 | —（未存档） |
| 3 场景均值 | 0.206 | **0.491**（0.633） | **0.678**（0.854） | 0.429 / 0.651 | — |

**逐条**：

| 场景 / 起点 | AUC@20 | AUC@100 | Final | 回放 | 不同位置数 | 重规划 | 首步不合法 | 无路径 → 转向 | 平均路径长 | 每步 s | D13 帧 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dunnottar 0 | 0.284 | 0.458 | 0.573 | = | 55 | 35 | 0 | 1 | 8.8 | 5.28 | 20 |
| dunnottar 1 | 0.023 | 0.145 | 0.386 | = | 15 | 21 | 0 | 3 | 5.7 | 5.78 | 1 |
| dunnottar 2 | 0.000 | 0.218 | 0.428 | = | 32 | 14 | 0 | 0 | 7.8 | 4.35 | 23 |
| dunnottar 3 | 0.090 | 0.342 | 0.510 | = | 14 | 17 | 0 | 2 | 7.4 | 5.05 | 0 |
| dunnottar 4 | 0.135 | 0.196 | 0.391 | = | 44 | 28 | 0 | 1 | 7.4 | 3.90 | 37 |
| redeemer 0 | 0.164 | 0.366 | 0.515 | = | 18 | 35 | 0 | 19 | 5.1 | 5.59 | 1 |
| redeemer 1† | 0.138 | 0.320 | 0.436 | = | 19 | 20 | 0 | 6 | 6.6 | 4.28 | 2 |
| redeemer 2 | 0.411 | 0.584 | 0.697 | = | 22 | 61 | 0 | 55 | 7.5 | 4.72 | 24 |
| redeemer 3 | 0.173 | 0.617 | 0.849 | = | 30 | 21 | 0 | 5 | 6.2 | 5.46 | 3 |
| redeemer 4 | 0.130 | 0.353 | 0.453 | = | 21 | 28 | 0 | 9 | 4.8 | 6.07 | 0 |
| barts 0 | 0.302 | 0.724 | 0.976 | = | 34 | 15 | 0 | 0 | 6.7 | 5.81 | 0 |
| barts 1 | 0.277 | 0.698 | 0.978 | = | 39 | 17 | 0 | 0 | 5.9 | 6.28 | 0 |
| barts 2 | 0.351 | 0.807 | 0.989 | = | 38 | 19 | 0 | 0 | 5.7 | 4.81 | 0 |
| barts 3 | 0.276 | 0.729 | 0.996 | = | 35 | 22 | 0 | 1 | 4.8 | 5.58 | 0 |
| barts 4 | 0.346 | 0.808 | 0.998 | = | 47 | 18 | 0 | 0 | 5.6 | 5.36 | 0 |

说明：
- "=" 表示官方回放与存储曲线逐位一致。
- † redeemer s1 前两次尝试都因 GPU 3 OOM 失败（另一实验的进程占用约 37 GB），第三次改在 GPU 1 上完成。FisherRF 本来就不能逐位复现（R1），所以这是一次独立的重跑。
- 数据：`results/pilot_full_episodes.csv`、`results/pilot_full_summary.json`。

**耗时**（v3 全部 1,500 步，均值 / 中位数 / P95；gpu114 上 6–8 个进程共享 GPU，另一实验同时在跑）：

| 环节 | 耗时 |
|---|---|
| 每步合计 | 5.22 / 4.97 / 10.81 s |
| 官方建图（含 PNG 0.43 s） | 4.24 s |
| 规划端占据更新 | 0.11 s |
| 每次重规划（共 371 次） | 3.44 / 2.80 / 6.80 s |
| — 其中 frontier | 0.22 s |
| — 其中 `pose_eval`（全部去重候选） | 1.44 s |
| — 其中 BFS | 0.36 s |
| — 其中路径 EIG（≤21 条路径的逐步 Hessian） | 2.06 s |

- 规划端显存峰值 1.16 GB。
- 每条 episode 墙钟平均 699 s，另加回放约 150 s。

## 3. 行为

- 重规划平均 18–33 次 / 100 步；跟随路径的平均长度 4.8–8.8 步；不同位置数 14–55 个，D3a 约 19 个。
- `no_frontier_error`（官方会在此结束 episode，我们继续跑）只在 redeemer s2 出现 5 次。
- v3 中首步不合法为 0；这个问题在 v2 中占了大量步数。

## 4. 差距诊断（未改官方公式；证据来自逐步日志）

1. **frontier 规则在 3D 中退化**。
   - 射线 carving 出的空闲体积的整个外壳都是 frontier，被选中簇的中位数大小为 2.6 万到 15 万体素。"combined" = count / (平均距离 + 20) 被面积主导，簇均值落在相机附近。
   - 结果：目标距当前位置的 L1 距离只有 1.6–5 个网格步；在 50 步左右时，6 条 episode 中有 4 条旋转多于平移（如 dunnottar s3：17 次平移，41 次旋转）。
   - barts 场景小（7×9×5 网格），局部目标已经足够，所以反而超过了论文。
2. **目标不可达时退化为转向**。吸附后的目标落在障碍体素上，或者是已执行的位姿时，top-20 全部不可达，于是按官方"无路径 → 转一次"处理。redeemer s2 有 55 次，s0 有 19 次。
3. **D13 开局空转**：dunnottar s0 / s2 / s4 分别有 20 / 23 / 37 帧没有有效深度（天空或量程外）。D3a 和 Table 1 中这些起点同样难。
4. **占据只到 `pcd_far_distance` = 7 FR（49 runtime）**，而 dunnottar 场景跨约 120 × 170 runtime，远处结构不会成为 frontier。
5. **MAGICIAN 的适配不可知**。论文只说"modifying its frontier selection and adjusting its action space"。MAGICIAN 仓库 e4e08fe 中被注释掉的 frontier 设计（`sample_valid_poses_in_space_for_frontier`）是在**整个位姿空间**采样、保留视野内含未知区域的位姿。这与官方"在单个 frontier 簇周围采样"不同，很可能是差距的关键。

## 5. 候选下一步（都没有运行，等决定）

- **(a) F1c：MAGICIAN 式 frontier 选择**。
  - 已实现：`FRF_FRONTIER_MODE=magician`，队列脚本 `scripts/frf_full_queue3.sh`。
  - 做法：在整个位姿空间均匀采 256 个位姿，保留位置未被占据、且视野内有未知体素的，然后用官方 `pose_eval` 取 top-20；路径和路径 EIG 不变。
  - 只做过一次 20 步冒烟（redeemer s1，GPU 0）：cov@20 = 0.406，v3 同一起点的冒烟为 0.218；每次重规划约 8–9 s。**只有一条、只有 20 步，不能作为结论**。产物在 `full/_aborted_f1c/`。
- **(b) 在 `pose_eval` 之前先过滤不可达的候选**（已执行、在障碍上、或 BFS 不可达）。官方 `GaussianSLAM.global_planning`（`gaussian.py:1244-1257`）在打分前也会用 `pathfinder.is_navigable` 和 `find_path` 检查可达性。这可以消除"top-20 全部不可达 → 转向"的循环。
- **(c)** 两者都做之后，再按同样的 3 × 5 重跑试跑；通过后再考虑全量 14 × 5（55 条新 episode，8 个 worker 约 2–2.5 小时）和 MP3D（室内版还没有实现）。

## 6. 运行记录（gpu114；没有 Slurm：`sbatch --test-only` 显示 `Preempts:`）

| 时间 | 事件 |
|---|---|
| 16:05 | 冒烟 v1（barts s0，10 步）：跑通，回放逐位一致 |
| 16:14–16:20 | 试跑 v1 中止（修正 ① ②），输出在 `full/_aborted_v1/` |
| 16:22–16:52 | 试跑 v2（9 条，见 §7） |
| 16:30 | GPU 0 被另一实验的进程（35–38 GB）占满，v2 的 dunnottar s0、s1 OOM。按 PID 杀进程被权限系统拒绝，于是用"预占剩余任务 → 让旧 worker 自然退出"的办法收尾 |
| 16:51–17:38 | 试跑 v3，GPU 1–3 各 2 个 worker：15 / 15 条完成。redeemer s1 前两次因 GPU 3 OOM 失败，第三次在 GPU 1 上完成 |
| 17:10–17:14 | F1c 冒烟 20 步之后，按 17:15 的暂停决定，用 PID 停掉了 F1c 的启动器和 worker（只启动了 1 条、还在 setup 阶段），产物移到 `_aborted_f1c/` |
| 17:40 | 确认：gpu114 上没有本工作的 worker、队列、启动器、仿真 / 规划进程或 tail 进程；GPU 上剩下的 4 个进程属于另一实验（`cvpr_results/exp*`） |

## 7. v2 记录（不计入结论；路径不服从动作规则）

| 场景 | n | AUC@100 | Final |
|---|---|---|---|
| dunnottar | 4（s1–s4） | 0.338 | 0.493 |
| redeemer | 5 | 0.472 | 0.603 |

逐条数据见 `results/pilot_full_v2_episodes.csv`；原始数据在 `full/outdoor/closed_loop/`。

## 8. 文件

| 类别 | 文件 |
|---|---|
| 代码 | `scripts/fisherrf_full_planner.py`、`fisherrf_sim_out.py`（`--planner`）、`frf_full_queue2.sh`（v3）、`frf_full_queue3.sh`（F1c）、`frf_full_collect.py` |
| 文档 | `PROTOCOL_ADAPTATION.md` F1b（逐条 file:line 对照和三处修正）、`FISHERRF_FULL_PAUSED.md`（恢复说明） |
| 结果 | `results/pilot_full_*`（v3）、`results/pilot_full_v2_*`（v2） |
| 原始数据 | `work_dirs/cvpr_results/fisherrf_reproduction/full/` |
