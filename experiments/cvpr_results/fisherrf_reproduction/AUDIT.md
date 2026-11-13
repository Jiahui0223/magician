# FisherRF 复现 · 独立审计（AUDIT）

日期 2026-11-13。审计对象：Macarons++ 上的 FisherRF-full（f1c4，All-14 AUC@100 0.367 / Final 0.543）与 D3a，对照 MAGICIAN 论文 FisherRF 行（同 14 场景 0.541 / 0.766）。
范围：代码审查（对照官方 FisherRF-active-mapping 8b196cd）+ 在已记录状态上的小型诊断。**没有新的闭环运行**，没有修改官方代码、已有脚本或已有结果目录。
GPU：只经共享队列 `work_dirs/cvpr_results/gpu_queue`（任务 `frfaudit_*`），合计约 0.8 GPU-h。
新脚本：`scripts/audit/`（a1–a6）；输出：`work_dirs/cvpr_results/fisherrf_reproduction/audit/`。

**结论分级**：「已确认」= 有直接测量；「可能」= 机制成立但对闭环 AUC 的量没有测；「排除」= 测量表明不是原因。

---

## 0. 结论摘要

1. **没有找到会改变结论的实现 bug。** 位姿 / 内参 / 尺度约定、D12、RGB–深度对齐、EIG 计算、官方配置覆盖项、路径 EIG 公式都核对过，或已排除（§2）。
2. **根本问题在打分本身（已确认）。** 在 8 条已记录 episode 的 40 个固定状态上，用 GT 离线算每个候选视角的真实覆盖增益：
   - **FisherRF EIG 选出的视角不比随机好**：
     - 一步合法邻居：0.27 pp，随机 0.84 pp；
     - 全网格候选：2.73 pp，随机 3.09 pp。
   - 一个最朴素的 frontier 计数（视野内从未观测体素数）分别为 2.85 pp 和 9.26 pp（图 `audit/probe_f1c4/fig_scorer_top1.png`）。
   - 在 FisherRF 的"去幽灵 / 去大高斯 / 改 λ"变体中，同样没有一个超过随机。
   - 这个固定状态结果与闭环排序完全一致：D3a < Random，f1c4 ≈ Random。
3. **原因属于设计层面（已确认）。** EIG 只对地图里已有的高斯计分，看不见未观测空间。在本协议下：
   - 正则 λ = 0.1 远大于 H_train（中位数 1e-5 – 3e-3），所以 EIG ≈ "视野里已有高斯的 Hessian 总量"，对已观测区域没有折扣；
   - 它偏好近距离重看已建图的表面。
   
   官方系统的探索能力来自 **frontier 对候选集的约束**，不是来自 EIG。f1c4 的候选取自整个位姿空间，没有 frontier 约束（只要求视野里有"未知"体素，254/256 个都满足）；v3 / v4 的 3D frontier 又退化了（§2 F5）。所以两者都等价于"无信号打分 + 随机目标"。
4. **地图健康问题（已确认，但不是主因）。**
   - 有天空的户外场景里，少量高斯尺度失控，最大可达 1e3–1e4 runtime 单位，覆盖了旧帧 62–100% 的天空像素，最差帧 PSNR 2–3 dB。
   - 每个户外 episode 还有数百到数千个"相机中心幽灵高斯"。
   - 但在地图健康的场景里（barts、sestino、fushimi s3：PSNR 22–33、深度相对误差 < 1.1%），EIG 同样不比随机好。
   - 去掉幽灵高斯、去掉大高斯或改 λ，排序也都不会变好。
5. **建议：(4) 不再复现，引用论文数值**；本复现作为附录诊断保留。如果必须要同协议的一行，退路是 (3)：预注册一个"frontier 优先、再用 Fisher 打分"的适配，并明确标注为我们的适配（§5）。不建议 (1)、(2)。

---

## 1. 做了什么

| 编号 | 内容 | 脚本 | 输出 |
|---|---|---|---|
| A1 | 规划端几何约定的独立核对：用规划器自己的内参、F_FLIP、w2c、s = 7，把记录里 10% 采样像素的 zbuf 反投影，与仿真端（MAGICIAN 自己的反投影）写下的 `sample_points` 比较；同时统计全部 70 条 f1c4 的深度有效率 | `scripts/audit/a1_geometry_check.py`（CPU） | `audit/a1_geometry.json` |
| A2 | 规划端探针（无 GT）：从 svc 记录和 RGB 旁路文件按 `fisherrf_full_planner.py` 的方式重建在线地图（官方 GaussianSLAM、同样的覆盖项、D12、D13、同种子；占据更新逐行照抄；不逐位，见 R1），在 t = 5 / 15 / 30 / 50 / 75 上记录：<br>• 地图健康：在已执行位姿上重渲染，并预测下一帧；<br>• 占据统计；<br>• 候选打分：合法邻居 + 48 个均匀网格位姿 → 官方 `pose_eval` 及其逐高斯分解 | `scripts/audit/a2_probe_planner.py` | `audit/probe_f1c4*/<ep>.json` |
| A3 | GT 端（离线，只做诊断）：冻结 EpisodeSim 回放同一 episode，官方覆盖率逐步等于记录值；在同一状态上用冻结的 `m_gain_eval` 计算每个候选的真实增益。**校验：40 / 40 个已执行动作的增益与记录中的覆盖率增量逐位相等** | `scripts/audit/a3_gt_gain.py` | `audit/probe_f1c4/<ep>_gt.json` |
| A4 | 行为统计：70 条 × {f1c4、v4、v3、D3a、MAG-B、RANDOM} | `scripts/audit/a4_behaviour.py`（CPU） | `audit/a4_behaviour.json` |
| A5 / A6 | 汇总与作图 | `scripts/audit/a5_analyze.py`、`a6_figure.py` | `audit/probe_f1c4*/a5_summary.*`、`fig_scorer_top1.png` |

**探针 episode**（f1c4 记录，覆盖好 / 坏、有天空 / 无天空）：

| episode | f1c4 AUC@100 |
|---|---|
| barts s0 | 0.742 |
| redeemer s1 | 0.608 |
| colosseum s0 | 0.117 |
| dunnottar s3 | 0.289 |
| fushimi s3 | 0.065 |
| sestino_museum s0 | 0.255 |
| eiffel s0 | 0.252 |
| bridge s0 | 0.290 |

**补充重跑**（只跑规划端，GT 增益按候选 key 复用；候选集由种子决定，逐个相同）：
- `probe_f1c4_h`：H_train 量级与 λ 变体；
- `probe_f1c4_scale`：高斯尺度与漂浮物归因；
- `probe_f1c4_big`：大高斯对 EIG 的占比。

---

## 2. 发现（按严重程度）

### F1【严重 · 设计层面 · 已确认】在本协议下，FisherRF EIG 没有覆盖率信号；f1c4 / v3 / v4 都缺少一个有效的 frontier 约束

**证据 1：固定状态打分检验**（A2 + A3，`audit/probe_f1c4/a5_summary.txt`，8 条 episode、40 个状态）。
- Top-1 = 按该打分取 argmax 的视角的真实增益（覆盖率 pp），平局取期望。
- 方括号内为按状态 bootstrap 的 95% 区间。

| 决策规则 | 合法邻居（D3a 决策集，34 个状态） | 48 个均匀网格位姿（f1c4 候选分布，40 个状态） |
|---|---|---|
| Oracle（GT 上界） | 3.27 | 18.75 |
| 视野内从未观测的体素数（frontier 式计数） | **2.85**（较随机 +2.01 [+0.80, +3.40]），ρ 中位数 +0.67 | **9.26**（+6.17 [+4.01, +8.47]），ρ +0.65 |
| 规划端"未知"标签计数（f1c4 过滤量） | 2.83 | 9.79 |
| 随机 | 0.84 | 3.09 |
| **FisherRF EIG（官方 `pose_eval`）** | **0.27（−0.57 [−0.91, −0.28]）**，ρ ≈ 0，仅 12% 的状态 ≥ 随机 | **2.73（−0.36 [−1.43, +0.81]）**，ρ +0.23，25% 的状态 ≥ 随机 |
| EIG 去掉幽灵高斯 | 0.35 | 2.24 |
| EIG 重跑（`probe_f1c4_big`，同 8 条，不逐位） | 0.26 | 2.57 |
| EIG，λ = 1e-6（8 条重跑；5 条的 `probe_f1c4_h` 为 0.47 / 1.41，对应随机 0.80 / 2.52） | 0.49 | 3.02 |
| EIG 不扣 H_train（Σ H_cand / 0.1，8 条重跑） | 0.31 | 2.72 |
| EIG 同时去掉幽灵高斯和尺度 > 3.5 rt 的高斯（8 条重跑） | 0.78 | 2.36 |
| 先按 frontier 计数取前 5，再在其中按 EIG 取最大 | —（候选太少） | 8.79（重跑 9.70）；前 5 中随机选 9.46 |

- 逐 episode 看结论一致，地图健康的 barts、fushimi s3、sestino 也一样（`a5_summary.json` 的 `scorer_rows`）。
- 这一检验**预测对了闭环**：
  - 合法邻居集上 EIG 比随机差 → D3a 闭环 0.301 < Random 0.353；
  - 全网格候选上 EIG ≈ 随机 → f1c4 闭环 0.367 ≈ Random（+0.014 [−0.023, +0.048]）。

**机制**（已确认）：
- `pose_eval` = Σ H_cand ⊙ (H_train + 0.1)⁻¹（官方 `models/SLAM/gaussian.py:1312-1335`）。没有高斯的方向贡献为 0。
- 在均匀候选上：
  - EIG 与可见高斯数的相关为 ρ = +0.88，与渲染出的"新颖度"（silhouette < 0.5 的像素比例）的相关为 ρ = −0.79；
  - 每个状态平均 57% 的候选 EIG ≈ 0（看不到已有地图），而这些候选的平均真实增益（3.11 pp）**高于** EIG > 0 的候选（1.93 pp）。
- EIG 系统性地奖励"重看已建图的表面"，惩罚"看向未观测区域"。官方系统靠 frontier 把候选限制在已知 / 未知边界附近（`planning/astar.py:307-410, 556-603, 704-746`），EIG 只在这些候选之间排序。

**本适配中 frontier 约束的状态**：

1. **f1c4**（`scripts/fisherrf_full_planner.py:384-416`，`candidates_magician`）：
   - 照 MAGICIAN e4e08fe 的 `sample_valid_poses_in_space_for_frontier` 采位姿、算视野内未知数并排序，**但没有用这个排序**：所有计数 > 0 的候选都送进 `pose_eval`，再按 EIG 取 top-20（`:540-547`）。
   - 实测 254/256 个采样都被保留（例：barts s0 t0 的 `planner_logs`）。所以 frontier 约束实际上不存在。
   - 而 MAGICIAN 那段被注释的设计是"按未知计数从高到低，取第一个有 Dijkstra 路径的位姿"（`git show e4e08fe:macarons/testers/frontier_planning.py`，被注释的 `path_record` 块）。也就是说，那里 frontier 计数本身就是选择准则。
2. **v3 / v4**：官方 2D frontier 推广到 3D 后退化（F5）。

**能解释多少差距**：可能是主因。
- 一个不含覆盖信号的决策规则，闭环结果就应该落在 Random 附近，实测正是如此。论文 0.546 比我们协议下的 Random 高约 0.19，这只能来自有信号的决策成分。在固定状态上，有信号的只有 frontier 成分。
- 定量的闭环影响无法在不跑闭环的前提下给出。
- 固定状态上，"先 frontier 取前 5，再按 EIG 取最大"选出的视角平均增益 8.8–9.7 pp，随机 3.1 pp，每步多约 6 pp；但与"前 5 中随机选"（9.46 pp）没有差别。也就是说，增益全部来自 frontier 成分。

**修正方案**（未实施）：
- 见 §5 的 (3)：frontier 计数作为一级准则，取 top-K 后再交给 Fisher 路径评估；
- 未知 / 已观测改用单次观测即可雕刻的标签，而不是官方 argmax 的两次规则。

---

### F2【高 · 适配 / 尺度 · 已确认机制；对排序的影响小】λ = 0.1 是绝对量，在本协议的单位和分辨率下远大于 H_train，EIG 失去"扣除已观测"的作用

**证据**（`probe_f1c4_h`，5 条 episode × 5 个状态）：
- 非幽灵高斯的 H_train 行和中位数为 1.9e-5 – 2.9e-3，99% 分位为 2.7e-3 – 17.8；
- 任一分量超过 λ = 0.1 的高斯只占 0.1–10%；
- 候选 EIG 与"不扣 H_train"的 Σ H_cand / 0.1 之比，中位数多在 0.74–0.99（个别状态 0.36–0.65）。

**原因**：
- H 的量级与像素数、焦距的平方、场景单位都有关。官方是 800×800、fx = 400、米制、约 500 个关键帧（每 4 帧一个，2000 步，`configs/base_config.py:42`）；
- 我们是 456×256、fx = 221.7、s = 7、≤ 100 个关键帧。
- `pose_eval` 里的 0.1 写死在代码中（`gaussian.py:1315`）。

**影响**：
- 改成 λ = 1e-6，或去掉 H_train，Top-1 都没有超过随机：
  - λ = 1e-6：合法邻居 0.49 pp，均匀候选 3.02 pp；
  - 去掉 H_train：0.31 pp 和 2.72 pp；
  - 对照随机：0.84 pp 和 3.09 pp。
  - λ = 1e-6 时 ρ 由 +0.20 升到 +0.30，但 argmax 选出的视角不变好。
- 所以这是一个真实的"适配后语义偏移"，但不是差距的主因。

**修正**（未实施，属于偏离官方）：把 λ 按 H 的量级重新标定，例如取 H_train 非零行的中位数。必须写明这是偏离。

---

### F3【高 · 地图健康 · 已确认；对差距是次要因素】有天空的户外场景里高斯尺度失控，覆盖大片天空

**证据 1：尺度**（`probe_f1c4_scale`，同一组状态）。

| episode | 最大尺度（runtime 单位） | > 3.5 rt 的高斯数 | > 35 rt 的高斯数 |
|---|---|---|---|
| barts（几乎没有天空） | ≤ 7.4 | ≤ 29 | 0 |
| colosseum | 最高 1359 | 最多 655 | 25 |
| dunnottar | 1915 | 460 | 17 |
| eiffel | 565 | 589 | 31 |
| redeemer s1（t = 75） | **11662** | 329 | 55 |

**证据 2：漂浮物归因**。在最差的旧帧上，只渲染"尺度 > 3.5 rt"的高斯（只占全部高斯的 0.5–2%）：
- 它们覆盖了 62–100% 的天空像素（无有效深度的像素）；
- 其余高斯只覆盖 0.7–8%。

**证据 3：在已执行位姿上重渲染**（`probe_f1c4`，各状态均值）。

| 场景类型 | episode | 当前帧 PSNR | 旧帧 PSNR | 旧帧深度相对误差中位数 |
|---|---|---|---|---|
| 无天空 | barts | 28.8 | 28.8 | 0.75% |
| 无天空 | sestino | 27.0 | 32.0 | 1.1% |
| 无天空 | fushimi s3 | 22.5 | 22.1 | 0.84% |
| 有天空 | colosseum | 13.4 | 16.4 | 8.8% |
| 有天空 | eiffel | 14.6 | 15.7 | 3.0% |
| 有天空 | redeemer s1 | 12.3 | 13.4 | 3.5% |

有天空的场景中，最差帧 PSNR 只有 0.7–3 dB。下一帧的预测深度误差：colosseum 中位 55.7%，dunnottar 13.5%。

**直观证据**：官方建图 PNG `full/outdoor/closed_loop_f1c4/colosseum_s0/planner_work/colosseum_s0/mapping/mapping_iter_0031.png`（天空处渲染为大片白灰色团块，silhouette ≈ 1）和 `mapping_iter_0012.png`。

**机制**：官方代码在户外暴露的问题。
- 建图 loss 的 mask 是 `depth > 0`（`gaussian.py:195`），天空和超量程像素（D11）既不受深度监督，也不受颜色监督；
- 删除大高斯要求 `iter ≥ remove_big_after = 100`（`slam_external.py:217-219`，yaml `pruning_dict`），而每次建图只有 60 次迭代，`it` 每次从 0 开始，所以永远不会触发。官方也是如此。
- 室内 Habitat 里几乎所有像素都有有效深度，失控会被 loss 抑制；户外天空占比大（f1c4 的有效像素比例：dunnottar 0.34、redeemer 0.26，A1），失控就不受约束。

**对 EIG 的影响**（`probe_f1c4_big`）：
- 被选中视角（argmax-EIG）的 EIG 一半以上来自尺度 > 3.5 rt 的高斯的状态比例：合法邻居集 12%，均匀候选集 25%。
- 这些"天空团块"会把一部分决策引向天空方向，与 F6 中 100 个无深度的目标位姿相符。
- 同时去掉幽灵高斯和大高斯之后，Top-1 为 0.78 / 2.36 pp，仍不超过随机（0.84 / 3.09）。

**能解释多少**：次要。
- 地图健康的 fushimi s3（AUC 0.065）和 sestino s0（0.255）同样远低于论文；
- 在健康地图上 EIG 也不比随机好（F1）。

**修正**（未实施，属于偏离官方）。可选两种：
- 每次建图后删除尺度 > 若干 FR 的高斯；
- 把无命中的像素作为"空"来监督（silhouette → 0）。

---

### F4【中 · 已确认】get_pointcloud 的 max-pool 掩码问题，在户外产生大量"相机中心幽灵高斯"，并在部分状态主导 EIG

**机制**：官方 `get_pointcloud` 的 mask 先做 4×4 max-pool，深度却取块内左上像素（`gaussian.py:96-104`，下采样 4）。块内有有效像素、而左上像素无效（天空边缘）时，就会在相机中心生成一个尺度为 0 的高斯（log_scale = −inf）。SMOKE_TEST §4.5 已记录过这一点。

**证据**：
- 幽灵高斯数（每个探针状态）：barts 0–1；bridge 1124；eiffel 852；sestino 607；colosseum 440；redeemer 500（各 episode 均值，最多 3439）。
- 被选中视角（argmax-EIG）的 EIG 有一半以上来自幽灵高斯的状态比例：合法邻居集 26%，均匀候选集 15%，两者合并 35%。
- 合法邻居中，10% 的候选有 ≥ 88% 的 EIG 来自幽灵高斯（`a5_summary.txt`）。

**影响**：
- 会改变一部分具体决策；
- 但去掉后 Top-1 并不变好（0.35 / 2.24 pp，见 F1）。

**修正**（未实施）：在适配器侧，每次 `track_rgbd` 之后删除 log_scale = −inf 的高斯（不改官方代码），并记为偏离。

---

### F5【中 · 已确认；影响 v3 / v4 的 frontier 和 f1c4 的过滤】官方占据规则在大步长下要两次观测才能离开"未知"，3D frontier 因此退化

**机制**：
- 官方 2D 规则是 `occ_map += occ / (sum + 1e-5)`，未知先验为 1，按 argmax 判定（`planning/astar.py:200-211`）。一个格子只被观测一次时，"空闲"或"占据"通道恰好 < 1 或 = 1，平局判为未知。我们的 3D 版本照搬了这条规则（`fisherrf_full_planner.py:273-278, 297-299`）。
- 官方的智能体每步只走 0.05 m / 5°，同一区域会被连续观测几十次，这条规则无害。
- 本协议每步跨一个网格（约 1–1.6 FR），大量区域只被看到一次。

**证据**：
- 已被观测的体素中，仍被标为未知的比例：barts 22%、colosseum 32%、bridge 43%、eiffel 58%。
- 这部分恰好等于"只被一帧观测过的体素"（`probe_f1c4` 中的 `occ` 字段）。
- t = 0 时障碍体素数为 0（例：barts s0 t0 的 `n_obst_vox = 0`），因为第一帧看到的表面全部仍是未知。
- 后果见 PILOT_FULL §4.1：v3 / v4 的 frontier 是"相机周围的大壳"，目标只离 1.6–5 个网格步。

**影响**：
- 对 v3 / v4 是 frontier 失效的直接原因之一；
- 对 f1c4 影响不大：规划端"未知"标签计数与真正"从未观测"计数的打分效果几乎相同（F1 表：2.83 vs 2.85，9.79 vs 9.26）。

**修正**（未实施）：frontier 改用"至少被观测一次即为已知"的标签。

---

### F6【中 · 行为 · 已确认】路径评估实际上等于"去 EIG 最大的目标"；f1c4 的轨迹统计与 Random 几乎相同

**路径评估**（f1c4 全部 868 次重规划）：
- 路径项（point-EIG 的对数项）只占总分的中位数 0.03%；
- 92.7% 的决策就是 EIG 第一名的目标。

这与官方公式一致（`tester.py:830-893`，`path_end_weight` = 30），不是 bug。但它意味着"路径评估"没有带来额外信息，中间帧的价值被忽略。

**轨迹统计**（A4，`audit/a4_behaviour.json`，70 条均值）：

| 方法 | 平移 | 转向 | 不同位置数 | 回访位置 | 最大 L1 跨度 | 无有效深度帧 |
|---|---|---|---|---|---|---|
| f1c4 | 56.9 | 43.1 | 41.6 | 16.3 | 11.3 | **12.9** |
| RANDOM | 56.4 | 43.6 | 41.8 | 15.6 | 9.8 | — |
| D3a | 33.3 | 66.7 | 18.8 | 15.5 | 6.3 | 14.9 |
| v3 / v4（15 条） | 46 | 54 | 31–33 | 15–16 | 8.3–8.6 | 7–9 |
| MAG-B | 73.9 | 26.1 | 66.5 | 8.4 | 15.1 | **0.5** |

- f1c4 的 906 个无深度帧中，801 个在路径中途：平移时保持原朝向，常对着天空。另外 100 个就是 EIG 选出的目标位姿本身（共到达目标 567 次）。
- 目标距当前位置的 L1 距离：各 episode 中位数的均值为 5.4 个网格步；72% 的路径到达了目标。

**修正**：不建议单独修。这是 F1 的后果。

---

### F7【低 · 已确认】更常停在网格的"网格内部"位姿

按 MAGICIAN `occupied_pose.pt` 判定相机位置在网格内部的步数比例：
- f1c4 5.9%，D3a 5.1%；
- RANDOM 3.9%，MAG-B 2.1%。

`get_valid_neighbors` 不检查这一点（MAGICIAN `macarons/utility/macarons_utils.py:4228-4256`，有效性检查在 `:4248-4249` 被注释掉），所以对所有方法都一样，只是 EIG 偏好近距离表面，略多。影响小。

---

### F8【背景 · 可能】论文的 FisherRF 很可能不在我们的动作协议下运行

- 论文脚注说 "adjusting its action space"。
- 论文的 Random Walk 是 0.241（Table 2），我们协议下的 Random 是 0.353。而 MACARONS（0.660 vs 0.647）和 MAGICIAN（0.711 vs 0.721）是用官方代码在我们协议下重测的，两者吻合。
- 这说明论文里"非官方代码的基线"的动作空间或协议与我们的不同。因此，即使完全照搬 MAGICIAN 的适配，也不保证在我们的协议下得到 0.546。
- 未验证，因为没有对方代码。

---

### 已排除（附证据）

| 项 | 证据 | 结论 |
|---|---|---|
| 位姿 / 坐标轴 / 尺度 / 内参约定（含 F_FLIP、`to_w2c`、s = 7） | A1：规划器约定反投影与仿真端 `sample_points` 的相对误差中位数 0.25–0.49%（4 条 episode × 每 10 步）。这正是两种内参约定（焦距差 0.4%，PROTOCOL_ADAPTATION §3.2）的预期量级。故意去掉坐标轴翻转作对照，误差为 115–125% | 排除 |
| D12（init 位姿语义） | 全部 70 条 f1c4 和 70 条 D3a 的 hello 都是 `init_pose = docstring`。在无天空场景里，旧帧重渲染 PSNR 27–33、深度误差 < 1.1%，第一帧位置正确 | 排除（官方 tester 的写法才是 bug） |
| RGB–深度对齐 | 同上。旧帧在多视角下也一致（barts 的 t ≤ 75 全部帧 PSNR 均值 26–33） | 排除 |
| EIG 计算本身 | A2 逐高斯分解重算的结果与官方 `pose_eval` 的相对差 ≤ 3e-4（多数 < 1e-5） | 排除 |
| 配置覆盖（map_every / keyframe_every = 1、60 次迭代、学习率 / 剪枝 / 加密沿用 yaml） | hello 行；`fisherrf_full_planner.py:164-178` 对照 `configs/mp3d_gaussian_FR_eccv.yaml` | 合理的偏离（D2），不是原因 |
| 路径 EIG 公式 / 常数 | 与 `tester.py:830-893` 逐式一致；对决策影响 < 0.1%（F6） | 排除 |
| GT 端评估链 | 官方覆盖率回放 70 / 70 逐位一致；A3 中 40 / 40 已执行动作的增益与记录逐位一致 | 排除 |
| D13（无有效深度帧） | 只影响建图是否跳过；EIG 与随机的比较在已建图状态上成立 | 不是主因 |

---

## 3. 差距来源小结

| 来源 | 级别 | 是否解释差距 | 估计量 |
|---|---|---|---|
| F1 EIG 无覆盖信号 + 缺有效 frontier 约束 | 设计，已确认 | **是，主因（可能）** | 闭环 ≈ Random 的直接原因；论文值与 Random 之差（约 0.19）只能来自一个有信号的 frontier 成分。闭环量未测 |
| F8 论文协议 / 动作空间不同 | 背景，可能 | 部分（未知） | Random 本身相差 +0.11，提示协议差异不小 |
| F3 天空导致高斯失控 | 地图，已确认 | 次要 | 健康地图上 EIG 同样不比随机好；去掉大高斯后 Top-1 仍 ≤ 随机 |
| F2 λ 的尺度 | 适配，已确认 | 次要 | 改 λ 后 Top-1 不变好 |
| F4 幽灵高斯 | 官方问题，已确认 | 次要 | 去掉后 Top-1 不变好 |
| F5 两次观测规则 | 官方规则 + 大步长，已确认 | 只影响 v3 / v4 | — |
| 实现 bug | — | **未发现** | — |

---

## 4. E. 能否在官方原生基准上做端到端核对（未下载任何数据）

**官方实验用的数据**：
- Gibson 9 个场景（Greigsville、Denmark、Cantwell、Eudora、Pablo、Ribera、Swormville、Eastville、Elmira）；
- 5 个 HM3D 场景（DBjEcHFg4oq 等，`AM/README.md`）。

**代码需要的文件**：
- 场景：`<root>/data/scene_datasets/<scenes_dir>/<scene>.glb`（gibson）或 `<scene>/<scene>.glb`（mp3d 类型），外加同名 `.navmesh`；
- PointNav episode 文件：`data/datasets/pointnav/{gibson|mp3d}/v1/val/val.json.gz`（`datasets/dataloader.py` 中的 `ep_file_path`；`habitat.Env` 需要它）。

**本地现状**：
- 只有原始 MP3D 发行版（90 个 scan：`matterport_mesh` 的 OBJ + 纹理、`house_segmentations` 等）；
- 在 `~/scratch/data` 里没找到任何 `.glb`、`.navmesh` 或 PointNav `.json.gz`。

**可选做法**（大小未在下载页核实，只给量级）：

| 方案 | 需要的数据 | 许可 | 能检查什么 | 代价 |
|---|---|---|---|---|
| a. Gibson 子集（官方主实验） | Gibson habitat 场景（官方 9 个场景，数 GB 量级）+ `pointnav_gibson_v1` | Gibson Database 许可表（斯坦福，学术非商业，需填表等批准） | 唯一能与 FisherRF 论文数值直接对比的选项 | 2000 步、800×800，每次重规划要对全部关键帧（约 500 个）算两次 H_train。估计每场景 1–3 L40S GPU-h（未测），9 个场景约 10–25 GPU-h |
| b. HM3D 5 个场景 | HM3D val 的 glb + navmesh（每个场景百 MB 量级） | Matterport HM3D 学术 EULA，通过 API token 下载 | 同上（论文 HM3D 部分） | 5 个场景约 5–15 GPU-h |
| c. MP3D habitat 版 | `download_mp.py --task habitat`（90 个 scan，约十几 GB）+ `pointnav_mp3d_v1` | 用户已持有的 MP3D ToS | 只能检查官方 tester 端到端能跑、地图和覆盖率行为合理；**FisherRF 论文没有 MP3D 数字可对照** | 同上 |
| d. 不下载，用原始 OBJ 自建 glb + navmesh（habitat-sim `recompute_navmesh`）+ 自造 episode 文件 | 无 | — | 同 c，但还要额外排查坐标系（原始 OBJ 为 z-up）、材质、navmesh 参数造成的偏差 | 0.5–1 天工程；风险较高 |

**环境**：fisherrf env 里已有 headless（EGL）habitat-sim 0.2.4（`ENVIRONMENT.md` D3、E1），但没有在集群上实际初始化过渲染器，未验证。

**判断**：
- 这个检查只能验证"我们的构建在原生域里正常"。
- 本审计已经从另一侧说明了构建正常：健康地图下 PSNR 27–33、EIG 重算一致、几何约定正确。差距不是构建问题。
- 所以它对当前决策的价值低；只有在想发表"我们的 FisherRF 构建与官方一致"这类声明时才值得做，选 a 或 b。

---

## 5. 建议

**推荐 (4)：不再复现，在 CVPR 表中引用论文报告的 FisherRF 数值（单列"论文报告"组）。** 本复现（f1c4，FisherRF-full†）连同本审计放在附录，作为"同协议适配版在本协议下 ≈ Random"的诊断。

理由：
1. 差距不是实现 bug（§2 已排除列表）。所以 (1) 修 bug 后重跑当前设计，预期不会改变结论：F2–F4 的修正在固定状态上都没有让 EIG 超过随机。
2. (2) 更严格地照搬官方设计不可行，原因有三：
   - 官方 frontier 是 2D 地面机器人设计，配 0.05 m / 5° 的原子动作；
   - 严格移植的 v3 / v4 已经退化（F5），只到 0.491 / 0.492（3 场景）；
   - 要让它工作，就得改占据规则、改 3D frontier，这已经是 (3)。
3. 论文结果依赖 MAGICIAN 未公开的 frontier 修改，以及很可能不同的动作空间（F8）。任何同协议的数，都只是"我们设计的 frontier 探索器 + Fisher 打分"，无法核实是否忠实于论文。
4. 与 `paper_tables/` 已有的"论文报告组 / 重新评估组"分法一致。

**退路 (3)，仅在必须有同协议一行时采用**：预注册一个"frontier 优先"的适配，只跑一次，不调参。
- 做法：
  - 未知 / 已知改用单次观测雕刻的标签；
  - 从整个位姿空间采样，按"视野内从未观测体素数"排序取 top-K（K = 5，与 MAGICIAN e4e08fe 的排序设计对应）；
  - 对这 K 个目标走现有的 BFS 路径和官方路径 EIG 选择；
  - 可同时加 F3 / F4 的适配器侧清理。
- 固定状态上的预期：K = 5 时每步约 8.8 pp，随机 3.1 pp。闭环量未知。
- 流程：先在 3 场景 × 5 起点上试跑，再跑 All-14（约 70 × 13 min ≈ 15 GPU-h）。
- 结果必须标注为"FisherRF†（我们的 frontier 适配）"。由于它的探索主要来自 frontier 计数，这一行其实更像 frontier 基线，正文需要明确说明。

**不建议 (1) 和 (2)**，理由同上。

---

## 6. 复现本审计

```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene; A=$R/experiments/cvpr_results/fisherrf_reproduction/scripts/audit
PY=/bsuscratch/jiahuizhang/envs/fisherrf/bin/python; MG=/bsuscratch/jiahuizhang/envs/magician/bin/python
$PY -B $A/a1_geometry_check.py                       # CPU
$PY -B $A/a4_behaviour.py                            # CPU
# GPU（只经队列）：每条 episode 一个任务 = 规划端探针 + GT 增益
env -u LD_PRELOAD -u PYTHONPATH SLURM_JOB_QOS=kd-high $PY -B $A/a2_probe_planner.py --scene S --start K --states 5 15 30 50 75 --out OUT
env -u LD_PRELOAD -u PYTHONPATH $MG -B $A/a3_gt_gain.py --scene S --start K --probe OUT
$PY -B $A/a5_analyze.py --probe OUT; $PY -B $A/a6_figure.py OUT
```

注意：
- 队列任务脚本留在 `gpu_queue/done/frfaudit_*`；
- 探针重建地图不逐位（R1），但候选 key 由种子决定，可以跨重跑按 key 合并 GT 增益。
