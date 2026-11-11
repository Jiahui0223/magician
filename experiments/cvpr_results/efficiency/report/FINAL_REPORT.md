# EXP8 计算效率基准 — 最终报告（2026-11-11 03:35）

## 摘要
在同一型号 GPU（NVIDIA L40S，单卡独占）、同一组真实规划状态上：
- RAYSCOUT（V3-T0@40K + D1，K = 4）每步决策延迟中位数 3.30 s；MAGICIAN beam 为 5.96 s，约为 RAYSCOUT 的 1.8 倍。
- RAYSCOUT 在 630 个状态中的 619 个上更快。
- 瓶颈是 RayForge 的 Flow 采样，占 68.5%，主要由 CDGR 第二层的约 31 个预测视角触发。GeoAnchor 摊销后可以忽略。
- 到达相同覆盖阈值，RAYSCOUT 需要的步数更少；累计规划时间在两个可精确计算的场景上约为 MAGICIAN 的 31–61%，All-14 的估计值为 39–51%。
- 不利的一面：RAYSCOUT 的峰值显存与参数量都更高。

本实验没有修改任何模型、权重、规划算法或评估协议。EXP7 的 GPU 没有被占用或中断。

## 1. 资源与执行
| 项 | 内容 |
|---|---|
| 新申请 GPU | 3 个 1 卡作业：3494525（gpu117 GPU 1，Ours 全部配置）、3494526（gpu117 GPU 2，MAGICIAN 4 条）、3494528（gpu116 GPU 0，MAGICIAN 3 条加自身 2 条）。每个都用 `sbatch --test-only` 确认无需抢占，全部 COMPLETED（1 h 12 min / 44 min / 54 min） |
| 未触碰 | gpu114 分配 3489889（EXP7 UAV 在用），gpu107 分配 3491984 |
| 命令 | `sbatch -J eff_<X> experiments/cvpr_results/efficiency/scripts/eff_job.sbatch configs/plan_<X>.txt`；每行一次 `eff_replay.py --method M --src S --scene s --start 0 --tmax 100 --out work_dirs/cvpr_results/efficiency/runs` |
| 汇总 | `eff_analyze.py`（统计）→ `eff_thresholds.py`（阈值步数）→ `eff_report.py`（表与图）；参数量 `eff_params.py` |
| 日志 | `work_dirs/cvpr_results/efficiency/logs/sbatch_eff_*.out`；运行目录 `work_dirs/cvpr_results/efficiency/runs/<METHOD>/<src>_<scene>_s<k>/`（驱动端逐步记录、规划器日志、stderr）；逐步数据副本 `experiments/cvpr_results/efficiency/raw/*.jsonl` |

**测法：** 离线重放 CVPR Table 1 的真实轨迹，把记录的服务记录逐步喂给冻结的规划器子进程，协议与闭环完全相同。主测是 7 条轨迹 × 第 10–99 步，共 630 个决策；两种方法输入完全相同。完整口径见 `BENCHMARK_PROTOCOL.md`。

**有效性核对：**
- RAYSCOUT 在自己 7 条轨迹上重放，每条 100 / 100 步的选择与 Table 1 记录相同；带计时器的两条也是 100 / 100。
- MAGICIAN 在自己 2 条轨迹上重放，同样 100 / 100。

## 2. 结果
**Table E1** —— `tables/E1_efficiency.*`
| 方法 | 决策延迟 均值 ± 标准差 | 中位数 | P95 | 含地图更新的计算周期 | 峰值显存（中位数，范围） | 推理用参数 | 每步评估视角 |
|---|---|---|---|---|---|---|---|
| MAGICIAN beam（10 × 10） | 5.79 ± 0.92 s | 5.96 s | 7.10 s | 6.06 s | 1.4 GB（1.0–2.4） | 3.65 M | 701 次渲染 |
| **RAYSCOUT（K = 4，CDGR）** | **3.35 ± 0.74 s** | **3.30 s** | **4.68 s** | **3.73 s** | 3.8 GB（2.7–30.5） | 22.95 M | 38 视角 × K = 4 |

**Table E2**（模块拆分）—— `tables/E2_rayscout_breakdown.*`，`figures/E1_runtime_breakdown.*`；要点见 `RUNTIME_RESULTS.md`。

**Table E3**（缩放）—— `tables/E3_scaling.*`，`figures/E2_efficiency_coverage.*`：

| 配置 | 决策延迟中位数 | AUC@20 / AUC@100 / Final（Macarons++ All-14） |
|---|---|---|
| R1（K = 1，一步） | 0.31 s | 0.390 / 0.737 / 0.886 |
| R2（K = 4，一步） | 0.56 s | 0.399 / 0.740 / 0.889 |
| R3（K = 1，CDGR） | 0.98 s | 0.420 / 0.756 / 0.900 |
| R4（K = 4，CDGR） | 2.78 s | 0.420 / 0.763 / 0.911 |
| MAGICIAN | 5.01 s | 0.307 / 0.711 / 0.906 |

这一组只用 redeemer、eiffel 两条轨迹计时；覆盖率来自 Table 1 / 3 的完整评估。

**Table E4**（覆盖阈值）—— `tables/E4_steps_to_threshold.*`、`tables/E4b_cumulative_planning_time_exact.*`、`tables/steps_to_threshold*.csv`。基于 Table 1 的全部真实曲线，每个方法 70 条。

| 阈值 | 达到的 episode 数（RAYSCOUT / MAGICIAN） | 场景中位数的中位数（步） | 两者都达到的 episode 上的配对差 |
|---|---|---|---|
| 50% | 70 / 68 | 11 vs 15.5 | 平均 −6.4 步，47 条更少、4 条相同、17 条更多 |
| 70% | 66 / 67 | 23.5 vs 31 | −8.4 步（45 / 2 / 17） |
| 80% | 64 / 66 | 31 vs 40.5 | −5.9 步（39 / 0 / 24） |
| 90% | 51 / 51 | 52 vs 56.5 | −6.8 步（34 / 0 / 15） |

**累计规划时间：**
- 精确值：取两种方法各自重放过的自身轨迹，两者都有的场景只有 eiffel 和 pisa（起点 0）。

  | 场景 | 方法 | 50% | 70% | 80% | 90% |
  |---|---|---|---|---|---|
  | eiffel | RAYSCOUT | 39.9 s | 56.6 s | 70.2 s | 126.3 s |
  | eiffel | MAGICIAN | 117.9 s | 182.8 s | 204.2 s | 278.1 s |
  | pisa | RAYSCOUT | 47.9 s | 81.7 s | 114.2 s | 214.2 s |
  | pisa | MAGICIAN | 78.3 s | 172.5 s | 265.7 s | 422.9 s |

- 估计值：All-14 上用"步数 × 每步延迟中位数"估算。只是估计，假设每步延迟恒定。

  | 方法 | 50% | 70% | 80% | 90% |
  |---|---|---|---|---|
  | RAYSCOUT | 36 s | 78 s | 102 s | 172 s |
  | MAGICIAN | 92 s | 185 s | 241 s | 337 s |

## 3. 验收问题
**Q1. RAYSCOUT 的完整单步规划时间。**
- 决策延迟（地图就绪 → 动作，GeoAnchor 按实际 1/10 频率摊销）：均值 3.35 s，中位数 3.30 s，P95 4.68 s，标准差 0.74 s。数据为 L40S，7 条真实轨迹，630 个决策。
- 含在线地图更新：均值 3.73 s，中位数 3.67 s。

**Q2. MAGICIAN beam 在相同协议下的时间。**
- 同一组状态：决策延迟均值 5.79 s，中位数 5.96 s，P95 7.10 s；含地图更新 6.06 s。
- 在它自己的轨迹上：中位数 6.14 s。

**Q3. RAYSCOUT 是否更快、快多少、是否可复现。**
- 是。决策延迟中位数之比 1.81 倍（均值 1.73 倍）；含地图更新为 1.68 倍。
- 630 个配对状态中 619 个更快；7 条轨迹都更快，比值 1.42–1.87 倍。
- 可复现性：
  - 两种方法的重放都与记录逐步一致；
  - 同两条轨迹上，带计时器与不带计时器的两次独立运行中位数相差 0.4%（2.79 s 对 2.78 s）；
  - 生产日志（共享 GPU）中的排序相同：Ours 每条均值 4.9–9.4 s，MAG 15.9–20.4 s。
- 局限：只测了一种 GPU 型号（L40S），每条轨迹只跑了一次。

**Q4. 瓶颈在哪里。**
- RayForge 的 Flow 采样（K = 4，10 步 Euler）占 68.5%。其中 57.8% 来自 CDGR 第二层的约 31.6 个预测视角，10.7% 来自第一层约 7.1 个候选。
- CDGR 自身的覆盖评分（value_cov）占 14.4%。
- GeoAnchor 每次编码 3.8 ms、每 10 步一次，摊销后占 0.03%。
- 1D Ray Encoder 占 1.5%，深度解码占 1.5%。
- 结论：瓶颈是 CDGR 两步展开带来的生成调用次数，而不是 GeoAnchor，也不是 Ray Encoder。

**Q5. K = 4 与 K = 1 的成本差。**
- 耗时：一步规划时 0.56 s 对 0.31 s（1.85 倍）；CDGR 时 2.78 s 对 0.98 s（2.85 倍）。不是简单的 4 倍，因为 K 是在批量里一起计算的。
- 覆盖增益（Table 3，All-14）：AUC@100 一步时 +0.002，CDGR 时 +0.007。K 的主效应 CI 含 0，规划深度的主效应显著。

**Q6. 相同覆盖目标下是否需要更少步数或更少累计规划时间。**
- 步数：在 50 / 70 / 80 / 90% 阈值下，RAYSCOUT 的场景中位数都更少（11 / 23.5 / 31 / 52 步，MAGICIAN 为 15.5 / 31 / 40.5 / 56.5 步）。两者都达到阈值的 episode 中，RAYSCOUT 平均少 6–8 步。但 50% 阈值下仍有 17 / 68 条 episode 是 RAYSCOUT 用的步数更多。
- 累计规划时间：在两个可精确计算的场景上，RAYSCOUT 用时约为 MAGICIAN 的 31–61%；All-14 的估计值约为 39–51%。
- 90% 阈值两者各只有 51 / 70 条 episode 达到。colosseum 上 RAYSCOUT 大部分起点达不到 70%，这是已知的平台期问题。

**Q7. 能否在 Title / Abstract / Introduction 中使用 "Computationally Efficient" 或 "Fast Planning"。**
- **已证实：** 在同一 GPU 和同一组真实状态下，单步决策延迟比 MAGICIAN beam 低约 1.8 倍；在相同覆盖目标下所需步数更少；累计规划时间更少（2 个场景是精确值，其余为估计）。
- **部分支持：** 一般意义上的"计算高效"。
  - 显存更高：中位数 3.8 GB 对 1.4 GB，大场景 30.5 GB。
  - 推理参数更多：23 M 对 3.65 M。
  - 只与 MAGICIAN 一个方法比较过，只在 Macarons++ 室外状态上计时。
- **尚未证实：** "实时"或"可在机载设备上运行"——每步 3.3 s，且只测了服务器 GPU。也不能宣称比 MACARONS / SCONE / FisherRF 更快，它们没有测量。

## 4. CVPR 写作建议
- 标题中不建议使用 "Computationally Efficient"。
- 摘要或引言可以写成下面这类限定清楚的表述，并在实验节给出 Table E1 和 Figure E1：
  "… reaches coverage targets in fewer steps and plans 1.8× faster per step than MAGICIAN's beam search on the same GPU (3.3 s vs 6.0 s per decision on an NVIDIA L40S)"
- 可以作为附加卖点：轻量配置 R1（K = 1、一步）每步 0.31 s；在同两条轨迹上 MAGICIAN 为 5.0 s，约快 16 倍。R1 的 AUC@20 / @100 仍高于 MAGICIAN（0.390 / 0.737 对 0.307 / 0.711），但 Final 更低（0.886 对 0.906）。
- 必须同时报告显存和参数量，并说明计时是在单卡独占条件下测得的。

## 5. 未完成 / 说明
- 没有测 Euler 5 / 20 步，这是可选项。
- 没有为 MP3D 和 UAVScenes 做独立计时。
- 没有测其他基线（MACARONS / SCONE / FisherRF）。
- MAGICIAN 的内部计时没有显式 CUDA 同步，阶段边界依赖隐式同步。总时间以同步的往返墙钟为准，两者相差 < 0.6%（计算周期 6.06 s，往返 6.09 s）。
