# FisherRF F3 — Macarons++ 室外结果（2026-11-11）

> **状态更正（2026-11-11）：** 本文件中标为 FisherRF 的数值来自 **D3a 版本**：官方 3DGS 建图加官方 `pose_eval` 评分，但候选只有相邻一步的合法动作，**没有官方的 frontier 目标和 A\* 路径评估**。它不是 MAGICIAN 论文所用的 FisherRF 适配版（论文 Table 2：AUC 0.546、Final 0.786），只能作为消融，记作 **FisherRF-score-1step**，不能作为 FisherRF 行进入 CVPR 表格。按 MAGICIAN 论文做法保留 frontier 目标和路径评估的 **FisherRF-full** 正在实现，见 `PILOT_FULL.md`（完成后生成）。以下原文保留不改。


**协议：** 与 CVPR Table 1 相同。
- All-14 场景 × 5 个官方起点，100 步；官方设置网格；官方邻居 → `get_valid_neighbors` → GT 线段碰撞；官方 MAGICIAN 覆盖率；场景等权平均。
- 模拟器为冻结的 `t24_out_sim` 链，FisherRF 作为规划器子进程接入（`scripts/fisherrf_sim_out.py` + `fisherrf_planner.py`）。

**FisherRF 设置：**
- 官方 FisherRF-active-mapping `8b196cd2` 的 GaussianSLAM。
- 每步跟踪当前帧，用 GT 位姿；每步建图 60 次迭代。
- 每个合法候选用官方 `pose_eval` 的 EIG 打分（`backward_power = 2`，λ = 0.1），取 argmax。
- 决策 D1–D13 见 `PROTOCOL_ADAPTATION.md` 和 `SMOKE_TEST.md`；其中 D12 改用 docstring 语义的初始位姿，D13 对无有效深度的帧跳过建图。
- 作业：3495338（校准）、3495366 / 3495368 / 3495370（全量，每卡 4 个进程），均为 L40S。

**验证：**
- 70 / 70 条 episode 完成。
- 官方覆盖率重放与闭环存储曲线逐位相同：70 / 70（`fisherrf_out_cov.py`，即 Table 1 的 t25 流程；sbatch 3495384）。

| 方法（All-14，场景等权） | AUC@20 ↑ | AUC@50 ↑ | AUC@100 ↑ | Cov@20 ↑ | Cov@50 ↑ | Final ↑ |
|---|---|---|---|---|---|---|
| Random Walk† | 0.144 | 0.248 | 0.353 | 0.226 | 0.384 | 0.517 |
| **FisherRF†（本复现）** | 0.130 | 0.219 | 0.301 | 0.197 | 0.329 | 0.439 |
| MACARONS† | 0.352 | 0.540 | 0.660 | 0.538 | 0.741 | 0.810 |
| MAGICIAN | 0.307 | 0.544 | 0.711 | 0.537 | 0.806 | 0.906 |
| Ours | 0.420 | 0.633 | 0.763 | 0.646 | 0.854 | 0.911 |

† 表示在同一冻结协议下适配。除 FisherRF 外，其余行都是 CVPR Table 1 的同协议数值。

**配对差值（按场景 bootstrap 的 95% CI；括号后为 FisherRF 更高的场景数）：**

| 对比 | AUC@20 | AUC@100 | Final |
|---|---|---|---|
| FisherRF − Ours | −0.290 [−0.333, −0.248]，0 / 14 | −0.463 [−0.492, −0.432]，0 / 14 | −0.472 [−0.521, −0.410]，0 / 14 |
| FisherRF − MAGICIAN | −0.177 [−0.213, −0.142] | −0.410 [−0.454, −0.365] | −0.468 [−0.521, −0.403] |
| FisherRF − Random | −0.014 [−0.030, +0.001]，6 / 14 | −0.052 [−0.087, −0.016]，4 / 14 | −0.078 [−0.120, −0.036]，4 / 14 |

**行为：**
- 每条 episode 101 个位姿里平均只有 18.8 个不同位置，大部分步数在原地改变俯仰或方位。
- 7,000 步中 1,046 步的当前帧没有有效深度，按 D13 跳过建图；32 步因还没有地图而按种子随机选择；没有平局。
- 原因：EIG 偏好对已有 Gaussian 补拍的近距离视角，候选只有相邻一步（D3a：与所有方法相同的统一动作空间），没有官方的 frontier 目标引导。

**结论：** 在我们的协议下，FisherRF 的覆盖率低于 Random Walk（AUC@100 −0.052，Final −0.078）。这里报告的是复现结果，不是论文数值。官方 Gibson / HM3D 实验与本协议不同，不可比较。

数据：`results/outdoor_episodes.csv`、`results/summary.json`；原始 episode 在 `work_dirs/cvpr_results/fisherrf_reproduction/outdoor/closed_loop/<scene>_s<k>/`。
