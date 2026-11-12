# FisherRF F4 — 运行时间（2026-11-11）

> **状态更正（2026-11-11）：** 本文件中标为 FisherRF 的数值来自 **D3a 版本**：官方 3DGS 建图加官方 `pose_eval` 评分，但候选只有相邻一步的合法动作，**没有官方的 frontier 目标和 A\* 路径评估**。它不是 MAGICIAN 论文所用的 FisherRF 适配版（论文 Table 2：AUC 0.546、Final 0.786），只能作为消融，记作 **FisherRF-score-1step**，不能作为 FisherRF 行进入 CVPR 表格。按 MAGICIAN 论文做法保留 frontier 目标和路径评估的 **FisherRF-full** 正在实现，见 `PILOT_FULL.md`（完成后生成）。以下原文保留不改。


**计时：** 规划器内部计时，每段前后都调用 `torch.cuda.synchronize`。
- 视角选择：H_train、候选 Hessian 和 EIG。
- 在线 3DGS 更新：官方 `track_rgbd`，即每步跟踪加 60 次建图迭代，含官方写 PNG 的开销。
- 窗口：室外 t = 10..99，室内 t = 10..149。

| 条件 | 视角选择均值 | 3DGS 建图均值（其中写 PNG） | 合计均值 | 中位数 | P95 | 规划器峰值显存 |
|---|---|---|---|---|---|---|
| 室外，独占 L40S（校准作业 3495338，pisa s0，n = 90） | 0.27 s | 1.20 s（0.39） | 1.47 s | 1.37 s | 2.15 s | 0.32 GB |
| 室内，独占 L40S（校准作业 3495340，Z6MF L0 S6，n = 140） | 0.40 s | 1.06 s（0.37） | 1.46 s | 1.41 s | 2.07 s | 0.41 GB |
| 室外全量，每卡 4 个进程共享（70 条，n = 6,300） | 1.02 s | 2.53 s | 3.54 s | 3.96 s | 5.94 s | 0.39 GB |
| 室内全量，每卡 4 个进程共享（44 条，n = 6,160） | 1.86 s | 3.27 s | 5.13 s | 5.41 s | 7.34 s | 0.52 GB |

**注意：**
- 独占的两行各只有 1 条 episode，状态来自 FisherRF 自己的轨迹，不是 EXP8 的配对状态。RGB 只在 FisherRF 的运行中渲染，Table 1 的记录里没有 RGB，所以无法在完全相同的状态上配对。
- 与 EXP8 的 RAYSCOUT（决策中位数 3.30 / 3.57 s）只能做量级比较：FisherRF 每步更便宜，约 1.4 s，每步只对约 7 个相邻候选打分；但覆盖率远低于其他方法（见 OUTDOOR_RESULTS / INDOOR_RESULTS）。

数据：`results/runtime_summary.json`。
