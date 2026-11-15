# Experiment G — Generative vs Deterministic（任务书；用户 2026-11-14 提出，原文保存）

## 用户的核心问题（英文原文）
The most important ablation we are missing is "generative" vs "deterministic". We argue that ours is better than magician because ours is generative and magician only uses deterministic occupancy predictions. We should have an experiment that can directly prove that: when everything else is the same, generative is better than deterministic.

用户希望的表格形式：Method（deterministic U-Net / generative flow / ours）、Generative（– / yes / yes）、K（1 / 1 / 4）、AUC@20、AUC@100 等。

## 用户粘贴的实验安排（原文）
四、实验怎么安排？
建议在项目中建立：

experiments/exp6_v3/ablation_study/experiment_G_generative_vs_deterministic/

整体流程只需要三个阶段：
1. 实现 Deterministic Head：保留几何编码网络和 U-Net 主干，改成一次前向预测 (b,y)，使用原数据训练。
2. 固定状态测试：比较 D0、G1、G4 的 First-Hit Error 和候选视角排序，确认模型正常工作。
3. 完整闭环评估：Macarons++ All-14、每场景五个起点、100 steps；输出 AUC@20、AUC@100、Final Coverage。
最重要的是，不要同时更换 D1 Planner 或删除 Ray Encoder，否则实验无法隔离生成与确定性预测的差别。
我的建议
就先做这三个版本，不需要一开始就设计复杂的四五组网络。

我尤其期待的是 G1 vs. D0：
- 如果 G1 已经优于 D0，说明生成式训练本身可能有价值。
- 如果 G1 与 D0 接近，但 G4 明显更好，说明主要收益来自多假设预测与 Coverage Aggregation。
- 如果 D0 与 G4 相近，那我们就不能把优于 MAGICIAN 的结果主要归因于生成建模，而应重新审视 CDGR 的贡献。
所以最关键的不是简单移除 Flow，而是用一个公平的 Deterministic Predictor 替换它，保持其余 Pipeline 不变。 这会是我们回应教授意见最直接、也最有说服力的一项 Ablation Study。

## 用户的执行指示（2026-11-14 晚，原文）
我觉得可以，我们先跑一下吧，重点是结合table，我需要的是一个可以用的table3，并且注意, 我们重点是 D0的确定性u_net的结果。这个非常重要，所以我们先跑一下吧。不要怕浪费时间，充分验证并且确保我们的模型是可对比的。而不是变量很多从而增加步确定性。不过，目前的实验过程设计还是可以的。

## Claude 的设计分析中已说明、并按用户"变量尽量少"的要求取舍的要点（2026-11-14）
- G1（`V3-T0-K1D2`，Table 3）和 G4（`V3-T0-D1`，Table 1）已有同协议的 All-14 结果，直接复用；只需新训练 D0。
- D0 的主定义：同一 V3Model（VAE + LatentRayEncoder + FlowUNet T0）、同一配置，只把 flow 头换成一次前向回归，x_τ 通道置 0、τ 固定 0，以同样的掩码对 x1 做平方误差；recon / KL 项与全部训练设置不变；推理用同样的解码规则和 D1（K=1 路径）。
- BCE 变体、D0-soft（概率加权）不做，因为它们会增加变量。G0（V3-T0 权重、零噪声确定性解码，无需训练）只作诊断行，用来区分"生成式训练"和"采样"两种效应。
- 室内 MP3D 不在本轮。
