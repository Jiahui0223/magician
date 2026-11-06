# Experiment CT — V3 Continued Training & Cross-Domain Generalization Study

（用户任务书原文，2026-11-05。可执行的冻结参数、状态集与门槛见 `configs/CT_config.yaml`。）

我们继续开展 seen2scene 项目的 EXP6 V3 研究。

上一轮 Experiment F 已完成，主要结论是：继续修改 Planner 没有取得足够可靠的收益，而 Future Depth Prediction Quality 仍然存在明显的潜在提升空间。

因此，本轮研究重点转向 V3 的继续训练，并同时评估 Indoor MP3D 和 Outdoor Macarons++。

**请完整检查当前代码、历史实验报告、Checkpoint、训练配置和评估协议，再开始实验。不要直接根据历史报告中的简写假设训练条件完全一致。**

## 0. 实验目录及基本要求

创建独立目录：

`experiments/exp6_v3/ablation_study/experiment_CT_continued_training/`

建议目录结构：

```text
experiment_CT_continued_training/
├── EXPERIMENT_PLAN.md
├── FINAL_REPORT.md
├── configs/
├── scripts/
├── results/
│   ├── CT0_checkpoint_audit/
│   ├── CT1_training_steps/
│   ├── CT2_domain_balance/
│   ├── CT3_prediction_diagnostics/
│   └── CT4_closed_loop/
├── figures/
└── logs/
```

不要修改或覆盖 Experiment E、F、CVPR Table 1、V3 历史 Checkpoint 和原有报告。

大模型权重、训练缓存和体积较大的中间数据放在适当的 work_dirs 中。

本轮只生成一份最终总结 `FINAL_REPORT.md`，其他结果使用 JSON、CSV、图片及日志保存。

## 1. 冻结模型架构与 Planner

本轮必须保留：

- V3-T1：3D VAE + Camera-Ray Cross-Attention + Rectified Flow。
- D100 Partial TSDF 输入。
- S10 Novel Surface Scoring。
- K=4 Stochastic Future Depth Samples。
- Euler10。
- R10。
- Two-Step Planning。
- Outdoor D-v2 已观测几何可见性约束。
- 当前模型的输入通道、输出定义及原始训练目标。

Indoor 保留当前 Local P1，不改为 Global P1。

Outdoor 保留固定 Global P1 / ROI。

两种 P1 的范围可以不同，但不得因本实验修改其网格定义、体素分辨率、更新规则或历史观测融合逻辑。

Indoor 与 Outdoor 使用相同的 V3 主网络架构。本轮研究共享 Checkpoint 在两个数据域上的表现。

Indoor 规划器需要先审计现有实现：如果 D-v2 尚未适配 Indoor，继续沿用当前冻结的 Indoor Planner。不得在本轮混入新的 Indoor Planner 改动。

## 2. CT0 — Checkpoint & Training Lineage Audit

这是本轮的前置任务，必须优先完成。

检查：

1. V3-T1 40K Checkpoint 的准确路径、SHA256 和权重来源。
2. 已有 50K Checkpoint 的准确路径、SHA256。
3. 50K 是否真正从 T1 40K 继续训练。
4. 50K 使用的训练数据、Loss、学习率、优化器、EMA 和采样策略。
5. 40K 与 50K 之间是否仅有训练步数差异。
6. 原始训练的 LR Scheduler 是否在 40K 时已经衰减到接近零。
7. 是否存在可恢复的 Optimizer、Scheduler 和 EMA 状态。
8. 训练分片的完整性，以及 Indoor/Outdoor 的真实数据比例。

注意：

如果历史 50K 还修改过风险参数、训练 Loss 或其他条件，必须明确说明。

尤其不要把 Planner 的 λ 从 0.25 改成 0 与训练权重变化混为一谈。

如果已有 50K 不是纯续训版本，应将其标记为 Historical Fine-tuning Checkpoint，而不是干净的 Training Steps Ablation。

输出 CT0 审计结果，并判断能否继续使用现有 50K。

## 3. CT1 — Continued Training Steps Ablation

核心目标：

判断延长 V3 训练是否真正改善 Future Depth Prediction、Camera Ranking 和最终 Coverage。

优先比较：

- CT1-A：V3-T1 40K，冻结基线。
- CT1-B：已有 50K，经过 CT0 审计。
- CT1-C：从冻结的 40K 起点执行统一续训，累计到 60K，并保存 50K 中间权重。
- CT1-D：累计到 80K，只有前期结果支持时才执行。

重要：

新增的 50K / 60K 应来自同一条可复现的续训分支，保持训练数据、损失、采样比例和学习率计划一致。

需要明确区分：

- Resume Training：恢复原优化器、Scheduler 等训练状态。
- Fine-tuning：从已有模型权重初始化，使用新优化器或新学习率计划。

如果原 40K 训练的 Cosine LR 已接近零，需要设计并冻结合理的续训学习率计划，不能直接采用零学习率继续训练。

不要因为 50K 与 60K 的表现不同，就在实验过程中不断修改学习率或训练配方。

每个 Checkpoint 都必须评估同一批固定状态。

重点记录：

- Validation Flow Loss。
- HIT/NOHIT 分类表现。
- First-Hit Depth MAE。
- 有效 UNKNOWN Ray 上的预测误差。
- K=4 生成结果的分布和稳定性。
- Camera Top-1 Accuracy。
- Oracle Camera Regret。
- Prediction Score 与 GT Coverage Gain 的排序相关性。

所有预测指标应分别报告 Indoor、Outdoor，以及 Early/Middle/Late 阶段。

## 4. CT2 — Indoor / Outdoor Domain Balance

这一阶段研究混合训练是否导致某个域的性能下降。

先使用原始混合数据和训练配方作为标准续训版本：

**CT-M：Original Mixed Training**

保持原始 Indoor/Outdoor 采样比例及有效射线 Loss 归一化方式。

然后仅在发现稳定的域间冲突时，运行：

**CT-B：Domain-Balanced Training**

建议将两个域的有效监督损失分别归一化：

`L_mix = 0.5 × L_indoor + 0.5 × L_outdoor`

注意这是候选实验配方，具体实现应先核对原代码的 Loss Normalization，不能直接重复归一化。

CT-M 和 CT-B 必须使用：

- 同一初始化 Checkpoint。
- 同一网络架构。
- 同一累计训练步数。
- 相同训练样本来源。
- 可比较的全局有效训练 Batch。
- 相同 LR Schedule 和随机种子配置。

本轮不默认增加 Outdoor-only 或 Indoor-only 主模型。

我们的最终目标仍然是一套共享模型权重能够同时处理两种环境。

如果域平衡训练没有带来可靠收益，就保留原始 Mixed Training。

## 5. CT3 — Prediction Quality Diagnostics

Experiment F 的 F0 发现，在模型可表达范围内使用理想 Future Depth，能够显著降低 Camera Selection Regret。

但目前尚未确定预测误差来自什么环节。

请将预测误差分解为：

### A. HIT / NOHIT

分析模型是否错误预测了表面存在性。

包括 False Positive、False Negative、Precision、Recall，以及 UNKNOWN Ray 上的分类表现。

### B. Depth Localization

仅对有效且类别正确的预测命中分析深度位置误差。

必须在真实物理深度空间计算，而不只是比较 Flow 输出的归一化 y。

### C. Sampling Diversity

对 K=4 样本分析：

- 不同随机样本之间的 Depth 差异。
- 生成命中分布是否覆盖真实 First-Hit。
- 采样方差是否随探索步数改变。
- K=4 相比 K=1 的 Camera Decision 收益。

不要把四个独立随机 Future Depth 称为四个跨 Camera 一致的完整 3D Worlds。

### D. Decision Error

重点分析：

- 预测误差较大的 Camera 是否更容易被错误选择。
- Depth MAE 改善是否与 Oracle Regret 改善相关。
- 是否存在预测指标改善但 Camera Ranking 变差的 Checkpoint。
- 前 20 步和后期是否存在不同错误机制。

对于训练标签中被标记为 BLOCKED 的 Ray，单独统计其占比、所在场景和可能的影响，但本轮不添加新 Loss。

## 6. CT4 — Indoor / Outdoor Closed-loop Evaluation

固定状态分析完成之后，再进行配对闭环。

### Indoor MP3D

保持历史评估协议：

- Local P1。
- 原始候选相机与动作规则。
- 固定规划器。
- 150 步。
- 同一场景、起点、噪声和评价器。

报告：

- AUC@20。
- AUC@150。
- Coverage@20。
- Final Coverage。
- 逐楼层结果。
- Episode 级与楼层级差异。

### Outdoor Macarons++

保持：

- 固定 Global P1。
- D-v2。
- K=4。
- Euler10。
- R10。
- 100 步。
- 官方 MAGICIAN Coverage Evaluation。

报告：

- AUC@20。
- AUC@100。
- Coverage@20。
- Final Coverage。
- 逐场景结果。
- 与原 40K + D-v2 的配对差异。

优先使用开发数据选择候选 Checkpoint。

只有在固定状态上发现可靠改善，或者为了检查训练长度对真实闭环的影响而预登记了必要的对照时，才进入较大规模闭环。

先做小规模闭环，然后才考虑完整 Outdoor All-14。

Test-3 此前参与过研究分析，不能称为严格盲测集。不得反复根据 Test-3 调整训练参数。

置信区间使用场景级或楼层级配对 Bootstrap，不得把同一场景中的大量状态当作独立样本。

## 7. 训练与 GPU 运行

在 Boise State HPC 上执行前，检查：

- 可用 L40/L40S GPU。
- Slurm 队列。
- 当前已有实验任务。
- 实际 GPU 显存和使用情况。
- Checkpoint 与训练缓存所在路径。
- BSU Scratch 可用空间。

优先复用已有训练代码和缓存。

使用独立 Slurm 作业运行训练与评估，记录 Job ID、运行时间、GPU 使用量和训练中断情况。

支持断点续训。

不要因为资源充足就无条件进行多个 80K/100K 训练。

先完成 CT0 和已有 Checkpoint 评估，再决定是否投入额外训练预算。

如需同时使用两组 GPU，应确保不会与其他已有任务冲突，也不要让多个实验在同一模型文件上竞争写入。

## 8. 结果判定

本轮不预设更长训练一定更好。

需要区分以下情况：

**情况 A：Indoor 和 Outdoor 同时改善。**

说明继续训练可能提高共享 V3 的泛化能力，可考虑采用新的 Checkpoint。

**情况 B：Indoor 改善，Outdoor 退化。**

分析域间训练冲突，考虑 CT-B 域平衡训练。

**情况 C：Depth 预测改善，Camera Ranking 不变。**

说明训练损失与 NBV 决策之间可能存在目标错配，不应盲目继续训练。

**情况 D：Camera Ranking 改善，但闭环 AUC 不变。**

分析状态分布变化和长期探索行为，不得将离线结果直接写成闭环增益。

**情况 E：所有指标均未改善。**

停止单纯延长训练，基于 CT3 的错误归因研究新的监督方式。

## 9. 最终报告

在：

`experiments/exp6_v3/ablation_study/experiment_CT_continued_training/FINAL_REPORT.md`

生成唯一面向研究总结的报告。

必须包含：

1. Checkpoint 谱系与训练条件审计。
2. 40K / 50K / 60K / 80K 的实际完成情况。
3. 训练与验证 Loss 曲线。
4. Indoor 与 Outdoor 的预测质量比较。
5. Camera Ranking 和 Oracle Regret 比较。
6. Early/Middle/Late 分阶段结果。
7. Indoor MP3D 的 AUC、Final 与逐楼层表现。
8. Outdoor Macarons++ 的 AUC、Final 与逐场景表现。
9. 原始 Mixed Training 与 Domain-Balanced Training 的差异。
10. GPU 成本、训练速度和可复现性。
11. 有统计支持的结论与尚未证实的趋势。
12. 最终推荐使用哪个 Checkpoint，以及是否值得继续训练。

不要只报告最佳结果。所有完成的主要对照与失败实验都需要保留。

**最终必须明确回答：继续训练能否在不改变 V3 架构、P1 表示和 Planner 的前提下，同时提高 Indoor 与 Outdoor 的实际 Coverage？**

## 10. 执行原则

先审计并制定冻结协议，再按 CT0 → CT1 → CT3 → CT4 执行；CT2 仅在存在明确域间冲突时开展。

不要在结果不佳时临时扩大超参数搜索。

如果已有 50K 不满足纯续训条件，应明确标注，并使用新的统一训练分支完成公平比较。

允许在当前资源和合理实验预算内持续执行，但不能将尚未运行的阶段写成已完成。

本轮重点是**训练研究，而不是修改模型架构**。

保持现有 V3-T1 40K + D-v2 作为论文主基线，只有新模型经过充分验证后，才提出替换建议。

请开始检查代码和历史 Checkpoint，完成 CT0 审计，然后按照已冻结的实验协议执行后续步骤。
