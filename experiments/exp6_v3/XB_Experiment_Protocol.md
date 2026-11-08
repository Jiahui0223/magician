# Experiment XB — V3/XA Bottleneck & Data Scaling Audit

## 0. 背景与目标

目前 Experiment XA 已完成六组架构实验，全部取得 Macarons++ Outdoor All-14 官方 Coverage Closed-Loop 结果。

主要结论：

- 原始 V3-T1 + D1：AUC@20 = 0.425，Cov@20 = 0.654，AUC@100 = 0.765，Final = 0.910。
- 六个 XA 变体均未显著超过 T1。
- XA-SR：AUC@20 = 0.423，Cov@20 = 0.660，AUC@100 = 0.765。
- XA-MI：AUC@20 = 0.423，Cov@20 = 0.659，AUC@100 = 0.765。
- 六组 XA 仅使用单一训练种子，目前无法估计训练种子间方差。

**本次任务不再修改网络结构，而是寻找实际性能瓶颈。**

需要分别研究：

H1：Outdoor Training Data Diversity 不足。

H2：Partial TSDF / P1 / VAE Geometry Representation 存在信息瓶颈。

H3：Cross-Attention / Multi-Scale Attention 没有充分利用输入的几何特征。

本实验命名为：

`Experiment XB — Data, Geometry, and Attention Bottleneck Audit`

建议目录：

`experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/`

不要修改 XA 历史实验、Checkpoint 和既有结果。

---

## 1. 首先审查已有代码和报告

开始运行之前，完整阅读：

- `CVPR_MODEL_AUDIT.md`
- Experiment XA 的 `FINAL_REPORT.md`
- XA 的训练配置、模型结构、结果表
- V3-T1@40K 原始配置与 Checkpoint
- V3-T1、XA-SR、XA-MI 的模型代码
- 当前 D1 Planner 与官方 Coverage 评估协议

首先确认：

1. P1 的 D100、S10、R10 实际含义。
2. 3D VAE 如何编码 EMPTY / BAND / UNKNOWN。
3. 3D VAE 的 μ、logσ² 和 KL 的梯度路径。
4. 2D Spatial、Ray、Interval Query 的具体实现。
5. Attention Residual 是否参与 Flow Velocity Prediction。
6. 原始 Outdoor 训练场景、有效训练视角和样本采样分布。
7. Indoor/Outdoor 的真实采样比例。
8. XA 六组实验的训练步数、训练种子、网络参数量和推理预算是否一致。

所有结论必须以实际代码、配置、日志和 Checkpoint 为准。

先生成 `AUDIT_BASELINE.md`，记录这些事实。

---

## 2. Task A — Attention Contribution Audit

**优先执行，不重新训练。**

选择三个模型：

- V3-T1@40K
- XA-SR
- XA-MI

目标是判断 Attention 是否真正改变了 Future First-Hit Prediction，以及这种改变是否与 Camera Selection 有关。

### A1. Attention Residual Magnitude

逐层记录：

- Attention 输入特征的 RMS / Norm。
- Attention 输出投影后的 RMS / Norm。
- Residual 相对于输入的比例。
- Attention 输出的方差。
- 不同 Head 的 Attention 分布。
- Fine / Mid / Coarse Features 的贡献比例（适用时）。

特别检查输出投影 W_o，不能仅凭 W_o 权重大小判断模块是否有效。

### A2. Attention Intervention

对已经训练的模型执行：

1. 正常 Attention。
2. Attention Residual 置零。
3. Scene Token 特征打乱。
4. Scene Token 位置编码打乱。
5. 分别关闭 Fine / Mid / Coarse Attention（适用时）。

使用相同的固定状态、相同噪声、相同 Camera Candidates。

比较：

- First-Hit Prediction Error。
- HIT / NOHIT 识别能力。
- Camera Ranking。
- Top-1 Camera Agreement。
- Oracle Regret。
- Flow Velocity Prediction Difference。

注意：这些干预可能引起分布偏移，因此只用于诊断已训练模型的依赖程度，不能替代重新训练的结构消融。

### A3. Geometry Sensitivity

检查 Attention 是否关注：

- 已知 BAND 表面。
- 已观测 EMPTY 空间。
- UNKNOWN Boundary。
- Ray 附近的几何。
- 远处的 Global Context。

通过 Token 遮蔽及几何位置干预进行验证，不允许仅凭 Attention Map 的视觉效果得出结论。

最终回答：

**Attention 是否真正参与预测？如果参与，为什么没有带来更好的 Outdoor Closed-Loop？**

输出 `A_ATTENTION_AUDIT.md`。

---

## 3. Task B — P1 / Geometry Information Audit

**不重新训练，优先于数据缩放训练。**

这里需要严格区分历史观测信息和未来 GT。

### B1. Observed-Geometry Fidelity

在固定状态下，只使用已经执行的历史视角，检查：

- 真实观测的 Depth 有多少被写入 P1。
- 已观测表面是否在 TSDF 聚合时丢失。
- 4× 聚合是否导致小型建筑结构消失。
- 8× VAE Latent 是否保留关键 UNKNOWN Boundary。
- 远距离建筑表面的表示是否明显差于近距离表面。
- P1 ROI 内外的可见表面是否存在系统性遗漏。

可使用由相同历史观测重建的高精度参考表示作对照，但不得引入未来视角数据。

D100 已经是现有输入密度协议的一部分，不能简单把“再次使用 D100”当成新的完整地图上限。

### B2. Oracle Upper-Bound Diagnostic

在固定状态和固定 Camera Candidates 上，利用 GT Mesh 获取参考 First-Hit。

比较：

1. V3-T1 预测的 First-Hit。
2. XA-SR / XA-MI 预测的 First-Hit。
3. GT First-Hit。
4. 在相同决策条件下，由理想预测产生的 Camera Ranking 与 Oracle Regret。

另做一组有严格标识的 privileged-geometry probe，用于研究当前输入中缺少的几何信息是否对预测有价值。

如果使用 GT 补充 Scene Features，必须：

- 与正常推理严格隔离。
- 保持固定状态及监督集合可比。
- 明确哪些信息属于未来不可见几何。
- 不把 GT 填充后的地图当成真实 Partial TSDF。
- 不把所有 UNKNOWN 直接改成 BAND。
- 不把作弊输入的预测结果计入正常 Closed-Loop 成绩。

如修改输入导致有效 UNKNOWN Ray 集合改变，必须单独分析，不能直接与原始 First-Hit 错误率比较。

### B3. Error Stratification

按以下维度分组：

- Early / Middle / Late Stage。
- Indoor / Outdoor。
- Near / Far Depth。
- UNKNOWN Segment Count。
- UNKNOWN Boundary Distance。
- Building Scale（若能可靠获得）。
- Surface / Empty / Unknown 的局部比例。

重点回答：

**模型预测失败是因为没有观察到足够的几何，还是因为已有几何没有被有效编码和使用？**

输出 `B_GEOMETRY_AUDIT.md`。

---

## 4. Task C — Outdoor Data Scaling Experiment

这是本次唯一预先安排的正式重新训练实验。

### C1. 训练模型

选择原始 V3-T1，而不是同时重新训练六个 XA。

原因是 T1 是当前最稳定的主模型。首先应该在固定架构下研究数据集问题。

### C2. 数据设置

比较：

- Baseline：原始 6 个 Macarons++ Outdoor 训练场景 + 原始 MP3D。
- Reduced：仅保留其中 3 个 Outdoor 训练场景 + 相同 MP3D 训练协议。

3 个 Outdoor 场景必须在训练前根据明确规则固定，尽可能覆盖不同建筑形态，不能根据验证集成绩反复选择。

记录每个场景的轨迹、状态、视角数、有效监督 Ray 数和有效 Label 分布。

本轮保留相同的 Indoor/Outdoor 采样比例。

### C3. 训练公平性

Reduced T1 与原始 T1 保持：

- 相同网络结构。
- 相同 40K 训练步数。
- 相同 Global Batch Size。
- 相同 Loss 与 KL 设置。
- 相同优化器、EMA 和学习率策略。
- 相同随机种子及评估协议。
- 相同 P1 密度、标签生成和数据增强。
- 从零训练，而不是从原始 T1 Checkpoint 继续训练。

如果 reduced 数据数量使采样重复程度增加，需要在报告中记录每个场景和每条轨迹的采样频率。

重要：仅比较 3 场景与 6 场景，无法完全分离独立场景多样性和训练样本数量。

若后续需要更严格的归因，可新增一个保留 6 个场景、但匹配 reduced 组样本数量的对照。该对照作为后续独立实验，不得混淆本轮结论。

### C4. 必须执行完整 Closed-Loop

训练完成后，不允许因为 Early Validation 不理想就取消闭环。

必须运行：

**Macarons++ Outdoor All-14：14 场景 × 5 起点 = 70 条完整 Episode，每条 100 步。**

采用原始 D1 Planner、相同 Monte Carlo 采样数和 Euler 步数、相同官方 Coverage 评价标准。

主要指标：

- AUC@20。
- Coverage@20。
- AUC@100。
- Coverage@100 / Final。

补充：

- AUC@50。
- Per-Scene Coverage。
- Train-6 / Dev-5 / Test-3。
- Unseen-8。
- Paired Confidence Interval。
- 前 20 步完整 Coverage Curves。

同时做少量 MP3D Indoor Sanity Check，但使用一致的室内评估协议，不要与 Outdoor 官方 Coverage 混合汇总。

### C5. 结果解释

如果 Reduced 明显退化：

说明当前模型对训练数据规模或组成敏感，但不能据此直接证明增加 UrbanScene3D 一定有效。

如果 Reduced 基本不变：

说明减少到三个场景没有明显改变当前测得性能，但不能证明数据已经充足，也不能排除场景多样性不足。

如果 Reduced 更好：

检查是否存在过拟合、数据域冲突、场景采样不平衡或训练优化差异。

任何结果都必须保存完整日志。

输出 `C_DATA_SCALING.md`。

---

## 5. Task D — Training-Seed Uncertainty

本阶段先读取现有 XA 的置信区间和全部 Episode 曲线，评估观察到的差异与随机波动相比有多大。

注意区分：

- 同一模型的 Episode / Scene Variation。
- 不同训练种子产生的 Model Variation。
- 规划器随机噪声变化。

不要把场景级 Bootstrap CI 当成训练种子方差。

如果 A–C 无法解释当前近似持平的现象，提出追加 1–2 个 T1 训练种子的具体方案，优先使用同一冻结 Planner 和相同 Outdoor 测试协议。

本轮不需要直接重复六个 XA 的全部训练。

---

## 6. 执行顺序与资源控制

按照以下顺序自动执行：

```text
Code / Checkpoint / Protocol Audit
              ↓
       Task A: Attention
              ↓
       Task B: Geometry
              ↓
   Task C: Reduced Data Training
              ↓
   Outdoor Full Closed-Loop 70/70
              ↓
      Indoor Sanity Check
              ↓
       Task D: Uncertainty
              ↓
        FINAL_REPORT.md
```

不要在每个阶段完成后等待人工批准；在已有实验定义明确且资源可用的情况下继续执行。

只有遇到数据损坏、不可恢复的工程错误、资源无法满足或会破坏现有实验结果的问题，才需要报告并请求进一步决定。

所有任务必须有：

- 实际运行命令。
- 配置与 Git Commit。
- Checkpoint 路径。
- 训练日志。
- 完成状态。
- 失败和恢复记录。
- GPU-hours。
- 原始 Coverage Curves。
- 可重复执行的统计脚本。

不得修改原始 XA 或 CT 权重和报告。

---

## 7. 最终交付要求

在以下目录完成报告：

`experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/`

最终生成：

- `AUDIT_BASELINE.md`
- `A_ATTENTION_AUDIT.md`
- `B_GEOMETRY_AUDIT.md`
- `C_DATA_SCALING.md`
- `FINAL_REPORT.md`
- `MASTER_TABLE.csv`
- `PER_SCENE.csv`
- `EARLY_CURVES.png`

最终报告必须明确回答：

1. XA 的新 Attention 是否真正改变模型预测？
2. 当前 3D VAE / P1 是否丢失重要的已观测几何？
3. GT First-Hit 与模型预测之间还有多少 Oracle Headroom？
4. 将 Outdoor 训练场景从 6 个减少到 3 个，是否明显影响完整闭环？
5. 目前有没有充分证据支持“数据集太小是主要瓶颈”？
6. 下一阶段是否值得引入 UrbanScene3D、BuildingNet、BlendedMVS？
7. 若需要继续训练，应优先扩充数据，还是修改 Geometry Representation / Training Objective？

**最终目标不是一定证明数据集有问题，而是通过真实实验，尽可能区分 Data、Geometry、Attention 和 Training Variance 四种解释。**

不要提前写结论，不要根据少量固定状态诊断推断完整闭环收益。所有判断必须基于实际数据和日志。
