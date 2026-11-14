# EXP9-SCONE — 任务书原文（用户 2026-11-14 粘贴；逐字保存，不修改）

> 注：仓库中已有 `experiments/cvpr_results/exp9_ablation_qualitative/`（另一个 EXP9）。为避免重名，本实验目录为 `experiments/cvpr_results/scone_baseline/`，文中称 EXP9-SCONE。用户同一条消息还要求：先完成完整版 FisherRF 在 MP3D 室内的运行，再跑 SCONE。

# EXP9 — SCONE Baseline Integration and Evaluation

我们的论文改名为Seen2Map. 我们现在需要为 CVPR 2027 论文 **Seen2Map** 补充一个正式的 SCONE Baseline 实验，最终结果用于论文 Table 1，以及室内、室外的 Coverage Curves 对比。

**本次任务的核心要求：必须尽可能忠实地运行完整 SCONE 方法，并且与目前的 Seen2Map、MAGICIAN、MACARONS、FisherRF 使用统一的评估协议。不能用简化启发式方法冒充 SCONE，也不能使用 Ground-Truth Future Information 进行视角选择。**

## 1. 首先检查已有实现，不要直接开始重写

请先检查当前项目中是否已经包含 SCONE、MACARONS 或相关的 Occupancy/Visibility Prediction 实现。

同时参考以下官方代码：

- SCONE：[https://github.com/Anttwo/SCONE](https://github.com/Anttwo/SCONE)
- MACARONS：[https://github.com/Anttwo/MACARONS](https://github.com/Anttwo/MACARONS)
- SCONE Paper：NeurIPS 2022, *Surface Coverage Optimization in Unknown Environments by Volumetric Integration*。

特别注意：MACARONS 官方仓库包含更新后的 SCONE 实现、训练和评估流程，因此优先检查这个版本是否更适合我们现有项目。

请核实官方代码中是否有可用的预训练权重、配置文件和评估入口，优先复用，而不是从零训练或重新实现整个模型。

## 2. SCONE 必须保留的核心算法

完整 SCONE 至少应包含：

1. 根据已经获取的真实深度观测，维护当前部分场景点云及历史相机位姿。
2. 使用 SCONE 的 Occupancy Prediction 模块估计未知空间的占据概率。
3. 使用 Visibility/Coverage Gain Prediction 模块评估候选相机位姿可能带来的新增表面覆盖。
4. 根据原方法的 NBV 选择策略确定下一相机位姿。
5. 执行选定动作，获取新的真实深度观测，更新场景表示，再继续下一轮探索。

这里的 Full SCONE 指原论文的完整算法流程，而不是人为加入 Seen2Map 的 RayForge 或 CDGR。

禁止直接使用 GT Mesh、GT Future Depth 或 GT Coverage Gain 决定下一视角。Ground Truth 只允许用于模拟环境中执行动作后获取观测，以及独立的最终 Coverage Evaluation。

如果必须对原方法进行工程适配，请明确区分哪些部分来自原 SCONE、哪些部分属于我们的适配。

## 3. 与现有评估协议完全对齐

请检查当前项目中 Table 1、EXP7、MAGICIAN、FisherRF 和 Seen2Map 所使用的统一评估入口、数据集定义、相机动作空间、起始位姿、Coverage 计算及 AUC 汇总方式，尽量复用现有基础设施。

**Outdoor：Macarons++**

- 14 个场景。
- 每个场景 5 个起始位姿。
- 每条轨迹 100 个探索步。
- 共 70 条轨迹。
- 记录 AUC\@20、AUC\@50、AUC\@80、AUC\@100。
- 记录 Cov\@20、Cov\@50、Cov\@80、Cov\@100。

**Indoor：MP3D**

- 22 个 Validation Floors。
- 每个 Floor 2 个起始位姿。
- 每条轨迹 150 个探索步。
- 共 44 条轨迹。
- 记录 AUC\@20、AUC\@50、AUC\@80、AUC\@100、AUC\@150。
- 记录 Cov\@20、Cov\@50、Cov\@100、Cov\@150。

必须保证：

- 场景列表和起始位姿与现有主实验一致。
- 使用同样的 Mapping Domain、传感器可见范围和 Coverage Evaluation。
- 使用相同的 Exploration Step 定义。
- 最终指标先对各场景或 Floor 内的轨迹求平均，再对场景或 Floor 等权汇总。
- 如果需要将 SCONE 的原始相机动作空间适配到当前 Candidate Pose Generator，必须记录适配方式，不得未经说明更改其他方法的动作约束。
- 不得修改现有 Seen2Map、MAGICIAN 和 FisherRF 的结果。

SCONE 是依赖 Depth Observation 的方法。如果当前统一模拟环境同时提供 RGB 和 Depth，SCONE 只应获得原方法所需的输入，不应额外使用 RGB 信息或 GT Geometry。

## 4. 模型权重、数据和训练问题

请先检查是否能够使用官方预训练 SCONE 权重完成推理。

如果预训练模型需要进行场景尺度、坐标系或点云格式适配，请优先处理输入输出转换，同时保持网络及核心算法一致。

如果确实必须重新训练或 Fine-Tune，先报告原因、原始训练数据要求、预期资源开销以及可能造成的公平性差异。目标数据集的验证和测试场景不能用于有监督训练。

不能把 SCONE 的 Occupancy/Visibility 模块替换成手写 Coverage Heuristic 后仍然称为原版 SCONE。

## 5. 执行顺序

请按以下顺序完成：

**Phase A — Code Audit**

检查官方实现和项目当前代码，明确可复用组件、预训练权重、环境依赖、模型输入输出、场景适配障碍以及评估入口。给出实施计划。

**Phase B — Integration**

实现 SCONE 与现有 Outdoor/Indoor Planner 的接口适配，包括当前部分点云、历史位姿、候选视角、Gain Scoring 和动作执行。

**Phase C — Smoke Test**

先选择少量室外、室内场景各运行一条短轨迹，检查 Occupancy Prediction、Candidate Scoring、动作有效性、Coverage 更新和 AUC 计算。确认没有使用未来 Ground Truth，也没有明显的坐标系或尺度错误。

**Phase D — Full Benchmark**

在通过检查后，运行完整的 Macarons++ 70 条和 MP3D 44 条轨迹。保存每步的 Coverage、所选动作、运行时间及实验配置，并支持中断后恢复。

如遇到官方方法无法直接适用于某个 Benchmark 的情况，请明确汇报具体阻碍和可行的忠实适配方案，不能自动把该实验替换成其他方法。

## 6. 输出结果

请生成：

- Outdoor 和 Indoor 的 SCONE Coverage/AUC 汇总 CSV。
- 每条轨迹逐步 Coverage 数据。
- 室外、室内的 SCONE Coverage Curves，并与已有方法叠加。
- 与现有主表相同口径的 Scene-Level 汇总结果。
- Scene-Level Bootstrap 95% Confidence Intervals（如果现有工具支持）。
- SCONE 代码来源、Commit、模型权重来源、配置和运行命令。
- 完整实验记录与复现说明。

现有对比图目录：

`experiments/cvpr_results/fig/compare_full/`

请先检查目录和现有 CSV/绘图脚本，再决定如何接入 SCONE，确保不会覆盖已有结果。

最后提供一个可以直接填写 Table 1 的结果摘要：

Outdoor：AUC\@20 / AUC\@50 / AUC\@100 / Cov\@20 / Cov\@50 / Cov\@100。

Indoor：AUC\@20 / AUC\@50 / AUC\@150 / Cov\@20 / Cov\@50 / Cov\@150。

**本次任务不需要修改论文 LaTeX 或重新撰写正文。重点是完成忠实、可复现、评估协议一致的 SCONE 对比实验。所有无法完成的实验必须明确标注，不能推测或编造结果。**
