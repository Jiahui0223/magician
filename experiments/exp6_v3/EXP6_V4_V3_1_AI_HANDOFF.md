# 发送给 Seen2Scene 代码 AI 的正式任务

请完整阅读同目录的 `EXP6_V4_V3_1_OUTDOOR_FIRST_IMPLEMENTATION.md` 并把它作为本轮主任务书。执行目标不是停留在计划，而是实现代码、通过 G0、在合法 Slurm GPU 上实际启动两个 Outdoor-only 实验并完成其第一阶段 G1/G2/G3。

**建立两个独立版本，禁止覆盖旧实验：**

- `experiments/exp6_v4/`：3D U-Net + 原 Ray Sampling + Ray-Aligned Cross-Attention + **原 2D Rectified Flow**，从零训练、只用 Macarons++ Train-6。
- `experiments/exp6_v3.1/`：复用 V3 Task0 的 Simple 3D VAE + 原 Ray Encoder + **原 2D Rectified Flow**，从零训练、只用 Macarons++ Train-6，不加载 Mixed V3-T0 的权重。

**先审计再执行：**核对 V2 Original EMA@60k 究竟用过哪些数据、V2-Mix/B2/E2 checkpoint lineage、V3 原训练细节、分集隔离、F0/F1/ray shapes、P1 输入与评分语义、G3 和 MAGICIAN 对齐情况。不确定处写出审计结果，不要编造。冻结统一第一阶段训练预算（任务书建议 40k steps）、初始化、数据、EMA checkpoint 选择和评估协议；不能用 Test-3 / MP3D G3 选最佳 checkpoint。

**资源：**优先复用经实时检查确认合法空闲的 Slurm 资源；如有两组四卡，可 V4 4 GPU + V3.1 4 GPU 并行。做 1k smoke 和完整 G0；检查 DDP/EMA/Resume/GT leak、K=4、Ray-Attn 数值与梯度。不强制占满 GPU，不抢占他人的任务，不提交大缓存。

**第一阶段必须真实评估两个数据集：**

1. Macarons++ Outdoor：30 条 × 100 steps；AUC@100、Final、Cov@20。
2. MP3D Zero-Shot：10 条 × 150 steps；**从未用 MP3D 训练**；AUC@150、Final、Cov@20。
3. 两域 G1 深度预测和 G2 被选相机**真实单步新增 Coverage Gain pp** / Top-1，Regret 放附录；更重视完整 G3。
4. 最终统一结果表必须直接展示 **MAG-G、MAG-B、V2 Original、B0、B2、E2-D100、V3-T0、V3-T1、V3.1、V4**，逐场景、配对 CI、耗时/显存/GPU-hours；标明所有历史模型的训练来源，避免不公平归因。

**Planner 不改：**原 DistP1、原候选、原 P1 与 K=4；MP3D 主对照 `n_win=5`。`n_win=7/9` 可以准备独立冻结模型 Planner 消融，但本阶段不用于主表，不得与模型结构变化混用。暂不做 Beam Search、Frontier Bonus、S1/S2/S3、2D Flow 重设计或学习 Pose Predictor。

**下一阶段：**请准备 MP3D Indoor-only Fine-tuning 的配置和双域保留能力评估设计，但**Outdoor-only 两模型完成双域 G3 并向我汇报前，不要自动提交 Indoor Fine-tune 训练作业**。后续 V4/U0 严格 X-Attn 对照也要先看 V2 Original 训练审计，避免平白增加第三组大训练。

按主任务书的目录和 Git 规范实现并 push 轻量代码与报告。正式训练启动后报告 code SHA、Job ID、模型参数量、实际 4+4 GPU/Batch/Steps、日志路径和 G0 结果；第一阶段完成后报告双域 AUC/Final/Cov@20 与完整 MAGICIAN/旧模型对照，保留真实异常和结论限制，不得凭固定状态 Regret 推断闭环。

现在从 Phase 0 训练历史和源码审计开始，审计完成后直接实施与运行第一阶段。
