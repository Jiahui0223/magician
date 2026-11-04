# PROTOCOL — EXP6 V5 / F1：Geometry Query Isolation & Early-Coverage Evaluation（protocol f1-v1）

**冻结**：2026-11-03，在任何 F1 正式训练开始之前冻结。只有 F1-B 的 `out_scale_init` 留待按 §3 的规则填入，同样在训练前完成。

**机器可读版**：
- `configs/f1_common.yaml`
- `configs/f1_a_baseline.yaml`
- `configs/f1_b_geometry_query.yaml`
- `configs/f1_eval.yaml`

**修改规则**：改动只能记入 `DEVIATIONS.md` 并新建配置版本，不得静默覆盖。

## 1. 研究问题与边界
**问题**：从 B0 出发，只把旧的 Ray32/Conv1D 几何条件换成 V5 的多尺度 Geometry-Memory Query，在同样的数据、预算与评估下：
- 候选视角的未来深度预测能否改善（G1）；
- 选点能否改善（G2）；
- 室外的 AUC@20、AUC@100、Final 能否提升（G3）。

**不引入**：L_surf、L_bridge、Teacher、Hazard、直接覆盖评分、VAE、语义头、4³ 粗格、Ray64、R1/R2/R5。

**不改动**：已有的 checkpoint、Table 1 结果与冻结协议文件（全部只读）。

## 2. 两个分支（共同部分见 `f1_common.yaml`）
| 项 | F1-A Continue（同预算对照） | F1-B Geometry Query |
|---|---|---|
| 初始化 | B0 = V2-Mix EMA@17.5k（sha `343f41ab…`），全部参数 | 同一份 B0：encoder 与 UNet2D **逐名逐形复制**，并用 `torch.equal` 核对；V5RayDecoder 随机初始化（seed 0）；**不加载** V5-Warm12k 的 decoder |
| 结构 | 原 V2Model（Ray32/Conv1D → 64） | V5Model（`seen2scene/futureview/v5_model.py`，代码不改）：F0 ≤ 24 / F1 ≤ 12 个对数深度 token、2×2×2 粗 token、null token、几何偏置交叉注意力、SurfaceMLP 探针摘要、CondAdapter → 64 通道；条件布局与 V2 相同（11 + 64 + 1） |
| 损失 | 只有 L_FM | 只有 L_FM（SurfaceMLP 只作为摘要特征，没有单独监督） |
| 可训练参数 / 学习率 | 全部参数；lr 2e-5，前 500 步预热，余弦退火到 0（E2-S10 的已验证配方） | 第 0–1999 步只训练 decoder（lr 1e-4，前 200 步预热，余弦退火到 5000 步）；第 2000–4999 步 decoder 与 encoder、UNet2D 全部训练，继承部分 lr 2e-5（从 2000 步起预热 500 步，余弦退火到 0） |
| 数据 | Macarons++ Train-6 去掉 S-c，**S10** 分片（T1.1）；不使用 MP3D | 同左 |
| 预算 | 5,000 次优化器更新，全局 batch 64 视角 / 8 个状态，4 卡 DDP | 同左 |
| 随机性 | numpy 样本流 `[seed 0, rank]`；torch 在模型构建之后重设为 `0 + 1000·rank`（eps / tau） | 同左，两分支的 batch、eps、tau 相同（逐步摘要 `digest.jsonl` 核对） |
| EMA / checkpoint | EMA 0.999；EMA 保存在 0 / 1k / 2k / 3k / 4k / 5k；`ckpt_last` 每 500 步保存一次（可续训） | 同左 |

**比较性质**：
- F1-A 与 F1-B 比较的是“Query + 两阶段适配流程”与“原模块继续训练”。这是系统级的配对比较，不是严格排除优化路径因素的纯架构消融：F1-B 的继承参数只训练 3k 步，F1-A 训练 5k 步。
- **F1-B 第二阶段解冻全部继承参数**（学习率与 F1-A 相同，都是 2e-5，属于小学习率，不是整网大学习率重训）。理由：
  1. 历史 V5 的 Joint 阶段同样解冻全部参数（当时 lr 1e-5）；
  2. 如果只解冻接口层，F1-A 对室外 S10 数据做的 encoder / U-Net 适配 F1-B 无法获得，比较会偏向 F1-A。
- 第二阶段的解冻**以短程冒烟为前提**：联合阶段的损失与各模块梯度范数有限，切换阶段时没有发散。结果见 `reports/TRAINING.md`。

**数据域说明**：B0 本身是用混合数据训练的（MP3D 与室外各 32 个视角）。F1 两个分支都只用室外 S10 继续训练，这与任务书 §2.2 一致。两分支相同，所以不影响 A 与 B 的配对，但两者相对 B0 都有数据域的变化（见 `CONFIG_ALIGNMENT.md`）。

## 3. F1-B 输出尺度的冻结规则
- `out_scale_init` = rms(B0 的 64 维射线上下文) / rms(新 decoder 在 out_scale = 1 时的输出)。
- 在训练状态上测量（`scripts/smoke_compare.py`，96 个视角，种子键 `exp6-F1|smoke|stats`），使 F1-B 起步时的条件尺度与 B0 相同。
- 测量值写入 `f1_b_geometry_query.yaml` 后才开始正式训练。
- 历史 V5 的取值为 0.05（V4 的射线上下文 RMS 约 0.03）。

## 4. 评估（`f1_eval.yaml`）
**推理设置**：所有模型都是 S10 推理 + S10 DistP1 评分、R10、K = 4、Euler 10、Greedy-NV，规划区域为整个场景。

**G1 / G2**：
- 状态：Dev-3（barts、pantheon、colosseum）的 E1.2 固定状态，B0 层与 B2 层分别报告，再合并。
- 复用 `e2_fixed.run_episode`：状态、候选、噪声键、回放规则、oracle 都相同。
- 模型：B0、A@2k、A@5k、B@2k、B@5k。
- B0 的分数必须与 E2 的行（scores.B0）逐位一致，否则 STOP。
- 早期状态 t ≤ 20 单独报告。

**G3**：
- 冻结的 M 协议模拟器，通过 `sim_out_v5.py`（`V4_P1_MODE=sparse`）运行。
- 训练前已核对：B0 权重经这条链重跑 barts_s0，与 Table 1 的 B0 记录逐位一致（执行动作与官方覆盖曲线）。
- **预注册 Pilot**：Dev-3 × 起点 {0, 1} × 100 步，即每个模型 6 个 episode。
- **扩展**：起点 {2, 3, 4}，合计 15 个 episode / 模型。
- B0 parent 直接复用 Table 1 的规范记录。

**覆盖口径**：
- 主口径：MAGICIAN 官方覆盖（`t25_out_cov` 重放必须与存档曲线逐位一致）；
- 辅口径：单调 S10 覆盖。

**指标**：
- Cov@k 与 AUC@k，k ∈ {10, 20, 40, 50, 60, 80, 100}；
- AUC@k = trapz(C[0..k]) / k；
- Final = C[100]，并检查与 Cov@100 相等。

**统计**：
- 场景先对起点取均值，再在场景间等权平均；
- 场景级配对 bootstrap 95% CI（10k，seed 0），报告胜 / 负场景数；
- 只有 3 个场景，CI 只作探索性参考。

**checkpoint 选择**：
- 只看 Dev-3 的 15 个 episode：每个分支在 2k 与 5k 之间选官方口径 AUC@20 更高者，差值小于 1e-4 时选 5k；
- AUC@100 与 Final 独立报告，不宣称存在单一的“最佳”；
- Test-3、Train-6 与 All-14 不用于任何选择。

**G1 / G2 不是 G3 的门槛**：无论 G1 / G2 结果如何，G3 Pilot 都必须完成。

## 5. Table 1 接口
- `table1_export/f1_model_registry_entry.yaml`：状态为 `PENDING_TABLE1_EVALUATION`，直到 Table 1 用它自己的评估器完成评估。
- F1 的 Dev-3 Pilot 数字不得直接拷贝到 Table 1 充当正式结果。

## 6. 资源与运行
- **QOS 限制**：账户只能同时用 8 张 GPU，已被两个交互作业占满（3443137 / gpu115、3461041 / gpu114），无法再提交新的 sbatch。
- 因此 F1 在这两个已有的 Slurm 分配内运行（ssh + setsid / nohup）。
- 所有步骤都可续训或续跑：训练有 `ckpt_last`；G1 / G2 和 G3 按 episode 写完成文件，并用认领文件避免重复运行。
- 偏差记入 `DEVIATIONS.md`。
