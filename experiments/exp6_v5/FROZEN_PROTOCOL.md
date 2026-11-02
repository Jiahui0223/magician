# EXP6 V5 冻结协议 v1.0（2026-11-01，正式训练前冻结）

依据：`EXP6_V5_ARCHITECTURE_DESIGN.md`（结构与数学语义，用户已确认）、`EXP6_V5_AI_IMPLEMENTATION_TASK.md`（工程任务与验收）。
冻结后任何改动记入 `reports/V5_DEVIATIONS.md`，不根据 S-c 以外的任何评估结果改动。

## 1. 组
| 名称 | 定义 |
|---|---|
| **V5-Warm12k-S10Score** | V5Model（`configs/model_v5.json`）：V4@40k EMA 的 3D U-Net 与 2D Flow + 新 V5RayDecoder；Bridge 2k + Joint 10k；规划器冻结（S10 DistP1 评分） |
| **V4-Continue12k-S10Score** | V4@40k EMA 全部参数继续训练 12k 步，同数据 / batch / EMA / 种子 / 采样流，L = L_FM；训练延长对照（不叫 U0） |
| 历史对照（不重跑） | MAG-G、MAG-B、V2 Original、B0、B2、E2-D100、V3-T0、V3-T1、V3.1、V4@40k（A0）、Experiment A 的 A1（D100 评分，仅诊断） |

## 2. 训练（`configs/train_v5_outdoor.json`、`configs/train_v4_continue.json`）
| 项 | V5 | V4-Continue |
|---|---|---|
| 初始化 | encoder + unet ← V4 EMA@40k（sha `1b31afff…`，逐名复制）；dec 随机（seed 0） | 全部 ← V4 EMA@40k |
| 数据 | Train-6 去 S-c 138 轨迹（E2 D100 分片）+ Train-6 多表面标签 `work_dirs/exp6_v5/labels/v1`；MP3D 0 | 同左（不用标签） |
| 步数 / batch | 12,000 步；全局 64 视角 / 8 状态；4 rank × 16 视角 / 2 状态；stale_p 0.5，stale_max 9 | 同 |
| 阶段 | 0–1,999 Bridge：encoder / unet 冻结，只训练 dec；2,000–11,999 Joint：全部训练 | 全程全部训练 |
| 优化器 | AdamW β (0.9, 0.99)，wd 0.01，梯度裁剪 1.0，bf16；dec lr 1e-4（200 步 warmup，余弦到 0 @12k）；继承部分 lr 1e-5（自 2,000 步起 500 步 warmup，余弦到 0 @12k） | AdamW 同参数；lr 1e-5（500 步 warmup，余弦到 0 @12k） |
| 损失 | L = L_FM + 0.37 · L_surf + λ_bridge(s) · L_bridge，λ_bridge(s) = 0.096 · max(0, 1 − s / 4000) | L_FM |
| λ 规则 | G0 梯度探针（300 步 Bridge，λ 暂取 1，固定冒烟轨迹）后 30 步新模块梯度 L2 中位数：FM 0.108、surf 0.148、bridge 1.125；λ_surf = 0.5 · g_FM / g_surf，λ_bridge0 = 1.0 · g_FM / g_bridge | — |
| L_surf | Δ = 8 VS；正例 = 存储表面 ±Δ/4；每表面旁 Δ..2Δ 难负例；8 个对数均匀查询；8 个首表面后均匀查询；标签 = Δ/2 内有表面；[t_lo, t_hi] 外不监督；BCE 在（正例、首表面前负例、首表面后负例）三组间均衡；每视角 256 条标签射线（从 1,024 条存储射线中随机） | — |
| EMA / ckpt | EMA 0.999；ckpt_last 每 1,000 步（可恢复）；EMA 存于 0 / 2k / 4k / … / 12k | 同 |
| **最终权重规则** | **固定 EMA@12000**（训练前冻结；不按 S-c、Dev-3、Test-3、MP3D 选择） | 同 |
| 监控 | S-c（Train-6 场景留出轨迹）每 2,000 步：FM、采样指标、belief（V5）；只监控 | 同 |

## 3. 评估（全部在冻结 checkpoint 上；测试场景已多次使用，**不是盲测**）
| 门 | 内容 |
|---|---|
| G1 | 固定状态（与 V4 相同的 365 态：MP3D VALSEL 90、Macarons++ Dev-3 B0 150 + B2 125）上的一步预测质量（未知 / 新 / 已知射线 MAE、命中一致、K = 4 离散度），V5 / V4C / V4 / B2 同状态同噪声；**belief**：评估端多表面 GT（Dev-3 网格 `dev_eval` 权限，MP3D VALSEL 网格；`eval_belief_gt.py`）上的首表面召回、第二及更后表面召回（按与首表面间隔、按深度分桶）、网格查询 precision / recall / Brier / ECE、首表面后的虚假表面率；GT 只在评估后用于指标 |
| G2 | 同 365 态：被选候选的真实一步覆盖增益（pp）、Top-1、early / mid / late、与 V4 选择不同的比例 |
| G3 | Macarons++ 6 场景 × 5 起点 × 100 步 = 30 ep；MP3D zero-shot 5 楼层 × 2 起点 × 150 步 = 10 ep；冻结规划器（候选生成器、K = 4、10 步 Euler、n_win = 5、P1 b2：D100 输入 / S10 稀疏 DistP1 评分）；主指标 AUC / Final / Cov@20，MP3D 停滞；场景 / 楼层级配对 bootstrap 95 % CI；判定沿用 Task 3 §4 标准 |
| 成本 | 训练 GPU-h、s / 步、峰值显存；推理每步时间、峰值显存 |
| 主配对 | V5 − V4C（同父节点、同数据 / 步数；V5 多了网格辅助监督，比较的是整个系统，不单独证明几何查询的纯架构效应）；V5 − V4@40k；V4C − V4@40k；V5 / V4C vs B2、E2-D100、MAG-G、MAG-B、V3.1 |

## 4. 停止条件
- 标签门失败 → 不训练多表面、报告阻塞（已通过，见 `AUDIT_V5.md` §5、`reports/LABEL_QUALITY.md`）。
- 训练非有限损失 → 停止并报告，不改协议重跑。
- 资源不足 → 只报告真实完成的部分。
- 完成 G0、正式训练、双域 G3 与报告后**停止本轮**；不启动 U0、Indoor FT、Memory、Frontier、其他 Planner 或新模型任务。
