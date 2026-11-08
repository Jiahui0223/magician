# Experiment XB — 冻结协议（XB-v1）

2026-11-08 16:50 冻结，早于任何 XB 训练步和诊断运行。依据：`experiments/exp6_v3/XB_Experiment_Protocol.md`（用户任务书）。之后的改动新建版本并记入 `logs/EVENTS.md`；不根据结果修改。

## 1. 被审计的系统（不改动）

| 模型 | checkpoint | sha256（前 16 位） |
|---|---|---|
| V3-T1@40K | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt` | `639c9d3ae3f3c069` |
| XA-SR | `work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/checkpoints/xa_sr/final.pt` | `60069dc4c1419c32` |
| XA-MI | `work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/checkpoints/xa_mi/final.pt` | `71825e2f02f9b6d1` |

规划器 D1、模拟器、官方覆盖、室内 H0 路径全部沿用 XA 的冻结定义（`../experiment_XA_multiscale_ray_scene/FROZEN_PROTOCOL.md` §1、§4）。XA / CT 的权重、结果和报告不改动。

## 2. Task C — 室外数据缩放（唯一的重新训练）

| 项 | Baseline（已有） | Reduced-3（本轮训练） |
|---|---|---|
| 模型 | V3-T1（`model_v3.json`，xattn 开） | 同 |
| 室外训练场景 | Train-6 去 S-c：bannerman、dunnottar、fushimi、liberty、pisa、redeemer（138 条轨迹） | **bannerman、fushimi、liberty**（68 条轨迹） |
| 场景选择规则（训练前固定） | — | 把 6 个场景按建筑形态分三组：城堡 {bannerman, dunnottar}、雕像 {liberty, redeemer}、塔 / 神社 {fushimi, pisa}；每组取字母序第一个。不看任何验证或闭环成绩 |
| S-c 选模集 | 42 条（6 个场景的留出轨迹） | **不变**（42 条，含 3 个未训练场景的轨迹），保证选模分数与 Baseline 同口径 |
| MP3D 训练 / VALSEL | 228 / 9 条 | 不变 |
| 采样 | 每步每卡室内 8 + 室外 8 视角（全局 32 + 32）；状态权重 t ≤ 20 ×2；陈旧度同 V2 | 不变（室外每条轨迹被抽到的频率约为 Baseline 的 2 倍，训练后由日志统计） |
| 训练配方 | `train_v3.json`：seed 0、40,000 步、AdamW 1e-4 余弦、bf16、EMA 0.999、`L_FM + L_recon + 0.02·KL`、4 rank | 不变；**早停关闭**（T1 当时也没有触发） |
| 初始化 | 从零，公共模块哈希 `6a9d6071…` | 从零，同 seed，须得到同一哈希 |
| 选模 | 冻结规则（ES 比值，VALSEL + S-c） | 同 |
| 评估 | — | 室外 D1 All-14 70 条 × 100 步 + 室内 H0 6 条 × 150 步（与 XA 相同的起点、噪声键、队列与重放校验） |
| 对照 | — | 与 Table 1 的 V3-T1 + D1 记录做同起点配对，场景 bootstrap 95% CI；分 Train-6 / Dev-5 / Test-3 / Unseen-8。另报告“保留的 3 个场景”与“去掉的 3 个场景”两组 |

说明：Reduced-3 同时改变了场景多样性和样本数量（约 47% 的室外有标签射线），本轮不分离两者；任务书 §4.3 的“6 场景匹配样本量”对照不在本轮。

## 3. Task A — 注意力贡献审计（不训练）

固定状态池：Experiment CT 的 g1（室内 VALSEL 90 个状态、室外 Dev-5 495 个、室外 Train 260 个；状态、候选、K = 4 噪声键与 CT / F 相同）。真值：F 行的官方一步增益（室外）、`oracle.npz`（室内）。

- A1：对三个模型，在每个状态上记录注意力块的输入 RMS、输出投影后 RMS、残差 / 输入比例、输出方差、各头注意力熵与最大权重、M 记忆里全局（F16 / F8）与局部 F4 的注意力质量分配。
- A2：干预（相同状态、噪声、候选）：正常；残差置零；场景 token 特征在 token 之间打乱（位置编码保留）；位置编码打乱（特征保留）；MI 另加分别去掉 F16 / F8 / 局部 F4。比较首次命中误差、HIT / NOHIT 混淆、规划器分数的 Spearman、Top-1 一致率、官方 regret / Top-1、速度场差异（固定 τ 与噪声下的 RMS）。
- A3：几何敏感性——按 token 的几何类别（BAND 表面、已观测 EMPTY、UNKNOWN 边界附近、距 Query 射线的距离、远近）统计注意力质量，并做按类别遮蔽的干预。
- 这些干预只用于判断“已训练模型依赖什么”，不替代重新训练的消融。

## 4. Task B — P1 / 几何信息审计（不训练）

- B1：在固定状态上只用历史观测，比较（i）当前 D100 P1 的 BAND / EMPTY 体素，（ii）由同一组历史深度图直接反投影的点云与 4× / 8× 体素占据，统计已观测表面在 P1、4× 聚合、8× 潜变量网格各级被保留的比例，按距离分层；不引入任何未来视角。
- B2：用 CT 的反事实工具（`ct_cf.py`：`no_fp`、`no_fn`、`hit_oracle`、`depth_oracle`、`perfect`、`perfect_all`）在同一批状态上得到 T1、XA-SR、XA-MI 的 oracle 上界与 regret；privileged-geometry probe 单独标识、单独报告，不进入任何闭环成绩。
- B3：按阶段（early / middle / late）、域、深度远近、UNKNOWN 段数、到 UNKNOWN 边界的距离、局部 BAND / EMPTY / UNKNOWN 比例分层统计预测误差。

## 5. Task D

只用已有数据：XA 六个变体与 T1 的 70 条曲线，区分场景间、起点间、规划器噪声与训练种子方差（后者没有数据，只能给出需要几个种子的估算）。若 A–C 不能解释持平，给出追加 T1 种子的方案；本轮不训练种子。

## 6. 执行与资源

- gpu115（分配 3465477，11-09 15:12 到期）：Task C 训练 + 闭环（约 16.2 + 2.2 小时）。**因为分配到期时间紧，Task C 的训练与 Task A / B 并行启动，而不是在 A / B 之后**——这是对任务书 §6 顺序的偏差，Task C 的定义不依赖 A / B 的结果。
- gpu114（分配 3461041，11-08 18:30 到期）：到期前跑 Task A / B 的短诊断；到期后诊断转到 gpu115 与训练共用显存（每卡最多 1 个诊断进程，训练峰值 33.4 GB + 诊断 ≤ 8 GB）。
- 没有按指标取消的逻辑；闭环 70 / 70 + 6 / 6 必须跑满；工程故障按 XA 的重试规则处理。
- 代码哈希、配置、git 版本记在 `manifests/`；事件记在 `logs/EVENTS.md`。
