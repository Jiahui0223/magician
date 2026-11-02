# EXP6 V5 —— Checkpoint 谱系

| 名称 | 文件 | 来源 / 父节点 | 训练数据 | 步数 | 规则 | sha256 |
|---|---|---|---|---|---|---|
| V4@40k（父） | `work_dirs/exp6_v4/checkpoints/final.pt` | V4 Outdoor-only 从零训练（exp6_v4，EMA@40k，只读） | Macarons++ Train-6 去 S-c（138 轨迹，D100），MP3D 0 | 40,000 | 固定 EMA@40k | `1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4` |
| V5 冒烟初始 | `work_dirs/exp6_v5/checkpoints/smoke_v5_init.pt` | V4@40k encoder / unet + 未训练 V5 decoder（seed 0） | — | 0 | **只用于 G0 冒烟，不是结果** | `eefde72f7ee9…` |
| **V5-Warm12k-S10Score** | `work_dirs/exp6_v5/checkpoints/v5_final.pt`（发布自 `runs/v5/ema_012000.pt`） | V4@40k encoder + unet 逐名复制；新 V5RayDecoder 随机初始化；Bridge 2k（继承部分冻结）+ Joint 10k | 同父节点数据 + Train-6 网格多表面标签（`labels/v1`）；冻结 V4 Teacher 的 H_old 只用于 L_bridge | 12,000 | 固定 EMA@12000（训练前冻结） | 训练后填写 |
| **V4-Continue12k-S10Score**（对照，不是 U0） | `work_dirs/exp6_v5/checkpoints/v4c_final.pt`（发布自 `runs/v4c/ema_012000.pt`） | V4@40k 全部参数（含 32 点 RayEncoder 与 X-Attn） | 同父节点数据，L = L_FM | 12,000 | 固定 EMA@12000（训练前冻结） | 训练后填写 |

- 两个新组共用：同一数据列表、同一全局 batch（64 视角 / 8 状态 / 步）、同一 EMA 0.999、同一种子与每 rank 采样流、同一步数；新优化器（AdamW），不沿用 V4 已退火到 0 的调度器状态。
- 父文件只读：每次加载前核对 sha256，`tests/test_teacher_migration.py` 核对加载前后 mtime 与 sha 不变。
- V5 推理只加载 V5 checkpoint（encoder / unet / dec）；V4 Teacher 只在训练进程内存在。
