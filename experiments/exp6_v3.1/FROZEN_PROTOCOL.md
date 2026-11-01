# EXP6 V3.1 冻结协议 v1.0 —— 见 `../exp6_v4/FROZEN_PROTOCOL.md`（两组共用同一文件；V3.1 列为该文件 §3 右列）

V3.1 专属要点（`configs/outdoor_train.json`、`configs/model_v3_1.json`）：
- 架构 = V3-T0（`xattn.enabled = false`），从零初始化（seed 0；初始公共模块 sha 与 T0 的 `6a9d6071a225…` 相同，`tests/test_v31_model.py`），不加载 V3 任何权重。
- 数据 = Macarons++ Train-6 去 S-c（138 条，D100 分片 + 4× TSDF 缓存），MP3D 样本 0；S-c 只监控。
- 训练 = V3 配方（lr 1e-4、λ_rec 1.0、β 0 → 0.02 / 5k、EMA 0.999、40k 步、64 视角 / 8 状态 / 步），`accum = 2`（8 视角 / 1 状态 × 2 / rank）且 L_FM 按整步全局有效射线数归一化（= 64 视角批的梯度）。
- checkpoint = 固定 EMA@40,000；无早停；无任何评估场景参与。
- 评估 = 与 V4 相同的双域 G1 / G2 / G3（B2 口径，V3 规划器路径 `exp6_v3/shared/v3_{in,out}_planner.py`）。
- 本轮变动只是数据域：V3.1 对 V3-T0 的差异要描述为"同架构的 Outdoor-only 训练对照"，不是 VAE 架构改进。
