# G0 验证报告 —— V4 与 V3.1（2026-10-31 20:10–20:45）

> 全部 12 门通过；证据为单测输出、冒烟运行的 `run_*.json` / `log.jsonl`、回放比对 JSON。单测：`exp6_v4/tests`（9）+ `exp6_v3.1/tests`（1）**10 passed**（CPU，63 s）；`tests/futureview` **12 passed**（V2 既有测试）；`exp6_v3/tests` 6 passed。冒烟：V4 在 gpu115（作业 3435033，`work_dirs/exp6_v4/{runs,logs}`），V3.1 在 gpu117（作业 3429774，`work_dirs/exp6_v3_1/{runs,logs}`）。

| 门 | 检查 | V4 | V3.1 | 证据 |
|---|---|---|---|---|
| G0-1 基线审计 | checkpoint / 训练来源 / 配置可追溯 | ✅ | ✅ | `AUDIT_BASELINES.md`、`CHECKPOINT_LINEAGE.md`（sha256 实测） |
| G0-2 数据划分 | Train-6 ∩ Dev-3 / Test-3 = ∅；138 训练 / 42 S-c 轨迹 `heldout` 标志一致；V4 与 V3.1 的训练列表相同；MP3D eval ∩ train / VALSEL = ∅；D100 根目录无 MP3D | ✅ | ✅ | `test_split_and_leakage.py::test_g0_2_*` |
| G0-3 DDP / 累积归一化 | 分片求和 / (2 × 全局射线数) 与整批逐位（loss 与 U-Net 梯度 rtol 1e-4）；4 卡 1k 实跑 loss 单调下降 | ✅ 1k：1.77 → 0.128 | ✅ 1k：3.36 → 0.359（数据端 / 前向端射线计数逐步核对） | `test_loss_normalisation.py`；`runs/smoke_short/run_short.json` |
| G0-4 V4 形状 | cond [V, 76, 64, 128]、velocity [V, 2, 64, 128] 有限；V2 模块参数不变（2,412,000 / 76,224 / 22,333,890），X-Attn 25,430 | ✅ | — | `test_v4_model.py::test_g0_4` |
| G0-5 X-Attn 梯度 | 10 个 X-Attn 张量梯度非零有限；tiny overfit 第 10 步 norms encoder 0.092 / rays 0.082 / ray_xattn 0.181 / unet 5.86；200 步 loss 1.96 → 0.144（drop 0.927 > 0.9），采样符号一致 0.976 > 0.95 | ✅ | — | `test_g0_5`；`runs/smoke_overfit/run_overfit.json` |
| G0-6 相机条件 | 同 P1 两个相机：采样坐标随外参平移、注意力分布不同（mean |Δ| 0.016）、条件图不同；同方向平移原点 → 残差不同 | ✅ | — | `test_g0_6` |
| G0-7 可见性 mask | 网格外采样点全 mask 的射线：残差 0、注意力 0、无 NaN；全 UNKNOWN / 域外网格的网格内点仍是合法 token；部分 mask 的整网络前向有限 | ✅ | — | `test_g0_7` |
| G0-8 V3.1 VAE | 架构 = T0（参数 23,539,622，seed-0 初始公共模块 sha 与 T0 manifest 相同）；μ / logvar / KL / 重建有限并反传到编码器；tiny overfit 200 步 3.70 → 0.373（drop 0.899），第 10 步梯度 vae_enc 1.19 / vae_dec 2.64 / rays 0.043 / unet 6.19 | — | ✅ | `test_v31_model.py`；`runs/smoke_overfit/run_overfit.json` |
| G0-9 零泄漏 | 训练键不含 `tstar`；编码器输入索引 t − Δ ≤ t；粗网格已观测体素随 t 单调不减 | ✅ | ✅ | `test_g0_9_no_future_information` |
| G0-10 checkpoint / resume | V4（单卡，精确 RNG 恢复）：A 40 步 vs B 20 + 20：model 最大差 7.0e-6（rms 0.029）、EMA 4.2e-7，loss 序列一致；V3.1（4 卡，V3 规则：重新播种）：A 60 vs B 30 + 30：checkpoint 状态恢复、resumed_at 30、损失有限，第 1 步 loss 逐位相同 3.688944 | ✅ | ✅ | `runs/resume_check.json` |
| G0-11 基线回放 | (a) 开关关闭：参数名与 V2 Original checkpoint 相同，cond / 采样与改动前模块逐位相同；(b) 冻结规划器 2 步闭环（T0 / V2-Mix × MP3D / Macarons++）的位姿与覆盖率与存档正式 episode 逐位相同（4/4 PASS） | ✅ | ✅（共用） | `test_g0_11a`；`work_dirs/exp6_v4/eval/g0_replay/replay_check.json` |
| G0-12 吞吐 | 4 卡 1,000 步（冒烟子集 = 前 6 条训练轨迹，小场景）：V4 0.360 s/步（p90 0.361），峰值 10.6 GB/卡；V3.1 0.397 s/步（p90 0.399），峰值 4.1 GB/卡；无 OOM | ✅ | ✅ | `runs/smoke_short/run_short.json`；**偏差 D-0**：不含 dunnottar 等大网格，正式吞吐以正式运行前 500 步实测为准 |

补充事实：
- V4 单卡 64 视角 / 8 状态的正式路径（resume 测试）峰值 21.3 GB；4 卡正式每 rank 16 视角 / 2 状态预计低于此。
- V3.1 大场景参考：V3 G0 实测 dunnottar 单进程 27.4 GB（1 状态 8 视角）；`accum = 2` 使每个 micro-batch 只含 1 个状态，峰值与 V3 T0 同量级（33 GB/卡）。
- 冒烟 EMA（`ema_smoke.pt`）与冒烟 run 目录不是结果，不参与任何评估。
