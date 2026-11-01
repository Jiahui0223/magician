# Checkpoint 继承关系与 sha256（EXP6 V4 / V3.1 审计，2026-10-31；全部由文件实测）

| 名称 | 路径 | sha256 | 大小 | 初始化来源 | 训练数据 / 阶段标签 | 选择 | G3 规划口径 |
|---|---|---|---|---|---|---|---|
| V2 Original EMA@60k | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt`（= `work_dirs/exp5_future_view/task1/training/main/ema_060000.pt` 硬链接） | `a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc` | 99,423,044 B | 随机 seed 0 | **Outdoor-only scratch**（Train-6 去 S-c，稀疏 P1），60k 步 | 固定 EMA@60k | 稀疏 / 稀疏（Task 2 V2O） |
| V2-Mix EMA@17.5k（B0 = B2） | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt` | `343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516` | 99,423,044 B | V2 Original EMA@60k | **Mixed fine-tune**（MP3D train + Train-6 稀疏），20k 步 | ES 比值最小 → 17.5k | B0：稀疏 / 稀疏；B2：D100 / 稀疏 |
| E2-D100 EMA@5k | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_005000.pt` | `61439612cb6132afbe74976b82a531c8f827802b06b4f87ad6de1a197d0a2fa5` | 99,423,044 B | V2-Mix EMA@17.5k | **Mixed fine-tune**（D100 分片），5k 步 | ES 比值最小 → 5k | D100 / 稀疏 |
| V3-T0 final（EMA@40k） | `work_dirs/exp6_v3/task0_vae/checkpoints/final.pt` | `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6` | 94,289,914 B | 随机 seed 0 | **Mixed scratch**（MP3D train + Train-6，D100 + TSDF 缓存），40k 步 | ES_in/ES_in(5k) + ES_out/ES_out(5k) → 40k | D100 / 稀疏 |
| V3-T1 final（EMA@40k） | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt` | `639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb` | 95,060,040 B | 随机 seed 0（公共模块同 T0） | **Mixed scratch**，40k 步 | 同上 → 40k | D100 / 稀疏 |
| **V4 Outdoor-only**（本轮） | `work_dirs/exp6_v4/checkpoints/final.pt`（= `runs/main/ema_040000.pt`） | 训练后填写 | | 随机 seed 0（V2 模块 + ray_xattn） | **Outdoor-only scratch**（Train-6 去 S-c，D100 分片），40k 步 | 固定 EMA@40k（训练前冻结） | D100 / 稀疏 |
| **V3.1 Outdoor-only**（本轮） | `work_dirs/exp6_v3_1/checkpoints/final.pt`（= `runs/main/ema_040000.pt`） | 训练后填写 | | 随机 seed 0（T0 架构，公共模块初始 sha 与 T0 相同：`6a9d6071a225…`，测试 `exp6_v3.1/tests` 断言） | **Outdoor-only scratch**（Train-6 去 S-c，D100 + TSDF 缓存），40k 步 | 固定 EMA@40k | D100 / 稀疏 |

继承链：V2 Original → V2-Mix（B0 / B2）→ E2-D100；V3-T0 / T1 独立从零；V4、V3.1 独立从零，不加载任何上表权重（训练器断言 / manifest 记录）。MAG-G / MAG-B 用 MAGICIAN 官方权重 `e697009`（Task 2 `magician_planner_grid.py` / `t24_mag_out_planner.py`），与上表无关。
