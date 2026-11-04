# CHECKPOINT_AUDIT — F1 的父模型、继承与新 checkpoint

**说明**：
- 所有数值来自训练 / 冒烟时的运行记录：`work_dirs/exp6_v5/F1/runs/<arm>/run_none_*.json`，以及 `results/smoke_compare.json`；
- 新 checkpoint 的 sha256 由 `scripts/aggregate.py` 写入 `results/checkpoints.csv`。

## 1. 父模型：B0 = V2-Mix EMA@17.5k
| 项 | 值 | 来源 |
|---|---|---|
| 文件 | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt`（只读） | Table 1 `MODEL_REGISTRY.yaml` B0 |
| sha256 | `343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516`；训练开始前与结束后各核对一次 sha，并检查 mtime 未变 | `train_f1.py` |
| 内容 | 只有 EMA（键：`ema` / `step` = 17500 / `model_cfg`），没有 raw 权重与 optimizer。两个分支都从 EMA 初始化，optimizer 全新创建 | 实测 |
| 结构 | `model_v1`：Encoder3D（32 / 64 / 96）+ RayEncoder（Ray32，Conv1D → fc 64）+ UNet2D（64 / 128 / 192 / 256）；共 390 个张量，24,822,114 个参数 | 实测 |
| 祖父模型 | V2 Original Outdoor S10 EMA@60k，sha `a379c0fb…` | Table 1 注册表 |
| 训练 | V2O EMA@60k 之后做 Mixed FT：MP3D 114 个楼层 + Macarons++ Train-6 去 S-c，均为 S10，每步 32 + 32 个视角；共跑 20k 步，按冻结的选择规则选中 17.5k | Table 1 注册表；`task2/protocol.md` |
| B0 / B2 | 同一份权重：B0 = S10 推理 + S10 评分，B2 = D100 推理 + S10 评分。F1 以 **B0 / S10** 为父协议 | Table 1 注册表 |

## 2. 加载与继承
| 分支 | 加载方式 | 结果 |
|---|---|---|
| F1-A | `V2Model(model_v1).load_state_dict(ema, strict=True)` | 缺失键 0、意外键 0，390 个张量全部加载 |
| F1-B | `V5Model` 按 `v5_model.load_v4_warmstart` 的逐名逐形规则复制 `encoder.*` 与 `unet.*` | 复制 382 个张量（24,745,890 个参数），复制后每个张量都 `torch.equal` 等于父模型 |
| F1-B 未使用的父参数 | `rays.c1` / `c2` / `c3` / `fc` | 共 8 个张量（76,224 个参数），只是不用，没有删除父文件中的任何内容 |
| F1-B 新模块 | `dec` = V5RayDecoder | 242,757 个参数，seed 0 随机初始化；**没有加载** V5-Warm12k 的 decoder |
| 参数总量 | F1-A 24,822,114；F1-B 24,988,647 | F1-B 多 166,533 个参数（+0.67%） |

- **代码版本**：`seen2scene/futureview/v5_model.py` 未改动，sha256 前缀 `67b500f6d6561e86`，最后一次修改在提交 ad8dc5c（V5 原始实现）；`model.py` 的 sha256 前缀为 `9a30386b167e1c21`。
- **配置版本**：F1-B 的 `v5` 块取自 `experiments/exp6_v5/configs/model_v5.json`（sha256 前缀 `64b3814f8c002b73`），只覆盖了 `out_scale_init` 一项。

## 3. 起步时两条条件路径的输出统计
- 来源：`results/smoke_compare.json`。
- 样本：14 个训练状态、96 个视角、700,416 条射线。

| 量 | B0 射线上下文（Ray32 / Conv1D） | F1-B decoder（out_scale = 1） |
|---|---|---|
| 均值 / 标准差 / RMS | −0.0009 / 0.028 / **0.02753** | 0.006 / 0.571 / **0.58191** |
| 每条射线 64 维范数，均值 / p95 | 0.211 / 0.369 | 4.56 / 5.00 |
| 有限值 | 是 | 是 |
| 新旧向量的余弦（均值 / 绝对值均值） | — | 0.008 / 0.061（随机初始化，与旧特征无关，符合预期） |

- **out_scale_init**：按冻结规则，0.02753 / 0.58191 = **0.0473**，在正式训练前写入配置。参照：历史 V5 用的是 0.05。
- **投影与有效列**：两条路径的 11 个投影通道和有效列，按 bf16（U-Net 输入卷积的精度）比较逐位相同。

## 4. 启动链回归（训练前）
- **做法**：用 F1 的 G3 链路（`sim_out_v5.py`，`V4_P1_MODE=sparse`，`V4_CKPT` 设为 B0 权重）重跑 barts_s0。
- **结果**：执行动作与官方覆盖曲线都与 CVPR Table 1 的 B0 记录**逐位一致**。规划器报告 `T3_P1_MODE=sparse`、sha `343f41ab…`。
- **结论**：B0 parent 的 G3 结果直接复用 Table 1 记录。

## 5. 新 checkpoint
- **保存位置**：`work_dirs/exp6_v5/F1/runs/{A,B}/ema_{000000,001000,002000,003000,004000,005000}.pt`。
- **内容**：每个文件包含 `ema` / `step` / `model_cfg` / `provenance`（父 sha、配置哈希、模型配置哈希、分支）/ `lineage`。另有 `ckpt_last.pt`，包含 model、EMA、optimizer、step、RNG、配置哈希和父 sha，供续训使用。
- **配置哈希**：A `b1749665f013…`，B `3b40d6fc9ca9…`。
- **F1-B 模型配置哈希**：`1b1ff32c4658…`。
- **代码**：训练代码固定在提交 23fc161，无本地改动（`git diff` 为空）。
- **sha256**：完整值见 `results/checkpoints.csv`（自动生成）。
