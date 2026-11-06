# Experiment CT — 暂停清单（CT-M 续训）

依据 `experiments/exp6_v3/XA_Experiment_Protocol.md` §1：先存档、校验，再停止。CT 的历史结果、原始 V3-T1 代码路径和 40K / Y50K checkpoint 都没有改动。

## 1. 暂停时的状态

| 项 | 值 |
|---|---|
| 运行 | CT-M（CT1-C）：V3-T1 从真实 40K 状态续训，LR 计划 2e-5（预热 500 步）余弦到 60,000 步 |
| 运行目录 | `work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/runs/ct_m` |
| 节点 / 分配 | gpu114，GPU 0–3，Slurm 分配 3461041（`vscode-dev`，分配本身没有取消） |
| 启动 | 2026-11-05 18:57:13，从 step 40000 恢复（父状态 `task1_xattn/runs/main/ckpt_last.pt`，sha256 `3d819782…`） |
| **可恢复检查点** | **global step 42000**，写于 19:43:24 |
| 停止 | 2026-11-05 19:44:54，约在 step 42070；先结束包装脚本（它带重试循环），再结束 torchrun 和 4 个 rank |
| 丢失的训练量 | 检查点之后约 70 步（约 1.6 分钟） |
| 里程碑 EMA | **没有**。EMA 文件每 5,000 步才单独保存并验证，CT-M 没有到 45K；EMA 权重在检查点内 |
| 代码版本（启动时） | git `f7d8f06` + 未提交改动（diff sha256 `03372c88…`，即当时未入库的 CT 目录）；训练器 `seen2scene/v3/trainer.py` 未改 |
| 配置 | `configs/train_ct.json`（sha256 `86c5f577…`）、`configs/CT_config.yaml`（CT-v1，sha256 `b9e865c0…`）、模型 `experiments/exp6_v3/configs/model_v3.json`（xattn 开，模型配置 sha256 `3deae01f…`） |
| 数据清单 | sha256 `9f432c3f…`（与 T1 相同） |
| 训练速度 / 显存 | 中位 1.38 s / step，峰值 33.4 GB / rank |

没有向训练进程发送任何“保存信号”：V3 训练器没有信号保存处理器，它每 1,000 步原子地写一次完整状态，存档用的就是这个周期检查点。

## 2. 存档

目录：`work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/archive_paused/`（只读，独立于运行目录；不入库）。

| 快照 | 文件 | sha256 |
|---|---|---|
| `snap_042000/`（**恢复用**） | `ckpt_step042000.pt`（380,317,806 字节） | `5b82500b2521b2867b7c0a49d9ccee769bca812a361fc0f3cc4cce76d3574ff4` |
| `snap_041000/`（较早的备份） | `ckpt_step041000.pt` | `3b65743103680aeb59a61d37a488b7ce85b29eb0bb1d960c03fef0f0ac58a10d` |

`snap_042000/` 还包含：`log.jsonl`、`train.log`、运行清单 `run_none_1793930233.json`、`train_ct.json`、`CT_config.yaml`、`train_ct.py`、`train_ct.sh`、`git_head.txt`、`git_status.txt`、`SHA256SUMS`（每个文件的 sha256）。`git_head.txt` 记录的是存档时刻的 HEAD（`7ae012e`，XA 提交）；启动时的代码版本以运行清单为准。

运行目录里的 `runs/ct_m/ckpt_last.pt` 仍在，sha256 与 `snap_042000` 的相同。

**检查点内容**（两个快照都已校验）
- 键：`model`、`ema`、`opt`、`step`、`np_rng`、`torch_rng`、`torch_cuda_rng`。
- `model` 和 `ema` 都能以 `strict=True` 载入 V3-T1 网络；EMA 权重全部有限。
- AdamW 状态 394 组；`step` 为 42000（41000）。
- 学习率不存状态：训练器按 `lr_at(step)` 由 global step 重新计算，所以调度器可以从 step 完全重建。

## 3. 恢复方法

在一个有 4 张空闲 L40S 的节点上：

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/seen2scene
bash experiments/exp6_v3/ablation_study/experiment_CT_continued_training/scripts/train_ct.sh \
  work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/runs/ct_m 29631
```

脚本发现 `runs/ct_m/ckpt_last.pt` 已存在，就从 step 42000 继续，不会重新从 40K 父状态播种。如果运行目录里的检查点丢失或损坏，先把存档复制回去并核对 sha256：

```bash
cp work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/archive_paused/snap_042000/ckpt_step042000.pt \
   work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/runs/ct_m/ckpt_last.pt
sha256sum work_dirs/exp6_v3/ablation_study/experiment_CT_continued_training/runs/ct_m/ckpt_last.pt   # 5b82500b…
```

## 4. 恢复的限制

- **随机数**：训练器恢复时按 (seed, rank, step) 重新播种每个 rank 的生成器（检查点里只存了 rank 0 的随机数状态，仅作记录）。恢复后的样本流可复现，但与不中断的运行不同。CT-M 从 40K 启动时已经是这种情况。
- **日志重复**：`log.jsonl` 的最后一行是 step 42050（检查点之后）；恢复后 42050 会再出现一次，分析时按 step 去重、取后一条。
- **代码**：恢复应使用与启动时相同的训练器和 `train_ct.py`（存档里有副本和 sha256）。此后入库的 XA 代码只新增了 `seen2scene/v3/xa/`，没有改训练器。
- **里程碑**：45K、50K、55K、60K 的 EMA 与验证都还没有产生，CT 协议里依赖它们的步骤（CT4 小规模闭环、CT1-D 的门槛判断、CT2 的触发判断）都未执行。

## 5. CT 其余部分在暂停时的状态

| 部分 | 状态 |
|---|---|
| CT0 谱系审计 | 完成 |
| 固定状态对比 40K vs Y50K（845 个状态，40K 回归逐位一致） | 完成 |
| 40K 闭环回归（barts s0、Z6MFQCViBuw_L0 S6） | 完成，与存档一致 |
| 40K 决策反事实（`states/cf_A_40K`） | **部分**：63 条轨迹中完成 26 条；19:34 停止，可续跑（按轨迹原子写入，有 claim） |
| Y50K 小规模闭环 | **未完成，0 条完成**：5 次室内规划器失败（与同卡任务抢显存，OOM）；19:20 停止时还在跑的 6 条室内、3 条室外（barts s0–s2）只有部分输出。室内的部分目录和所有 claim 已移到 `closed_loop/failed/`（没有删除）；室外的逐步检查点留在 `closed_loop/outdoor/B_Y50K/`，可由模拟器续跑 |
| CT-M 续训 | **暂停于 step 42000**（本清单） |
| CT1-D（80K）、CT2（域平衡）、CT4 All-14 | 未执行 |
