# PAUSED_RESTART — 夜间队列暂停记录与重启手册

- **暂停时间**：2026-11-03 14:06（用户要求暂停，先做下一个任务）
- **暂停方式**：
  1. 先停掉链式调度 `chain_stage2.sh`（PID 327）和 `chain_eval.sh`（PID 16524）；
  2. 等 V4-S10 写完第 38,000 步的完整 checkpoint 并核对；
  3. 再向 torchrun 发 SIGTERM。
- **运行中的进程**：0。GPU 0–3 空闲。
- **作业**：
  - 3443137（gpu115）保留运行，**11-04 15:11 到期**；
  - 3461041 是用户自己的 vscode-dev，排队中，没有动。
- **暂停时的代码版本**：git `28aa514`，分支 seen2scene。本文件和 `chain_resume.sh` 在其后提交。

## 1. 各项状态
| 项 | 状态 | 结果 / 报告 |
|---|---|---|
| Stage 0 Density A/B | **COMPLETED** | `experiments/exp_density_ab/FINAL_REPORT.md`、`NEXT_TRAINING_RECOMMENDATION.md`（64b6d9a） |
| Stage 1 Temporal Refresh | **COMPLETED** | `experiments/exp_temporal_refresh/FINAL_REPORT.md`（254d614）：保持 R10 |
| Stage 2 训练密度 | **COMPLETED**：S10（预注册规则，证据中等） | `experiments/exp_unified_outdoor/UNIFIED_PROTOCOL.md` |
| V2-S10 | **COMPLETED**：复用历史 V2 Original；固定状态 G1 / G2 已完成 | `experiments/exp_unified_outdoor/INTERIM_RESULTS.md` |
| V4-S10 60k | **PAUSED**：完整 checkpoint 在第 **38,000** 步（日志到 38,800） | 剩约 22k 步 |
| V4-S10 固定状态评估 | QUEUED（等 V4-S10 完成） | — |
| V4C-12k-S10、V5-warm-12k-S10 | QUEUED（等 V4-S10@48k） | — |
| S10 TSDF 缓存 | **COMPLETED**：180/180 | `work_dirs/exp_unified_outdoor/cache/tsdf4_s10`（7.7 GB） |
| V3-T1-S10 60k | QUEUED（先 profile 冒烟） | — |
| Mixed 20k | SKIPPED（按要求） | — |

## 2. Checkpoint 清单
### 2.1 V4-S10（`work_dirs/exp_unified_outdoor/runs/v4_s10/`，3.0 GB）
- **配置**：`experiments/exp_unified_outdoor/configs/v4_s10.json`（sha256 `10e22f8f463951aa92007375e30362a7176b5ec34b0046af46b2276e18e79605`）。
- **训练器**：`experiments/exp6_v4/scripts/train_v4.py`，经 `experiments/exp_unified_outdoor/scripts/train_unified.py` 调用（`uo_common` 补丁：S10 数据根、ExactResume）。
- **数据**：S10 分片 `work_dirs/exp5_future_view/task1/data/views`，Train-6 去 S-c 共 138 条轨迹 / 98,887 个视角，S-c 42 条轨迹用于监控。

| 文件 | 内容 | sha256 |
|---|---|---|
| `ckpt_last.pt` | **step 38000**：model、opt（AdamW 状态）、ema、np_rng、torch_rng、torch_cuda_rng | `c5df8906546c5d2a881f3c45489eb15ab2cea95b76e582e2151a2dbe6353422e` |
| `rng_states/s00XX000_r{0..3}.pt` | 每 1k 步一组、各 rank 的 RNG 快照（ExactResume），共 152 个文件，最新为 s0038000 | — |
| `ema_000000.pt` … `ema_038000.pt` | 每 2k 步一个 EMA（keys：arm、ema、model_cfg、step；model_v4；可严格加载为 V2Model，24,847,544 参数） | 见下 |
| `ema_010000.pt` | EMA@10k | `678d946131ef9c48cf00d0d7d058276ce409ad1662a286f609ee525aee88e560` |
| `ema_020000.pt` | EMA@20k | `f8bf535091db45f055762827def28983c836a091d2c13d4108e3deec33194557` |
| `ema_030000.pt` | EMA@30k | `a1a4929f1bc102750cfdc2e1db6dfbee4b1a734b6d1d193f44e999b63f282566` |
| `ema_036000.pt` | EMA@36k | `02753dc8e8f859e10ab1be6c5a1eceee4aea3d51c37966e9f757d5949f788bc4` |
| `ema_038000.pt` | EMA@38k（最新） | `466a2fead6d674d8eb521d11603b187b0871daba8b25ec9be79cdb196a2c3c2c` |
| `log.jsonl`、`eval.jsonl`、`resume_log_r*.jsonl`、`run_none_*.json`、`unified_launches.jsonl` | 训练日志、监控评估、续训记录（第 1000 步 4/4 EXACT）、启动记录 | — |

**注意**：
- 尚无 `ema_040000.pt` 和 `ema_048000.pt`。V4C / V5 的父节点 `ema_048000.pt` 要等 V4-S10 续训到 48k 才会生成。
- 第 38,000 步之后到暂停前的约 800 步会被丢弃，续训时由 checkpoint 重放。
- **38k 是中间 checkpoint，不能宣称性能。**

### 2.2 其他相关 checkpoint（只读，没有改动）
| 模型 | 文件 | sha256 |
|---|---|---|
| V2-S10（= V2 Original 60k，复用） | `work_dirs/exp5_future_view/task1/training/main/ema_060000.pt`（另有 ema_010000 / 020000 / 040000 / 045000 / 050000） | `a379c0fb9d7d80266a430ab6f48f39315fad54fb650c6c3740a999a02db079fc` |
| B2（V2-Mix EMA@17.5k） | `work_dirs/checkpoints/exp6_v2/B2/v2mix_ema17500.pt` | `343f41ab8d566c52007c881b5f32375c3253dd8f1c7a0d8a906d412a165e4516` |
| E2-S10 EMA@5k | `work_dirs/checkpoints/exp6_v2/E2/control_e2s10/e2s10_ema5000.pt` | `578855bb149956262f3eba976db837f09e61172542a4412a0404efcb69f1521b` |
| E2-D100 EMA@5k | `work_dirs/checkpoints/exp6_v2/E2/e2d100_ema5000.pt` | `61439612cb6132afbe74976b82a531c8f827802b06b4f87ad6de1a197d0a2fa5` |

### 2.3 已有的评估输出
| 输出 | 路径 |
|---|---|
| V2-S10 固定状态（39 个 episode，4 种推理 \| 评分） | `work_dirs/exp_unified_outdoor/eval_fixed/V2S10_60k/` |
| V2-S10 分析 | `experiments/exp_unified_outdoor/results/fixed_V2S10_60k.json` |
| Density A/B 原始数据 | `work_dirs/exp_density_ab/{fixed,g3}/` |
| Refresh 原始数据 | `work_dirs/exp_temporal_refresh/{fixed,g3}/` |

## 3. 重启命令（从仓库根目录 `/bsuscratch/jiahuizhang/projects/occupancy/seen2scene` 运行）
**推荐：一条链全部续上**（需要 4 张 L40S；每一步都会自动续训，已完成的臂会直接跳过）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup bash experiments/exp_unified_outdoor/scripts/chain_resume.sh > work_dirs/exp_unified_outdoor/logs/chain_resume.out 2>&1 &
tail -f work_dirs/exp_unified_outdoor/logs/chain.log      # 进度
```
顺序：
1. V4-S10 从第 38,000 步续训；
2. V4-S10 固定状态评估与 V4C-12k 并行；
3. V4C 评估与 V5-warm-12k 并行；
4. 缓存检查（已完整，会立即通过）；
5. V3-T1 profile 冒烟；
6. V3-T1 正式训练。

> ⚠️ **不要重新运行 `chain_stage2.sh`**：它包含一次性的“第 1000 步主动中断”测试逻辑，会在续训后再中断一次。

**也可以逐步手动运行**：
```bash
# 1) V4-S10 续训（从 ckpt_last.pt step 38000 精确续训；完成后应出现 runs/v4_s10/ema_060000.pt）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v4_s10
# 2) V4C / V5（父节点 runs/v4_s10/ema_048000.pt；首次启动时 sha 写入 PARENT.json）
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v4c12k_from48k_s10
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v5warm12k_from48k_s10
# 3) V3-T1（缓存已就绪）
UO_SMOKE_SCENE=dunnottar CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v3t1_s10 --smoke profile
CUDA_VISIBLE_DEVICES=0,1,2,3 bash experiments/exp_unified_outdoor/scripts/run_arm.sh v3t1_s10
# 4) 固定状态评估（每张 GPU 一个进程；NAME / CKPT 换成对应模型），然后与 B2 / E2 配对分析
source experiments/exp_density_ab/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 $PY experiments/exp_unified_outdoor/scripts/uo_fixed.py --name V4S10_60k --ckpt work_dirs/exp_unified_outdoor/runs/v4_s10/ema_060000.pt --set valsel &
CUDA_VISIBLE_DEVICES=1 $PY experiments/exp_unified_outdoor/scripts/uo_fixed.py --name V4S10_60k --ckpt work_dirs/exp_unified_outdoor/runs/v4_s10/ema_060000.pt --set dev3 --stratum B0 --part 0/2 &
CUDA_VISIBLE_DEVICES=2 $PY experiments/exp_unified_outdoor/scripts/uo_fixed.py --name V4S10_60k --ckpt work_dirs/exp_unified_outdoor/runs/v4_s10/ema_060000.pt --set dev3 --stratum B0 --part 1/2 &
CUDA_VISIBLE_DEVICES=3 $PY experiments/exp_unified_outdoor/scripts/uo_fixed.py --name V4S10_60k --ckpt work_dirs/exp_unified_outdoor/runs/v4_s10/ema_060000.pt --set dev3 --stratum B2 &
wait
$PY experiments/exp_unified_outdoor/scripts/uo_analyze_fixed.py V2S10_60k V4S10_60k
```

**在新的 sbatch 作业中重启**：
- 计算节点没有 /bin/zsh，脚本用 bash；
- 传 `--gres=gpu:L40:4 -x gpu118 --no-requeue`；
- 在脚本里用 SLURM 分配的 `CUDA_VISIBLE_DEVICES` 调用 `chain_resume.sh`。

可以参照 `experiments/exp_density_ab/scripts/g3_node.sbatch` 的写法。

**续训有效性**：
- 第 1000 步的 4 卡中断 + 续训测试为 4/4 EXACT（`runs/v4_s10/resume_log_r*.jsonl`）。
- 续训时必须使用**同一份配置**（run_arm.sh 每次都会记录 config sha 和 git HEAD）、同一个训练器、4 张 GPU（每 rank 16 视角）。
- GPU 数值计算的非确定性使续训不保证逐位一致，但 batch 和噪声序列一致（见 TRAINING_PROTOCOL_DRAFT D10）。

## 4. 剩余计算量（4 × L40S，独占时）
| 项 | 估计 |
|---|---|
| V4-S10 剩余 22k 步 | 约 2.7 h（0.44 s / 步） |
| V4C-12k + V5-12k | 约 1.8 h + 2.6 h |
| V3-T1 60k | 约 22 h（1.3 s / 步；profile 冒烟时会实测） |
| 固定状态评估 | 每个模型约 1–2 h，可与训练并行，但会让训练变慢约 20% |
| G3（若要做） | 每个模型 × 每种评分设置约 1.5–2 h |

**合计约 30 h × 4 GPU**。若在 3443137 上重启，作业 11-04 15:11 到期，只够完成 V4-S10、V4C、V5 和 V3-T1 的一部分。

## 5. 重启前需要决定的事项（详见 MORNING_SUMMARY.md）
1. 是否先做 V2 从零 S10 vs D100 的 10k 对照（约 1.5 h），检验 D100 训练分支。
2. 是否先估计闭环运行间方差：R10 × 多个噪声种子（约 2 h）。
3. V3-T1 的 LR：现在是统一值 2e-4，若出现非有限损失则回退到 1e-4。是否改用历史值 1e-4？需在**启动前**决定。
4. 训练完成的 V4 / V4C / V5 是否跑完整 G3。V5 / V3 的固定状态评估还需要接入各自的推理核心（`v5_core.install`、`planner_v3.install`），目前 `uo_fixed.py` 只支持 V2Model 兼容的模型（V2 / V4 / V4C）。

## 6. 运行中遇到的问题（避免重犯）
- **MP3D 闭环**：一张 44.5 GB 的 GPU 上不能同时跑两个规划器（各 18–21 GB）。调度必须限制每张 GPU 最多 1 个 MP3D episode（`da_g3_queue3.sh` / `tr_g3_queue.sh`）。
- **室外覆盖重算**：必须独占 `.cov` 认领，否则 replay-frame 目录会被并发争用（v2 调度的 bug）。
- **固定状态评估与训练并行**：显存可行（训练约 18.5 GB + 评估 3–12 GB），但训练会从 0.43 s / 步变慢到 0.55 s / 步。
- **内存**：训练的 4 个 rank 各加载全部数据，加上 CPU 建缓存时，峰值约 290 GB / 503 GB。
