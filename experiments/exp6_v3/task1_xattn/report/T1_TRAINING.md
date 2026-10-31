# T1 V3-VAE-XAttn — 训练记录

> 状态：**Completed**（2026-10-30 11:10 → 2026-10-31 03:26:18，EXIT 0，40,000 / 40,000 步，无早停；gpu115 4 × L40S，sbatch 作业 3435033；墙钟 16.3 h ≈ 65 GPU-h，含 8 min 数据加载）。**选中 checkpoint：EMA @40000**（冻结规则，分数 1.3946，单调下降；`task1_xattn/checkpoints/final.pt`，sha256 `639c9d3ae3f3…`）。协议 `../../protocol.md` v1.0；配置 `../../configs/train_v3.json` / `model_v3.json`（`xattn.enabled = true` 为唯一差别；sha256 见 `../../configs/FROZEN_SHA256.json`）；代码提交 `2021604`（代码与 `7de02cc` 逐文件一致，`2021604` 只移动了启动脚本副本）。

## 1. 运行信息
| 项 | 值 |
|---|---|
| 输出目录 | `work_dirs/exp6_v3/task1_xattn/runs/main/`（`log.jsonl` 每 50 步、`eval.jsonl` 每 5k 步、`ckpt_last.pt` / `ema_*.pt`） |
| 日志 | `work_dirs/exp6_v3/task1_xattn/logs/train.log`；`logs/srun_train_t1.log` |
| 启动方式 | `train_chain.sh` 轮询到 3435033 RUNNING 后 `srun --jobid=3435033 --overlap -N1 -n1 bash train_arm.sh t1`（`torchrun --standalone --nproc_per_node 4`）；SLURM 分配 GPU 0–3（`SLURM_STEP_GPUS=0,1,2,3`），32 CPU |
| 数据 | 与 T0 完全相同：室内 train 66,803 视角 / VALSEL 10,473；室外 train 25,225 / S-c 31,027；`data_manifest_sha256` 与 T0 相同 ✅ |
| 初始化 | 公共模块 sha256 与 T0 相同 ✅（`6a9d6071a225…`）；Cross-Attn 191,364 参数独立初始化，`W_o` 零 → 第 1 步损失 3.674514578 与 T0 逐位相同（跨节点确定性成立） |
| 每步 | 全局 64 视角 = 4 rank × (8 MP3D + 8 Macarons++)，来自 1 + 1 个状态 |

### Run manifest（`run_none_1793380713.json`，顶层摘录；完整内容见文件）
| 键 | 值 |
|---|---|
| `arm` | `t1` |
| `config` | `train_v3.json` |
| `config_sha256` | `a076e0880f20730c0b470a7fee4221badc20c2f4bd5406ef383c63fc63eff2e8` |
| `model_cfg` | ⟨9 项，见文件⟩ |
| `model_cfg_sha256` | `3deae01f0a322f73ac633a5b681f72dd4b77f6e3a815264183f9fa2b577ed9fa` |
| `train_cfg` | ⟨30 项，见文件⟩ |
| `overrides` | `[]` |
| `world_size` | `4` |
| `smoke` | `none` |
| `git` | `{"sha": "2021604931d6731462e16c6accee45fff55a2d57", "dirty": true, "diff_sha256": "bc222222c03fff62b0b7192fd9a69e63988bce4a4252c1ebbcea05821…` |
| `common_init_sha256` | `6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513` |
| `params` | `{"total": 23730986, "xattn": 191364, "vae_decoder_train_only": 585188, "inference": 23145798}` |
| `data_counts` | `{"in_train": 228, "in_val": 9, "out_train": 138, "out_sc": 42}` |
| `views` | `{"in_train": 66803, "in_val": 10473, "out_train": 25225, "out_sc": 31027}` |
| `data_files` | ⟨4 项，见文件⟩ |
| `data_manifest_sha256` | `9f432c3f943a891ab1d0090745f2bbc39534e49a7858577fa611bace6c25d3aa` |
| `host` | `gpu115` |
| `gpu` | `NVIDIA L40S` |
| `slurm_job` | `3435033` |
| `cuda_visible` | `0,1,2,3` |
| `torch` | `2.4.1` |
| `started` | `Fri Oct 30 11:18:33 2026` |
| `resumed_at` | `0` |

**Manifest 核对（12:35）**
- `config_sha256` == 冻结 `train_v3.json` ✅；`train_cfg` 与冻结文件逐键相同 ✅。
- `model_cfg` == `model_cfg('t1', 冻结文件)` ✅（`xattn.enabled = True`，其余键与 T0 相同）。
- `git.sha` = `2021604`，`dirty = True`（启动时仅文档未提交；代码与 `7de02cc` 一致，见 T0 记录）。
- 参数：{"total": 23730986, "xattn": 191364, "vae_decoder_train_only": 585188, "inference": 23145798}。
- `slurm_job` = 3435033；`host` = gpu115；`started` = Fri Oct 30 11:18:33 2026。

## 2. 进度与指标（每个评估点追加）
| 时间 | 步 | loss | FM | Recon | KL/unit | s/步 | 峰值显存 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 11:24:18 | 0（评估） | — | dev FM 1.9339 | — | — | — | — | step-0 评估 5.75 min；ES in 17.24 / out 19.60（= T0 的未训练基线，逐位相同） |
| 11:24:25 | 1 | 3.675 | 1.932 | 1.742 | 0.091 | — | — | grad-norm 10.8；与 T0 第 1 步逐位相同 |
| 12:33 | 3000 | 0.229 | 0.161 | 0.112 | 0.417 | 1.36 | 32.7 GB | grad-norm 2.5；lr 9.94e-05；β 0.012；剩余 37000 步 ≈ 14.7 h → 10-31 03:14 |
| 13:29 | 5200 | 0.186 | 0.072 | 0.070 | 0.296 | 1.37 | 33.1 GB | grad-norm 4.2；lr 9.72e-05；β 0.020；剩余 34800 步 ≈ 13.9 h → 10-31 03:24 |
| 15:29 | 10200 | 0.148 | 0.119 | 0.058 | 0.358 | 1.38 | 33.2 GB | grad-norm 3.7；lr 8.69e-05；β 0.020；剩余 29800 步 ≈ 11.9 h → 10-31 03:25 |
| 16:37 | 13200 | 0.133 | 0.078 | 0.042 | 0.251 | 1.37 | 33.2 GB | grad-norm 2.2；lr 7.77e-05；β 0.020；剩余 26800 步 ≈ 10.8 h → 10-31 03:23 |
| 17:29 | 15200 | 0.130 | 0.062 | 0.041 | 0.340 | 1.40 | 33.2 GB | grad-norm 6.5；lr 7.07e-05；β 0.020；剩余 24800 步 ≈ 10.1 h → 10-31 03:34 |
| 19:30 | 20200 | 0.114 | 0.083 | 0.042 | 0.413 | 1.35 | 33.2 GB | grad-norm 2.7；lr 5.12e-05；β 0.020；剩余 19800 步 ≈ 7.8 h → 10-31 03:18 |
| 21:30 | 25200 | 0.106 | 0.046 | 0.035 | 0.313 | 1.37 | 33.2 GB | grad-norm 1.6；lr 3.15e-05；β 0.020；剩余 14800 步 ≈ 5.9 h → 10-31 03:23 |
| 23:30 | 30200 | 0.096 | 0.085 | 0.027 | 0.333 | 1.38 | 33.2 GB | grad-norm 1.1；lr 1.48e-05；β 0.020；剩余 9800 步 ≈ 3.9 h → 10-31 03:26 |
| 00:39 | 33150 | 0.102 | 0.071 | 0.030 | 0.337 | 1.39 | 33.2 GB | grad-norm 0.7；lr 7.42e-06；β 0.020；剩余 6850 步 ≈ 2.8 h → 10-31 03:28 |
| 01:31 | 35150 | 0.096 | 0.063 | 0.033 | 0.359 | 1.39 | 33.2 GB | grad-norm 2.2；lr 3.77e-06；β 0.020；剩余 4850 步 ≈ 2.0 h → 10-31 03:29 |
| 03:30 | 40000 | 0.098 | 0.050 | 0.027 | 0.291 | 1.36 | 33.2 GB | grad-norm 0.4；lr 1.62e-13；β 0.020；剩余 0 步 ≈ 0.0 h → 10-31 03:30 |

## 3. 评估点（`eval.jsonl`）与选择分数
| 步 | dev FM (VALSEL) | ES in (VALSEL) | ES out (S-c) | 选择分数 | 备注 |
|---|---|---|---|---|---|
| 0 | 1.9339 | 17.24 | 19.60 | — | dev FM out 1.9461 |
| 5000 | 0.1308 | 9.58 | 8.95 | 2.0000 | dev FM out 0.0441 |
| 10000 | 0.1079 | 8.14 | 7.85 | 1.7263 | dev FM out 0.0340 |
| 15000 | 0.1003 | 7.40 | 7.61 | 1.6219 | dev FM out 0.0310 |
| 20000 | 0.0959 | 6.99 | 7.11 | 1.5237 | dev FM out 0.0294 |
| 25000 | 0.0931 | 6.74 | 6.83 | 1.4663 | dev FM out 0.0283 |
| 30000 | 0.0914 | 6.52 | 6.60 | 1.4187 | dev FM out 0.0277 |
| 35000 | 0.0904 | 6.45 | 6.56 | 1.4054 | dev FM out 0.0273 |
| 40000 | 0.0900 | 6.42 | 6.48 | 1.3946 | dev FM out 0.0272；**当前最优** |

## 4. 事件
- 03:26:18 训练结束（EXIT 0，第 40k 步 loss 0.098、FM 0.050、Recon 0.027、KL/unit 0.29 —— 与 T0 的终点几乎相同）；最终评估 @40k：dev FM 0.0900 / 0.0272（= T0），ES in 6.43 / out 6.48（T0 6.37 / 6.44），hit ratio 1.010 / 0.971，K=4 离散度 7.13 / 5.95，后验 16/16 活跃、μ std 0.51。
- 03:27:03 `select_final.py t1`：分数 2.000 → 1.726 → 1.622 → 1.524 → 1.466 → 1.419 → 1.405 → **1.395（@40k）**（T0 对应 1.380），选中 `ema_040000.pt` → `checkpoints/final.pt`。
- 03:29:08 闭环冒烟通过（室内 / 室外 2 步）→ T1 的 G3 队列在 gpu115 启动（40 项）。
- 整个训练过程中 T1 与 T0 在每个评估点的差都在 ±1% 内：Cross-Attention 在 40k 步内没有改变 dev 指标（见 §5 权重诊断）。
- 11:10 `train_chain.sh` 检测到 3435033 运行 → T1 在 gpu115 启动；11:18 数据加载完成（8 min，比 T0 慢：首次读缓存文件）；11:24 step-0 评估完成，开始计步。

## 5. 权重诊断（`final.pt` vs T0 `final.pt`，CPU，2026-10-31 03:35；事后分析，不改任何结果）
- Cross-Attention 参数 191,364（14 个张量）。输出投影 `wo.weight` 从零初始化长到 **rms 0.0060**（norm 1.54，absmax 0.031），`wo.bias` rms 0.004；作为参照，同一瓶颈层的自注意力 `mid.1.qkv` / `mid.1.proj` rms 0.035、残差卷积 rms 0.013。即 Cross-Attention 支路**不是零**，但其输出投影尺度只有相邻投影的约 1/6。
- `wq` rms 0.035、`wk` / `wv` rms 0.069、`q_pe` rms 0.058、`ln_q` / `ln_k` 权重 ≈ 0.98（几乎未动，bias ≈ 0）。
- **共享参数与 T0 的相对差异 ‖θ_T1 − θ_T0‖ / ‖θ_T0‖ = 0.29**（U-Net up 0.31、down 0.29、mid 0.25；VAE 编码器 0.12、解码器 0.12；Ray Encoder 0.11；时间嵌入 0.05；U-Net 输出卷积 0.01）。两组用同一数据流、同一种子、同一初始化，差异只来自 Cross-Attention 梯度的反馈 → T1 在参数空间里确实走了另一条路，但 dev 指标（FM、ES、hit ratio、离散度）与 T0 在每个评估点都在 ±1% 内。
- 读法：**"W_o 停在零所以 T1 ≡ T0"的假设不成立**；更可能是在 40k 步、当前 token 预算（≤ 512 个 8× latent 单元、视锥选择）与 2D 瓶颈分辨率（8 × 16）下，Cross-Attention 提供的场景信息与 Ray Encoder 已沿射线采样到的 μ 特征高度冗余。激活层面的贡献比（Cross-Attn 输出范数 / 残差流范数）留待 G1 / G2 报告中用真实状态测量。
