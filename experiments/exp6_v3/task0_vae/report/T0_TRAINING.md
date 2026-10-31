# T0 V3-VAE — 训练记录

> 状态：**Completed**（2026-10-30 06:24 → 22:18:48，EXIT 0，40,000 / 40,000 步，无早停；gpu117 4 × L40S，交互分配 3429774；墙钟 15.9 h ≈ 63.7 GPU-h）。**选中 checkpoint：EMA @40000**（冻结规则，分数 1.3803，单调下降；`task0_vae/checkpoints/final.pt`，sha256 `7ed0ddd72bb9…`，`final.json`）。协议 `../../protocol.md` v1.0；配置 `../../configs/train_v3.json` / `model_v3.json`（sha256 见 `../../configs/FROZEN_SHA256.json`）；代码提交 `7de02cc`。

## 1. 运行信息
| 项 | 值 |
|---|---|
| 输出目录 | `work_dirs/exp6_v3/task0_vae/runs/main/`（`log.jsonl` 每 50 步、`eval.jsonl` 每 5k 步、`checkpoints/`） |
| 日志 | `work_dirs/exp6_v3/task0_vae/logs/train.log` |
| 启动方式 | `work_dirs/exp6_v3/train_chain.sh` → `train_arm.sh t0`（`torchrun --standalone --nproc_per_node 4`）；06:19 首次启动因脚本写错目录未运行，06:24 重启（无任何训练发生在此之前） |
| 数据 | 室内 train 66,803 视角 / VALSEL 10,473；室外 train 25,225 / S-c 31,027；4× 缓存常驻内存（rank 0：9.5 + 1.3 + 1.2 + 1.2 GB） |
| 每步 | 全局 64 视角 = 4 rank × (8 MP3D + 8 Macarons++)，来自 1 + 1 个状态 |

### Run manifest（`run_none_1793363123.json`，顶层摘录；完整内容见文件）
| 键 | 值 |
|---|---|
| `arm` | `t0` |
| `config` | `train_v3.json` |
| `config_sha256` | `a076e0880f20730c0b470a7fee4221badc20c2f4bd5406ef383c63fc63eff2e8` |
| `model_cfg` | ⟨9 项，见文件⟩ |
| `model_cfg_sha256` | `609491ac3fc47c655e8f6c588de59f6f82fa554f43cd489ebf87e3a5d7f0d60c` |
| `train_cfg` | ⟨30 项，见文件⟩ |
| `overrides` | `[]` |
| `world_size` | `4` |
| `smoke` | `none` |
| `git` | `{"sha": "7de02cc1edc4c390d7a65cde93ee8c34a40d242a", "dirty": true, "diff_sha256": "ef63d777a301a1116c286469c0a958af4f4e332ce35b5636f2c64e7e3…` |
| `common_init_sha256` | `6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513` |
| `params` | `{"total": 23539622, "xattn": 0, "vae_decoder_train_only": 585188, "inference": 22954434}` |
| `data_counts` | `{"in_train": 228, "in_val": 9, "out_train": 138, "out_sc": 42}` |
| `views` | `{"in_train": 66803, "in_val": 10473, "out_train": 25225, "out_sc": 31027}` |
| `data_files` | ⟨4 项，见文件⟩ |
| `data_manifest_sha256` | `9f432c3f943a891ab1d0090745f2bbc39534e49a7858577fa611bace6c25d3aa` |
| `host` | `gpu117` |
| `gpu` | `NVIDIA L40S` |
| `slurm_job` | `null` |
| `cuda_visible` | `0,1,2,3` |
| `torch` | `2.4.1` |
| `started` | `Fri Oct 30 06:25:23 2026` |
| `resumed_at` | `0` |

**Manifest 核对（06:40）**
- `config_sha256` a076e088… = `configs/FROZEN_SHA256.json` 中 `train_v3.json` 的哈希；`train_cfg` 与冻结文件逐键相同。
- `model_cfg_sha256` 609491ac… 是内存字典（`json.dumps(sort_keys=True)`）的哈希，不是文件字节哈希（d42e5d26…）；已核对 `model_cfg` == 冻结 `model_v3.json`（T0 的 `xattn.enabled = false` 与文件一致，无任何键不同）。
- `git.sha` = `7de02cc`，`dirty = true` 仅因启动时文档（`STATUS.md`、`reports/*`、`experiments/README.md`）未提交；`seen2scene/`、`configs/`、`shared/`、`scripts/`、`tests/` 与 `7de02cc` 逐文件一致（`2021604` 只移动了启动脚本副本）。
- `slurm_job = null`：shell 未导出 `SLURM_JOB_ID`；实际分配为 gpu117 交互作业 3429774。

## 2. 进度与指标（每个评估点追加）
| 时间 | 步 | loss | FM | Recon | KL/unit | s/步 | 峰值显存 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 06:30:52 | 0（评估） | — | dev FM 1.9339 | — | — | — | — | step-0 评估 5.5 min；ES in 17.24 / out 19.60（未训练基线） |
| 06:30:55 | 1 | 3.675 | 1.932 | 1.742 | 0.091 | 1.84 | 11.1 GB | grad-norm 10.8；与 G0 resume 测试第 1 步 3.6745 相同（确定性） |
| 06:32 | 50 | 3.599 | 1.857 | 1.566 | 0.105 | 1.32 | 30.6 GB | grad-norm 19.1；按 1.32 s/步 + 9 次评估估计 ≈ 15.5 h → 约 22:00 结束 |
| 08:36 | 5300 | 0.189 | 0.047 | 0.079 | 0.374 | 1.37 | 33.1 GB | grad-norm 1.7；lr 9.70e-05；β 0.020；剩余 34700 步 ≈ 13.8 h → 10-30 22:26 |
| 10:30 | 10100 | 0.147 | 0.075 | 0.062 | 0.318 | 1.35 | 33.2 GB | grad-norm 2.4；lr 8.72e-05；β 0.020；剩余 29900 步 ≈ 11.8 h → 10-30 22:19 |
| 12:31 | 15200 | 0.131 | 0.096 | 0.038 | 0.340 | 1.38 | 33.2 GB | grad-norm 2.5；lr 7.07e-05；β 0.020；剩余 24800 步 ≈ 9.9 h → 10-30 22:28 |
| 14:26 | 20050 | 0.108 | 0.062 | 0.036 | 0.304 | 1.33 | 33.2 GB | grad-norm 3.5；lr 5.18e-05；β 0.020；剩余 19950 步 ≈ 7.7 h → 10-30 22:11 |
| 16:27 | 25150 | 0.108 | 0.073 | 0.031 | 0.309 | 1.34 | 33.2 GB | grad-norm 1.8；lr 3.17e-05；β 0.020；剩余 14850 步 ≈ 5.8 h → 10-30 22:16 |
| 16:37 | 25600 | 0.103 | 0.065 | 0.028 | 0.309 | 1.36 | 33.2 GB | grad-norm 1.8；lr 3.00e-05；β 0.020；剩余 14400 步 ≈ 5.7 h → 10-30 22:19 |
| 18:22 | 30000 | 0.096 | 0.056 | 0.030 | 0.316 | 1.35 | 33.2 GB | grad-norm 1.2；lr 1.54e-05；β 0.020；剩余 10000 步 ≈ 3.9 h → 10-30 22:18 |
| 20:22 | 35050 | 0.097 | 0.066 | 0.025 | 0.236 | 1.37 | 33.2 GB | grad-norm 0.8；lr 3.92e-06；β 0.020；剩余 4950 步 ≈ 2.0 h → 10-30 22:21 |
| 22:20 | 40000 | 0.098 | 0.050 | 0.028 | 0.290 | 1.34 | 33.2 GB | grad-norm 0.3；lr 1.62e-13；β 0.020；剩余 0 步 ≈ 0.0 h → 10-30 22:20 |

## 3. 评估点（`eval.jsonl`）与选择分数
| 步 | dev FM (VALSEL) | ES in (VALSEL) | ES out (S-c) | 选择分数 | 备注 |
|---|---|---|---|---|---|
| 0 | 1.9339 | 17.24 | 19.60 | — | dev FM out 1.9461 |
| 5000 | 0.1312 | 9.58 | 9.00 | 2.0000 | dev FM out 0.0443 |
| 10000 | 0.1081 | 8.11 | 7.86 | 1.7189 | dev FM out 0.0342 |
| 15000 | 0.1004 | 7.36 | 7.40 | 1.5910 | dev FM out 0.0310 |
| 20000 | 0.0958 | 6.96 | 7.04 | 1.5086 | dev FM out 0.0294 |
| 25000 | 0.0931 | 6.73 | 6.84 | 1.4625 | dev FM out 0.0283 |
| 30000 | 0.0913 | 6.51 | 6.57 | 1.4103 | dev FM out 0.0277 |
| 35000 | 0.0904 | 6.38 | 6.51 | 1.3882 | dev FM out 0.0273 |
| 40000 | 0.0900 | 6.37 | 6.44 | 1.3803 | dev FM out 0.0272；**当前最优** |

## 4. 事件
- 22:18:48 训练结束（EXIT 0，第 40k 步 loss 0.098、FM 0.050、Recon 0.028、KL/unit 0.29）；最终评估 @40k：dev FM 0.0900 / 0.0272，ES in 6.37 / out 6.44，hit ratio 1.006 / 0.969，K=4 离散度 7.17 / 5.99，后验 16/16 活跃、μ std 0.51。
- 22:19:48 `select_final.py t0`：选择分数 2.000 → 1.719 → 1.591 → 1.509 → 1.463 → 1.410 → 1.388 → **1.380（@40k）**，选中 `ema_040000.pt` → `checkpoints/final.pt`。
- 06:24 启动；06:25 数据加载完成（2 min）；06:30 step-0 评估完成；06:31 开始计步。
- 08:33 第 5k 步评估：dev FM 1.934 → 0.131（VALSEL）、0.044（S-c）；ES in 17.24 → 9.58、out 19.60 → 9.00；hit ratio 0.92 / 0.91、Brier 0.11 / 0.049；后验健康：active units 16 / 16、μ 标准差 0.56 / 0.46、KL/unit 0.57 / 0.37（β 已达 0.02）；重建 acc EMPTY / UNKNOWN / BAND 0.98 / 0.98 / 0.996，SDF MAE 0.08。checkpoint：`ckpt_last.pt` 360 MB，`ema_005000.pt` 90 MB。选择分数基准（5k）= 2.0。
