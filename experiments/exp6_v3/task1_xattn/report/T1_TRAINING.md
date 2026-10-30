# T1 V3-VAE-XAttn — 训练记录

> 状态：**Running**（2026-10-30 11:10 启动，gpu115 4 × L40S，sbatch 作业 3435033（48 h，约 11-01 11:09 到期），与 T0 并行）。协议 `../../protocol.md` v1.0；配置 `../../configs/train_v3.json` / `model_v3.json`（`xattn.enabled = true` 为唯一差别；sha256 见 `../../configs/FROZEN_SHA256.json`）；代码提交 `2021604`（代码与 `7de02cc` 逐文件一致，`2021604` 只移动了启动脚本副本）。

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

## 3. 评估点（`eval.jsonl`）与选择分数
| 步 | dev FM (VALSEL) | ES in (VALSEL) | ES out (S-c) | 选择分数 | 备注 |
|---|---|---|---|---|---|
| 0 | 1.9339 | 17.24 | 19.60 | — | dev FM out 1.9461 |

## 4. 事件
- 11:10 `train_chain.sh` 检测到 3435033 运行 → T1 在 gpu115 启动；11:18 数据加载完成（8 min，比 T0 慢：首次读缓存文件）；11:24 step-0 评估完成，开始计步。
