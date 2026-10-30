# EXP6 V3 — 全局状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-30 06:27**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待（排队或依赖前项）· ❌ 失败（后面跟修复行）· ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| P4-T0 | **V3-VAE 正式训练**（40k 步，4 卡 DDP，每 5k 步评估 + EMA，每 1k 步 ckpt_last） | gpu117 GPU 0–3 | 3429774（交互分配） | `work_dirs/exp6_v3/task0_vae/logs/train.log`、`runs/main/log.jsonl`、`eval.jsonl` | 06:24（06:19 的首次启动因脚本写错目录未运行，06:24 重启） | 约 21:30–22:30（估计 1.3–1.4 s/步 + 9 次评估） |
| P4 链 | `train_chain.sh`：T0 结束后若第二组 4 卡仍未启动则 T1 在 gpu117 顺序跑；若 3435033 / 3435072 任一启动则 T1 立即在该节点并行 | — | — | `work_dirs/exp6_v3/logs/train_chain.log` | 06:24 | — |

**GPU / 作业**
| 分配 | 作业 | 节点 / 卡 | 状态 | 用途 |
|---|---|---|---|---|
| 交互分配 | 3429774 `vscode-dev` | gpu117，4 × L40S 46 GB | 🔄 到 2026-11-01 14:50 | 缓存构建、G0、T0 训练（若第二组未到则 T0 → T1 顺序） |
| 第二组 4 卡 | 3435033（48 h）/ 3435072（24 h） | 4 × L40S | ⏳ 排队，调度器预计 10-31 22:46 启动 | 启动后 T1 并行训练、G3 |

---

## P0 存储清理 — ✅（2026-10-30 05:00–05:15，用户执行 `work_dirs/cleanup_2026-10-30{,_B}.sh`）
| 编号 | 内容 | 结果 |
|---|---|---|
| P0.1 | A 段：exp3b 渲染 / 重放帧、各实验 smoke / 临时输出、Task 3 B0 副本 → 指向 Task 2 的符号链接、Task 3 B1 / E2S / E2D 与 Task 1 闭环的 svc 记录（episode / planner 日志 / 指标保留） | ✅ 123 项（日志 `work_dirs/exp6_v3/logs/cleanup_2026-10-30.txt`） |
| P0.2 | B 段：EXP4 pilot 的 traj / cond、EXP4 消融 abl_1 / abl_2、EXP1 3D-FRONT GT、exp3b T5、exp_vis 的原始张量文件（npz / pt / ply / npy）；结果文件（json / jsonl / csv / png / log）保留；MAGICIAN 官方复现整体保留 | ✅ 释放 224 GB |
| P0.3 | 结果：`work_dirs` 630 GB → 333 GB；保留项核对（Task 2 闭环 + svc、Task 3 B2 svc、E2 D100 分片、V3 缓存、权重、MAGICIAN 复现）见本次检查 | ✅ |

## P1 V2 归档与 Push — ✅（2026-10-29）
| 编号 | 内容 | 状态 |
|---|---|---|
| P1.1 | 审计 git / 远端 / 未跟踪文件；无密钥 | ✅ |
| P1.2 | `pytest tests/futureview` 13 passed；包导入 | ✅ |
| P1.3 | 4 个提交 `[V2][chore/archive/test/docs]` → `efd7e14`，push 并核对远端 | ✅ |

## P2 EXP6 初始化 — ✅（2026-10-30）
| 编号 | 内容 | 状态 |
|---|---|---|
| P2.1 | 目录 `experiments/exp6_v3/{configs,shared,tests,task0_vae,task1_xattn,reports}` + `work_dirs/exp6_v3/...` | ✅ |
| P2.2 | `README.md`、`protocol.md` v0.9 草案、`configs/{model_v3,train_v3}.json` 草案、`reports/V3_DEVIATIONS.md`（#1–#6）、`V3_RESOURCES.md`、`V3_COMBINED_REPORT.md` 骨架 | ✅ |
| P2.3 | `experiments/README.md` 加 exp6_v3 行；`task0_vae/README.md`、`task1_xattn/README.md`；`T0_AUDIT.md`、`T1_AUDIT.md` | ✅ |
| P2.4 | git 提交 `[V3][setup]` | ✅ `7de02cc`（47 个文件），已 push 到 `origin/seen2scene`（2026-10-30 06:19） |

## P3 G0 工程审计、代码与测试
### P3.1 共享代码 — ✅
| 编号 | 模块 | 状态 |
|---|---|---|
| P3.1.1 | `seen2scene/v3/partial_tsdf.py`（4× 聚合、`structure_lean`、blob、重建目标、pack） | ✅（两次修正：显存精简、0.999 TRUNC 边界） |
| P3.1.2 | `vae.py`（Encoder / Decoder / KL / 后验统计） | ✅ |
| P3.1.3 | `cross_attention.py`（场景记忆、视锥选择、Cross-Attn 块） | ✅ |
| P3.1.4 | `flow_unet.py`、`model.py`（V3Model、sample、公共初始化哈希） | ✅（射线编码器通道数修正 1 次） |
| P3.1.5 | `data.py`（分片 + 缓存、采样规则、build_cond） | ✅ |
| P3.1.6 | `losses.py`、`trainer.py`（DDP、EMA、resume、manifest、冒烟模式、`--override`） | ✅（冒烟批量按 4 卡模拟修正 1 次） |
| P3.1.7 | `predictor.py`、`eval_dev.py` | ✅ |
| P3.1.8 | 实验脚本：`shared/build_tsdf_cache.py`、`eval_fixed.py`、`analyze_g12.py`、`planner_v3.py`、`v3_{in,out}_planner.py`、`sim_{in,out}_v3.py`、`out_cov_v3.py`、`analyze_g3.py`；`task{0,1}/scripts/train.py` | ✅（语法检查；运行验证见 P3.3–P3.9、P5） |

### P3.2 4× TSDF 缓存（417 条轨迹）— ✅（05:59）
| 编号 | 内容 | 状态 |
|---|---|---|
| P3.2.1 | 冒烟：MP3D 17DRP5sb8fy_L0/rnv_p2（150 状态，103 MB，172 s）、bannerman cg-A-0000（24.6 MB，23 s）；分片粗网格逐位核对通过 | ✅ |
| P3.2.2 | 第 1 次全量启动（12 进程）→ 8 次 OOM | ❌ → 改为逐量池化 / 半精度 / `structure_lean`（偏差 #5） |
| P3.2.3 | 第 2 次启动 → MP3D 核对 STOP（`structure_lean` 漏 0.999 TRUNC 边界） | ❌ → 修正，在 7y3sRwLe3Va_L0 t=0–2 逐位核对通过 |
| P3.2.4 | 第 3 次启动（00:20，13 进程） | ✅ 室外 180/180（约 02:40）；MP3D 236/237（05:54）；1 个进程 OOM 退出 |
| P3.2.5 | 构建结束后清理失效 claim（235 个）+ 补漏一遍 | ✅ 417/417（05:59）；缓存共 49 GB（MP3D 41 + 室外 7.9） |

### P3.3 单元测试 — ✅（06:10，6 passed，32 s）
| 编号 | 测试 | 状态 |
|---|---|---|
| P3.3.1 | `tests/test_partial_tsdf.py`（三态、填充、blob 往返、重建目标） | ✅ 2/2（修复史：CPU 无 half 池化 → float32；extent 浮点比较 → isclose） |
| P3.3.2 | `tests/test_model_shapes.py`（形状、FM→μ / Recon→μ,logvar / KL 梯度、T0/T1 公共初始化一致、`W_o` 零、Query 随相机） | ✅ 4/4（修复史：零初始化输出层使速度恒 0 → 扰动后检查；T1 的 `W_o` 检查需 retain_graph） |

### P3.4 真实数据检查 — ✅（`tests/g0_real_data_check.py`，GPU 3）
缓存 = 实时聚合（bannerman t=0,1,10,55,99 逐位）；`structure_lean` = `structure()`；粗网格 = E2 分片；T0/T1 前后向有限、梯度到 μ 头（6.15）/ logvar 头（3.23）；峰值 14.5 GB；FM 对射线编码器 / Cross-Attn 在第 0 步梯度为 0（零初始化输出层，V2 同样），第 10 步非零（P3.5）。

### P3.5 单卡 tiny overfit（200 步，8+8 视角固定批）— ✅
| 编号 | 组 | 结果 |
|---|---|---|
| P3.5-T0 | V3-VAE | loss 3.67 → 0.62（−83%）；FM 1.81 → 0.34；Recon 1.86 → 0.21（三类准确率 → 0.97/0.98/0.98）；KL/unit 0.09 → 0.31；梯度检查 4 个模块非零 |
| P3.5-T1 | V3-VAE-XAttn | loss 3.67 → 0.60（−84%）；第 1 步与 T0 逐位相同（`W_o` 零初始化）；梯度检查通过 |
| 失败记录 | 第 1 次 OOM（单卡按 world=1 取 32+32 视角） | ❌ → 冒烟按 4 卡每卡批模拟（8+8） |

### P3.6 4 卡 DDP profile（100 步；与缓存构建共享 GPU，非干净计时）— ✅
| 编号 | 组 | 结果 |
|---|---|---|
| P3.6-T0 | V3-VAE | 1.45 s/步（p90 1.64），每卡峰值 16.3 GB，参数 23,539,622 |
| P3.6-T1 | V3-VAE-XAttn | 1.46 s/步（p90 1.75），每卡峰值 17.8 GB，参数 23,730,986（+191,364 Cross-Attn）；公共初始化哈希 = T0 |

### P3.7–P3.10 — ⏳
| 编号 | 内容 | 状态 |
|---|---|---|
| P3.7 | 补漏构建 | ✅ 1 条轨迹重建（05:59） |
| P3.8 | 干净 4 卡 profile T0 / T1（100 步） | ✅ T0 0.94 s/步（p90 1.08）、T1 0.96 s/步（p90 1.10）；每卡峰值 15.8 GB；2.6 h / 10k 步（4 卡） |
| P3.9 | DDP resume 测试（A 60 步 vs B 30+30 步，比较参数与日志） | ❌ 第 1 次：B2 恢复时 `set_rng_state` 类型错误 → 改为按 (seed, rank, step0) 重播种（偏差 #8）；✅ 第 2 次（06:18）：B 从 30 恢复到 60，A/B 第 60 步 dev FM 1.9049 相同、ES 21.753 相同，EMA 最大差 2.7e-6（恢复后样本流按设计不同） |
| P3.11 | 大场景单步微基准（dunnottar 1.83 s / 27.4 GB；其余 0.7–0.9 s / 13 GB） | ✅ 06:20 |
| P3.10 | 冻结 `protocol.md` v1.0：40k 步、lr 1e-4（预热 1k，余弦）、λ_rec 1.0、β 0.02（5k 预热）、每 5k 评估、选择规则与早停规则；`configs/FROZEN_SHA256.json` | ✅ 06:10 冻结；提交 `7de02cc` |

## P4 从零训练 — 🔄
| 编号 | 组 | 计划 | 状态 |
|---|---|---|---|
| P4-T0 | V3-VAE | gpu117 4 × L40S（作业 3429774），`torchrun` 4 进程，40k 步；每 1k 步 ckpt、每 5k 步评估 + EMA（step 0 / 5k / … / 40k） | 🔄 06:24 启动（数据加载约 2 min 后开始计步）；预计 21:30–22:30 结束 |
| P4-T1 | V3-VAE-XAttn | 第二组 4 卡（3435033 / 3435072 任一启动即并行）否则 T0 结束后在 gpu117 顺序；配置与 T0 完全相同（`7de02cc`） | ⏳ 由 `train_chain.sh` 自动启动 |
| P4.3 | 训练曲线、dev 指标、checkpoint 选择（冻结规则）→ `T0_TRAINING.md` / `T1_TRAINING.md` | ⬜ |

## P5 评估与报告 — ⬜
| 编号 | 内容 | 状态 |
|---|---|---|
| P5.1-T0 / P5.1-T1 | G1 / G2 固定状态（VALSEL 90 + Dev-3 275；`shared/eval_fixed.py`、`analyze_g12.py`） | ⬜ |
| P5.2 | G2 → G3 放行判定（协议 §4） | ⬜ |
| P5.3-T0 / P5.3-T1 | G3 闭环（MP3D 10 + Macarons++ 30，各组；`sim_{in,out}_v3.py`、`out_cov_v3.py`、`analyze_g3.py`） | ⬜ |
| P5.3.3 | G3 参照组：B0 / B2 / E2-D100（Task 3、E2）+ **MAGICIAN MAG-G / MAG-B（Task 2 共享协议闭环）** 纳入同一表（AUC、Final、Cov@k、逐场景）；`analyze_g3.py` 读取 Task 2 的 MAG 闭环（tag 前缀不同） | ⬜ |
| P5.4 | 任务报告 `T{0,1}_G12.md`、`T{0,1}_G3.md`；`reports/V3_COMBINED_REPORT.md`、`V3_COMPARISON_TABLES.md`、`V3_RESOURCES.md` | ⬜ |
| P5.5 | git 提交 `[V3][task0]`、`[V3][task1]`、`[V3][eval]` 并 push | ⬜ |
