# EXP6 V3 — 全局状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-10-31 10:38（V3 + Ablation 全部完成）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待（排队或依赖前项）· ❌ 失败（后面跟修复行）· ⬜ 未开始

## 0. 此刻在跑什么
**2026-10-31 09:30 起进入 P6 Ablation Study**（任务书 `EXP6_V3_ABLATION_ONLY_PLAN.md`；详细板 `ablation_study/STATUS.md`）。V3 正式实验已于 04:15 完成；最终报告 `reports/V3_COMBINED_REPORT.md`、`reports/V3_COMPARISON_TABLES.md`。GPU 分配仍在：gpu117 交互作业 3429774（11-01 14:50 到期）、gpu115 sbatch 3435033（11-01 11:09 到期，空闲）、3435072（24 h，排队中）——可按需 `scancel`。下表为本轮的历史运行记录：
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 预计结束 |
|---|---|---|---|---|---|---|
| P4-T0 | **V3-VAE 正式训练** | gpu117 | 3429774 | `task0_vae/logs/train.log` | 06:24 | ✅ 22:18 结束，40k 步，final.pt = EMA @40k |
| P4-T1 | **V3-VAE-XAttn 正式训练** | gpu115 | 3435033 | `task1_xattn/logs/train.log` | 11:10 | ✅ 03:26 结束，40k 步，final.pt = EMA @40k |
| P5-G3-T0 | **T0 的 G3 闭环**：`post_arm.sh t0` → `final.pt` = EMA @40k（22:19）→ 冒烟 ✅（室内 2 步 27 s、室外 2 步 28 s，决策约 1 s/步）→ 4 worker 跑 10 MP3D + 30 Macarons++ 闭环 → 室外覆盖率 | gpu117 GPU 0–3 | 3429774 | `work_dirs/exp6_v3/logs/post_t0.log`、`eval/logs/queue_g3_gpu117_*.log`、`eval/closed_loop/{indoor,outdoor}/T0/` | 22:20 | ✅ 23:07 全部完成（10 室内 + 30 室外 + 30 覆盖率，0 失败；47 min ≈ 3.1 GPU-h）。临时均值（正式分析等 T1）：MP3D AUC T0 0.311 / Final 0.463（B0 0.322 / B2 0.334 / E2D 0.364 / MAG-B 0.292）；Macarons++ AUC 0.702 / Final 0.872 / Cov@20 0.568（B0 0.729 / B2 0.730 / E2D 0.711 / MAG-G 0.704）；配对 T0−B2 dAUC −0.023 / −0.028 |
| P5-G3-T1 链 | `post_arm.sh t1`（`srun --jobid=3435033 --overlap`）：等 T1 `EXIT 0` → `final.pt` → 冒烟 → T1 的 G3 闭环 | gpu115 GPU 0–3（T1 结束后） | 3435033 | `work_dirs/exp6_v3/logs/post_t1.log` | 03:29 启动 | ✅ 04:00:48 全部完成（10 室内 + 30 室外 + 30 覆盖率，0 失败；gpu115 31 min + gpu117 03:42 起加入 ≈ 3.3 GPU-h） |
| P5-G12 链 | `post_g12.sh`：等 T0 的 G3 结束 + T1 `final.pt` → `eval_fixed.py`（VALSEL B0、Dev-3 B0 / B2，各 4 份）→ `analyze_g12.py` → gpu117 加入 T1 的 G3 队列 → 覆盖率 | gpu117 GPU 0–3 | 3429774 | `work_dirs/exp6_v3/logs/post_g12.log`、`eval_fixed/` | 03:28 第 1 轮（不完整，作废）→ 04:02 补跑 | ✅ 04:14 完成：9/9 + 15/15 + 15/15 episode，B2 回放 365/365 逐位；`analyze_g12` → `reports/g12_s_final.md`；G2 放行 T0 PASS（B）、T1 PASS（A） |
| P5-G3 分析 | `analyze_g3.py`（两组 + B0 / B2 / E2D + MAG-G / MAG-B，场景级 bootstrap） | CPU | — | `work_dirs/exp6_v3/logs/analyze_g3.log` → `reports/g3_*.md` | 04:03 | ✅ 04:03 → `reports/g3.md`：T1−T0 meaningful on one（MP3D ΔAUC +0.020、Macarons++ ΔFinal +0.011）；T0−B2 退化；T1−B2 持平（CI 含 0） |
| P4 链 | `train_chain.sh`：T0 在 gpu117；3435033 于 11:10 开始运行 → T1 已在 gpu115 并行启动（不再顺序）；两组互不等待 | — | — | `work_dirs/exp6_v3/logs/train_chain.log` | 06:24 | T0 结束后链退出 |

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

## P4 从零训练 — ✅（两组 40k 步，10-31 03:26 全部结束）
| 编号 | 组 | 计划 | 状态 |
|---|---|---|---|
| P4-T0 | V3-VAE | gpu117 4 × L40S（作业 3429774），`torchrun` 4 进程，40k 步；每 1k 步 ckpt、每 5k 步评估 + EMA（step 0 / 5k / … / 40k） | ✅ 22:18:48 结束（40k / 40k，无早停，15.9 h ≈ 63.7 GPU-h）；最终 dev FM 0.0900 / 0.0272，ES in 6.37 / out 6.44；选择分数单调降至 1.380 → **final.pt = EMA @40000**（sha `7ed0ddd7…`）；记录 `task0_vae/report/T0_TRAINING.md` |
| P4-T1 | V3-VAE-XAttn | gpu115 4 × L40S（sbatch 作业 3435033，48 h），`torchrun` 4 进程，40k 步；与 T0 并行；配置与 T0 完全相同（`7de02cc`，仅 `xattn.enabled`） | ✅ 10-31 03:26:18 结束（40k / 40k，无早停，16.3 h ≈ 65 GPU-h）；最终 dev FM 0.0900 / 0.0272，ES in 6.43 / out 6.48（T0 6.37 / 6.44）；选择分数单调降至 1.395 → **final.pt = EMA @40000**（sha `639c9d3a…`）；记录 `task1_xattn/report/T1_TRAINING.md` |
| P4.3 | 训练曲线、dev 指标、checkpoint 选择（冻结规则）→ `T0_TRAINING.md` / `T1_TRAINING.md` | ⬜ |

## P5 评估与报告 — ✅（2026-10-31 04:30）
| 编号 | 内容 | 状态 |
|---|---|---|
| P5.1-T0 / P5.1-T1 | G1 / G2 固定状态（VALSEL 90 + Dev-3 275；`shared/eval_fixed.py`、`analyze_g12.py`） | ✅ 04:14（第 1 轮 03:28–03:42 不完整，偏差 #10；补跑后 39/39 episode，B2 回放 365/365 逐位）。T1−T0：G1 ≈ 0；VALSEL regret −0.014 [−0.038, 0.000]，Early regret −0.036 [−0.094, −0.002]，Early Top-1 +0.056；Dev-3 regret +0.019（CI 含 0）。T0−B2 MAE_unknown −0.038 [−0.052, −0.025]（9/0）。报告 `task*/report/T*_G12.md`、`reports/g12_s_final.md` |
| P5.2 | G2 → G3 放行判定（协议 §4） | ✅ T0 PASS（规则 B；Dev-3 Top-1 0.548 vs 门槛 0.551 差 0.003）；T1 PASS（规则 A）→ 两组 G3 均 formal |
| P5.3-T0 / P5.3-T1 | G3 闭环（MP3D 10 + Macarons++ 30，各组；`sim_{in,out}_v3.py`、`out_cov_v3.py`、`analyze_g3.py`） | ✅ T0 22:20–23:07（3.1 GPU-h）、T1 03:29–04:00（3.3 GPU-h），80 episode 0 失败；分析 04:03 `reports/g3.md`。MP3D AUC：MAG-B 0.292 / B0 0.322 / B2 0.334 / E2D 0.364 / **T0 0.311 / T1 0.331**；Macarons++ AUC：0.685 / 0.729 / 0.730 / 0.711 / **0.702 / 0.712**。T1−T0 MP3D ΔAUC +0.020 [−0.003, 0.042]（4/1）、Macarons++ ΔFinal +0.011 [0.005, 0.017]（6/0）→ meaningful on one；T0−B2 退化；T1−B2 持平（CI 含 0）；T1−MAG-B MP3D Final +0.131 [0.044, 0.218] |
| P5.3.3 | G3 参照组：B0 / B2 / E2-D100（Task 3、E2）+ **MAGICIAN MAG-G / MAG-B（Task 2 共享协议闭环）** 纳入同一表（AUC、Final、Cov@k、逐场景）；`analyze_g3.py` 读取 Task 2 的 MAG 闭环（tag 前缀不同） | ✅ 表 1 / 表 2（`reports/V3_COMPARISON_TABLES.md`）含 MAG-G / MAG-B 全部指标与配对差 |
| P5.4 | 任务报告 `T{0,1}_G12.md`、`T{0,1}_G3.md`；`reports/V3_COMBINED_REPORT.md`、`V3_COMPARISON_TABLES.md`、`V3_RESOURCES.md` | ✅ 04:30 全部写完；最终决定：问题 1 No-Go（T0 闭环相对 B2 退化）、问题 2 Conditional Go（XAttn 小而一致的增量，MP3D 有意义）；主线不变 |
| P5.5 | git 提交 `[V3][task0]`、`[V3][task1]`、`[V3][eval]` 并 push | ✅ `5c1b8ed` [task0]、`aa1cf2f` [task1]、[eval] 见最新提交 |

## P6 Ablation Study（Experiment A GT-Depth Oracle Decomposition / Experiment B Depth Uncertainty Planner）— ✅（2026-10-31 10:45）
详细编号板：`ablation_study/STATUS.md`。协议 `ablation_study/protocol.md` v1.0（09:55 冻结）。不训练、不启动 V4。
| 编号 | 内容 | 状态 |
|---|---|---|
| P6.0 | 资源 / 缓存审计（`experiment_A_gtdepth_oracle/report/CACHE_REUSE_AUDIT.md`）+ 协议冻结 | ✅ 09:55 |
| P6.A | Experiment A：Pool FX（365 状态）+ Pool CL（1,461 状态）上 B2 / E2D / T0 / T1 的 K=4 预测 + GTD + oracle 拆分 | ✅ 10:35（4.8 GPU-h）：MP3D 一步 regret 全模型 0.11–0.12 vs GTD 0.064（57% 评分 / 区域 / 候选）；停滞 = 候选空间限制（G\* 中位 0.019 pp）；Macarons++ B2 0.468 → T1 0.319（GTD 0.160），预测主导但未兑现到闭环 |
| P6.B | Experiment B：DepthU 评分器 P0 / P1 / P2 固定状态 G2 → Gate → G3 | ✅ 10:40 阴性：P1 恶化 65–200%，P2（λ=1）中性或更差，Gate 8/8 FAIL → 不进 G3 |
| P6.R | `reports/V3_ABLATION_SUMMARY.md` 等 + `[V3][ablation]` 提交 | ✅ 10:50 |
