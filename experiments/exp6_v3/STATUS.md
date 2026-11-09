# EXP6 V3 — 全局状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-11-09 09:25（P13 XB：Task A / B / C / D 全部完成（C：数据减半 → 验证 +6.7% 变差、闭环不变）；B2 的 6 个 colosseum / sestino 补跑中（~09:45），之后写 FINAL_REPORT；gpu115 分配 15:12 到期，GPU 3 空闲；P14 ✅ V3-T0-D1 = T1 + D1）**
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

## P7 Ablation Experiment C — GT-Depth Closed-Loop（2026-10-31 15:20 起；详细板 `ablation_study/STATUS.md` C 段）
| 编号 | 内容 | 状态 |
|---|---|---|
| P7.0 | 协议 + 代码 + G0 9 / 9 PASS（`experiment_C_gtdepth_closed_loop/report/C_G0_VALIDATION.md`）| ✅ 15:44 |
| P7.1 | C1（GT depth + DistP1）40 条闭环 → 阶段回复 | ✅ 16:10 / 16:25（MP3D AUC 0.357 vs B2 0.334、Final 0.513 vs 0.530；Macarons++ 0.741 / 0.894 vs 0.730 / 0.906；Cov@20 +0.058 显著）|
| P7.2 | C2（GT gain greedy）40 条闭环 + 室外覆盖率 | ✅ 16:27（MP3D AUC 0.400 / Final 0.615；Macarons++ 0.771 / 0.919；C2 − C1 有意义）|
| P7.3 | 分析 + 报告（`experiment_C_gtdepth_closed_loop/report/C_FINAL_REPORT.md`、`reports/V3_ABLATION_*` 更新）+ `[V3][ablation-c]` 提交；≈ 6.3 GPU-h | ✅ 16:50 |

## P8 Task 2 — Coverage-Aware NBV Planning（2026-10-31 17:00–19:15；详细板 `task2/STATUS.md`）
| 编号 | 内容 | 状态 |
|---|---|---|
| P8.1 | E1 评分器（S0–S4）固定状态 1,676 状态（4.9 GPU-h）+ Gate：室外 S1c 两模型通过；室内仅 T1 通过（S2h / S3 / S2S3） | ✅ 17:55 |
| P8.2 | G3：室外正式 B2 + S1c / T1 + S1c 无改善（退化 / 无差异）；MP3D 探索性 T1 + S2S3 AUC 0.354 / Final 0.530（vs T1 0.331 / 0.508；= C1）；R2 frontier bonus / trigger 严重退化（9.6 GPU-h） | ✅ 19:05 |
| P8.3 | E2 离线 R1（区域 n_win 9 → GT Top-1 0.894）/ R3（两步视野 2.3×）/ 失败案例；总报告 `task2/reports/TASK2_COMBINED_REPORT.md`；提交 | ✅ 19:15 |

## P9 Multi-View Coverage Reasoning Study（2026-11-05；`multiview_coverage_study/FINAL_REPORT.md`）
| 编号 | 内容 | 状态 |
|---|---|---|
| P9.0 | 口径冻结 `configs/mv_v1.yaml`；冒烟 colosseum s1（修正位姿键映射后，抓取集合重算分数 = 规划器） | ✅ |
| P9.1 | 395 个固定状态（E3 120 + G2 275），8 GPU，01:03–01:34，0 失败 | ✅ |
| P9.2 | 价值函数诊断（两轮）、案例重跑 + 真实渲染、汇总与图 | ✅ 02:20 |
| P9.3 | 结论：A/B/C 都不是选错主因；CVAR 门槛 FAIL；同一价值函数用真实深度 Top-1 0.44 < 预测 0.55，覆盖 0.80–0.85 → 价值函数是瓶颈；不跑闭环 | ✅ |

## P10 Experiment F — Stochastic Future Views & Coverage-Aligned Decision Study（2026-11-05；`ablation_study/experiment_F_stochastic_coverage_decision/FINAL_REPORT.md`）
| 编号 | 内容 | 状态 |
|---|---|---|
| F-cfg | 协议入库（`EXPERIMENT_PLAN.md`）；`configs/F_config.yaml` + 确认层 `confirmation_set.json`（6 个新场景 × 起点 0/2/4，D-v2 Table 1 状态，360 个）在任何计算前冻结 | ✅ 03:05 |
| F-orc | 确认层官方真值（E 的冻结 e_oracle），gpu115 4 卡，18/18 | ✅ 03:19 |
| F-cache | 755 个状态缓存（K=4/8 逐样本集合、软覆盖变体、GT 深度审计），gpu114+gpu115 8 卡，03:20–04:19，0 失败，约 8 GPU-h | ✅ |
| F0 | 755/755 逐位复现（确认层 = D-v2 闭环日志）PASS；GT 深度归因更正：原始 GT 深度含网格外命中（下标截断），可表达完美深度一步 regret 0.06–0.10 pp ≪ 预测 0.16–0.22 → 预测质量是主瓶颈 | ✅ |
| F1–F3 | 探索层冻结 γ*=0.25 / σ*=0.25·τ 深度权重 / 自适应 4→8（`results/selection.json`）；确认层：F1 TREND −0.007 pp，F2 TREND −0.007 pp，F3 FAIL +0.017 pp | ✅ 04:20 |
| F4 | Soft×K 交互 +0.000 [−0.026, +0.022]；无可采纳组合 | ✅ |
| F5 | 未进入（无 PASS）；无闭环、无 AUC | ⛔ 按协议不执行 |

## P11 Experiment CT — V3 Continued Training & Cross-Domain Generalization（2026-11-05 起；`ablation_study/experiment_CT_continued_training/`）
| 编号 | 内容 | 状态 |
|---|---|---|
| CT-cfg | `EXPERIMENT_PLAN.md`、`configs/CT_config.yaml`（CT-v1）、`configs/train_ct.json` 在任何 CT 评估或训练之前冻结 | ✅ 18:15 |
| CT0 | 谱系审计 `results/CT0_checkpoint_audit/CT0.json`：Y50K 是真实续训（恢复 model / EMA / AdamW，数据 / 损失相同），只有 LR 计划不同 | ✅ |
| CT1-fixed-g1 | 固定状态（Indoor 90 + Outdoor 755）A_40K + B_Y50K；8 卡；40K 回归 845/845 逐位；Y50K 深度误差显著改善，regret 未改善 | ✅ 18:55 |
| CT3-cf | 40K 决策反事实（去 FP / 去 FN / 深度 oracle），gpu115 g0-3，作业 3465477，日志 `logs/fixed/cf_A_40K_*` | ⏸ 19:34 停止：63 条轨迹完成 26 条（室内 9 / 9，室外 17 / 54），可续跑 |
| CT4-regress | 40K 闭环回归（barts s0 + Z6MFQCViBuw_L0 S6），gpu115 | ✅ 两条的执行位姿和覆盖都与存档一致 |
| CT4-Y50K | Y50K 小规模闭环（Dev-5 + Indoor），gpu115 | ❌ 19:08 起，与同卡的其他任务抢显存，5 次规划器失败（OOM）；19:20 由我停止，**未完成** |
| CT1-C | CT-M 续训 40K → 60K：gpu114 g0-3，作业 3461041，`runs/ct_m/train.log` | ⏸ 19:44:54 停止；可恢复检查点 step 42000 已存档并校验（sha256 `5b82500b…`，`CT_PAUSED_MANIFEST.md`）；没有 45K / 50K 的 EMA |
| CT3 | 预测误差分解（40K vs Y50K 固定状态部分） | ✅ 部分（`results/CT3_prediction_diagnostics/`） |
| CT4 | 小规模闭环（CT50K / CT60K）、All-14 | ⬜ 未执行（CT 暂停） |
| CT1-D / CT2 | 80K / 域平衡 | ⬜ 未执行 |

## P12 Experiment XA — Multi-Scale Geometry-Aware Ray–Scene Attention（2026-11-05 起；`ablation_study/experiment_XA_multiscale_ray_scene/`）
任务书 `XA_Experiment_Protocol.md`；冻结协议 `FROZEN_PROTOCOL.md`（XA-v1）。六个变体各从零训练 40,000 步 → 室外 D1 70 条 × 100 步 + 室内 H0 6 条 × 150 步 → 聚合；不按指标取消。
两条链在现有分配内无人值守运行（QOS 的 8 卡上限已被这两个分配占满，无法另开 sbatch）：链日志 `logs/chain_gpu114.log`、`logs/chain_gpu115.log`。

| 编号 | 内容 | 节点 / GPU / 作业 | 日志 | 状态 |
|---|---|---|---|---|
| XA-eng | 模型代码 `seen2scene/v3/xa/`、六组配置、工程检查 6/6、过拟合冒烟 6/6、大场景显存（M 系列比 T1 多约 1.2 GB）、短闭环冒烟 12/12、T1 回归（室内外逐位一致） | gpu115 | `results/engineering_checks.json`、`results/regression.json`、`logs/EVENTS.md` | ✅ 19:34（提交 `7ae012e`） |
| XA-S2D | 训练 → 选模 → 闭环 → 聚合 | gpu114 g0-3，3461041 | `work_dirs/.../runs/xa_s2d/train.log` | ✅ **COMPLETE** 11-06 15:10：训练 16 h 08 min（step 40000，sha256 `939b7d79…`）；闭环 70 / 70 + 6 / 6，81 次尝试中 5 次失败后重试成功 |
| XA-SI | 同上（链内第 2 个） | gpu114 | `runs/xa_si/train.log` | ✅ **COMPLETE** 11-07 09:40：训练 16 h 19 min（step 40000，sha256 `af5ac851…`）；闭环 70 / 70 + 6 / 6，76 次尝试 0 次失败（2 h 10 min） |
| XA-MR | 同上（链内第 3 个） | gpu114 | `runs/xa_mr/train.log` | ✅ **COMPLETE** 11-08 04:34：训练 16 h 35 min（step 40000，sha256 `d758d4df…`）；闭环 70 / 70 + 6 / 6，76 次尝试 0 次失败（2 h 18 min）；gpu114 链 04:34 结束 |
| XA-SR | 训练 → 选模 → 闭环 → 聚合 | gpu115 g0-3，3465477 | `runs/xa_sr/train.log` | ✅ **COMPLETE** 11-06 15:19：训练 16 h 11 min（step 40000，sha256 `60069dc4…`）；闭环 70 / 70 + 6 / 6，93 次尝试中 17 次失败后重试成功 |
| XA-M2D | 同上（链内第 2 个） | gpu115 | `runs/xa_m2d/train.log` | ✅ **COMPLETE** 11-07 10:12：训练 16 h 33 min（step 40000，sha256 `e5d0fb54…`，峰值 36.6 GB）；闭环 70 / 70 + 6 / 6，76 次尝试 0 次失败（2 h 19 min） |
| XA-MI | 同上（链内第 3 个） | gpu115 | `runs/xa_mi/train.log` | ✅ **COMPLETE** 11-08 05:19：训练 16 h 45 min（step 40000，sha256 `71825e2f…`）；闭环 70 / 70 + 6 / 6，76 次尝试 0 次失败（2 h 21 min）；gpu115 链 05:19 结束 |
| XA-report | `reports/MASTER_TABLE.csv`、`PER_SCENE.csv`、`EARLY_CURVES.png`、`FINAL_REPORT.md` | CPU | — | ✅ 11-08 05:30 定稿：六个变体都与 T1 + D1 持平（平均差 −0.004 / −0.001 / −0.003 / −0.008），两个设计因子都无可检出效应；XA-MR 预测明显更好但覆盖不变；454 GPU-h |

**XA 结果记录**（每出一个结果就补一行；闭环数字以 `reports/FINAL_REPORT.md` 为准）
| 时间 | 变体 | 事件 | 记录 |
|---|---|---|---|
| 11-05 21:40 | XA-S2D、XA-SR | 5,000 步验证（辅助指标，不用于取舍） | ES 室内 / 室外：S2D 9.52 / 8.92，SR 9.57 / 8.90，T1 同步数 9.58 / 8.95；差别在 1% 以内，没有可读的差异 |
| 11-05 23:50 | XA-S2D、XA-SR | 10,000 步验证（辅助指标） | ES 室内 / 室外：S2D 8.15 / 7.75，SR 8.12 / 7.85，T1 同步数 8.14 / 7.85，T0 8.11 / 7.86；仍在 1.3% 以内 |
| 11-05 23:56 | 全部 | 重新估时 | 训练实测 16.0 h / 变体（每 5,000 步含验证 2.00 h，无重启）；闭环按 T1 + D1 存档耗时估 1.2–2 h（原估 2.7 h 偏保守）；全部完成预计 11-08 凌晨 1–3 点（原估 05:00） |
| 11-06 11:45 | XA-SR | 训练完成 + 选模 | 16 h 11 min，1.37 s / 步，峰值 33.4 GB，0 次重启；选中 step 40000（分数 1.388；T1 1.395、T0 1.380）；40K 验证 ES 室内 / 室外 6.37 / 6.43（T1 6.43 / 6.48，T0 6.37 / 6.44）——差别约 1% 以内，读不出差异 |
| 11-06 11:53 | XA-S2D | 训练完成 + 选模 | 16 h 08 min，1.37 s / 步，峰值 33.3 GB，0 次重启；选中 step 40000（分数 1.393）；40K 验证 ES 6.43 / 6.40——同样读不出差异 |
| 11-06 12:12–15:01 | XA-SR、XA-S2D | ❌ 闭环工程故障 → 已修复 | 每卡 4 个 worker 的并发（我的设置）撞上线程上限 4,096 并使大场景显存不足；每个节点各有 1 条 episode 挂死，两条链从 13:49 / 14:02 空等到 15:01（约 1 h 05 min × 8 卡）。15:01 结束挂死进程后重试全部成功；队列脚本改为每卡最多 3 个 + 显存预留 + 90 分钟超时，监视器覆盖闭环阶段。结果不受影响 |
| 11-06 15:10 | XA-S2D | **闭环完成，COMPLETE** | All-14：AUC@20 0.418 / Cov@20 0.648 / AUC@100 0.758 / Final 0.897；对 T1 + D1：−0.008 [−0.016, −0.001] / −0.006 [−0.021, +0.009] / −0.007 [−0.015, −0.000] / −0.013 [−0.031, +0.001]。室内 6 条 AUC@150 0.350（T1 0.329） |
| 11-06 15:19 | XA-SR | **闭环完成，COMPLETE** | All-14：0.423 / 0.660 / 0.765 / 0.906；对 T1 + D1：−0.003 [−0.012, +0.006] / +0.006 [−0.009, +0.020] / +0.000 [−0.005, +0.005] / −0.004 [−0.015, +0.004]——持平。室内 6 条 AUC@150 0.338。SR − S2D：AUC@20 +0.005 [−0.004, +0.013]，AUC@100 +0.007 [−0.002, +0.017]（趋势，含 0） |
| 11-06 20:24 | XA-SI、XA-M2D | 进度检查（用户询问） | 32% / 30%，无重启；M2D 比 S 系列慢约 1%、显存多 3 GB。10,000 步验证 ES 室内 / 室外：SI 8.10 / 7.73，M2D 8.10 / 7.94，T1 8.14 / 7.85——仍在约 1% 以内 |
| 11-07 09:40 | XA-SI | **闭环完成，COMPLETE** | All-14：AUC@20 0.423 / Cov@20 0.653 / AUC@100 0.764 / Final 0.905；对 T1 + D1：−0.002 [−0.008, +0.004] / −0.001 [−0.014, +0.010] / −0.001 [−0.007, +0.004] / −0.005 [−0.016, +0.004]——持平。室内 6 条 AUC@150 0.340。闭环 0 次失败 |
| 11-07 10:12 | XA-M2D | **闭环完成，COMPLETE** | All-14：0.424 / 0.648 / 0.759 / 0.903；对 T1 + D1：−0.001 [−0.010, +0.006] / −0.006 [−0.024, +0.010] / −0.006 [−0.015, +0.002] / −0.007 [−0.022, +0.004]——持平。室内 AUC@150 0.322。M2D − S2D（记忆因子）：AUC@20 +0.006 [−0.004, +0.017]，AUC@100 +0.001；SI − SR ≈ 0 |
| 11-07 15:27 | XA-MR、XA-MI | 进度检查 | 35% / 31%；10,000 步验证 ES 室内 / 室外 MR 7.81 / 7.74、MI 7.52 / 7.70，比 T1（8.14 / 7.85）低 4–8%，是第一次出现可见的早期差异（辅助指标） |
| 11-07 23:00 | XA-MR、XA-MI | 进度检查（用户询问） | 80% / 76%，无重启。30,000 步验证 ES 室内 / 室外：MR 6.04 / 6.36（T1 6.52 / 6.60，低 7% / 4%），MI 6.30 / 6.60；MR 从 10k 起每个里程碑都领先。辅助指标，闭环待 11-08 |
| 11-08 02:15 | XA-MR | 训练完成 + 选模 | 16 h 35 min，1.40 s / 步，峰值 36.6 GB，0 次重启；选中 step 40000；40K 验证 ES 室内 / 室外 **5.92 / 6.18**（T1 6.43 / 6.48，低 7.8% / 4.7%），室内 Brier 0.071（T1 0.080）——六个变体里唯一在验证指标上明显拉开的 |
| 11-08 02:57 | XA-MI | 训练完成 + 选模 | 16 h 45 min，1.42 s / 步，峰值 36.7 GB，0 次重启；选中 step 40000；40K 验证 ES 6.24 / 6.42（低 2.8% / 1.0%） |
| 11-08 04:34 | XA-MR | **闭环完成，COMPLETE** | All-14：AUC@20 0.418 / Cov@20 0.651 / AUC@100 0.764 / Final 0.904；对 T1 + D1：−0.007 [−0.016, +0.002] / −0.003 [−0.019, +0.012] / −0.001 [−0.011, +0.009] / −0.006 [−0.023, +0.005]——持平。Dev-5 变差（AUC@20 −0.023 [−0.034, −0.009]，colosseum Final −0.106），Test-3 变好（AUC@100 +0.023 [+0.010, +0.031]，3 / 3）。验证指标的明显改善没有转化为覆盖改善。室内 AUC@150 0.343 / Final 0.524 |
| 11-08 05:19 | XA-MI | **闭环完成，COMPLETE** | All-14：AUC@20 0.423 / Cov@20 0.659 / AUC@100 0.765 / Final 0.899；对 T1 + D1：−0.003 [−0.010, +0.005] / +0.004 [−0.006, +0.015] / −0.000 [−0.007, +0.005] / −0.011 [−0.024, −0.000]——持平（Final 区间触 0）。室内 AUC@150 0.304 / Final 0.458，三层一致低于 T1（唯一） |
| 11-08 05:30 | 全部 | **Experiment XA 完成** | 6 / 6 COMPLETE；平均差 AUC@20 −0.004、Cov@20 −0.001、AUC@100 −0.003、Final −0.008；九对单因子比较区间全含 0；`FINAL_REPORT.md` 定稿；两个节点空闲 |

## P13 Experiment XB — Data, Geometry, and Attention Bottleneck Audit（2026-11-08 起；`ablation_study/experiment_XB_bottleneck_audit/`）
任务书 `XB_Experiment_Protocol.md`；冻结协议 `FROZEN_PROTOCOL.md`（XB-v1）。不改网络；分离 Data / Geometry / Attention / Training-variance 四种解释。
| 编号 | 内容 | 节点 / GPU / 作业 | 日志 | 状态 |
|---|---|---|---|---|
| XB-0 | 基线审计 `AUDIT_BASELINE.md`（D100 / S10 / R10、VAE 编码、梯度路径、Query 实现、采样分布、XA 一致性） | CPU | — | ✅ 16:50 |
| XB-C | Task C：V3-T1 Reduced-3（bannerman、fushimi、liberty）从零 40K → 选模 → 室外 D1 70 条 + 室内 6 条 → 聚合 | gpu115 g0-3，3465477 | `work_dirs/.../runs/xb_t1_red3/train.log`、`logs/chain_gpu115.log` | ✅ 训练 07:11（40K，EMA@40k；验证 ES 室内持平、室外 **+6.7% 变差**）；闭环 09:18 完成 76 / 76 0 失败：All-14 AUC@20 0.422 / AUC@100 0.767 / Final 0.908，对 T1 + D1 −0.003 / +0.002 / −0.002（CI 含 0）；保留 / 去掉场景无模式 → **数据减半：预测变差、闭环不变**（`C_DATA_SCALING.md`） |
| XB-A | Task A：注意力贡献审计（T1 / XA-SR / XA-MI × 6–9 种干预 = 21 个核，室内 90 + 室外 Dev 80 + Train 60 个状态） | gpu114 4 卡（17:52 全部完成，18:30 到期） | `A_ATTENTION_AUDIT.md`、`results/A_attention/`、`logs/a/` | ✅ 17 / 17 完成 18:30。T1 注意力惰性（三域所有干预结果相同）；XA 注意力被使用但方向在域间相反（MI 室内 zero_wo regret 0.158 → 0.266，室外 Dev 0.258 → 0.218）；最有用的 K/V = 局部 F4 沿射线样本（MI drop_local Dev FNR 0.257 → 0.436），drop_f16 反而降 FNR |
| XB-B | Task B：B1 观测几何保真（8 条室外轨迹）、B2 反事实上界（3 模型 × 15 条）、B3 误差分层（3 模型 × 17 条） | gpu115 g1 / g2–3 / g0，与训练并行 | `logs/b1 b2 b3/queue.log` | ✅ 01:40：B1 8 / 8、B2 45 / 45（colosseum / sestino × 3 共 6 个待 Task C 后补）、B3 17 / 17；`B_GEOMETRY_AUDIT.md`：存在性保留 100%、丢位置 / 薄结构；no_fn ≈ hit_oracle 拿到几乎全部上界（Dev 一步 regret 0.43 → 0.19），depth_oracle 无收益；完美预测仍剩 0.06–0.31 regret；FNR 由 UNKNOWN 段长度驱动（短段 0.34–0.42 vs 长段 0.08–0.10） |
| XB-D | Task D：不确定性分解（已有 7 个模型 × 70 条曲线） | CPU | `results/D_uncertainty.json` | ✅ 16:50：7 个近似等价模型的 All-14 标准差 0.003 / 0.005 / 0.003 / 0.004；起点间 sd 0.06–0.09（约 20–30 倍）；分辨 0.005 需 3–7 个种子（上界） |
| XB-report | `AUDIT_BASELINE.md`、`A_ATTENTION_AUDIT.md`、`B_GEOMETRY_AUDIT.md`、`C_DATA_SCALING.md`、`FINAL_REPORT.md`、`MASTER_TABLE.csv`、`PER_SCENE.csv`、`EARLY_CURVES.png` | CPU | — | ⬜ |

## P14 CVPR Table 1 补跑：V3-T0 + D1（2026-11-08 18:55 起；`experiments/cvpr_results/table1/`）

| 编号 | 内容 | 资源 | 日志 / 结果 | 状态 |
|---|---|---|---|---|
| T1-T0D1 | 新行 `V3-T0-D1`：V3-T0（task0_vae final.pt，sha 7ed0ddd7…，无交叉注意力）+ 冻结 D1（与 V3-D1-H3-NORISK 同 d_policy / d_config），All-14 × 5 起点 = 70 条，官方覆盖重放核对后写入规范记录 | gpu115 4 卡，每卡 3 个 worker（dunnottar 单独先跑），`scripts/run_t0d1.sh` | `work_dirs/cvpr_results/table1/g3/logs/queue_gpu115_*.log`、`episodes/V3-T0-D1/` | ✅ 18:53 → 20:44，70 / 70，0 失败 0 重试，单条均 16.8 min（12 worker 并行），19.6 GPU-h。All-14：AUC@20 0.420 / Cov@20 0.646 / AUC@50 0.633 / AUC@100 0.763 / Final 0.911；对 T1 + D1 五项差均在 CI 内（AUC@20 −0.005 [−0.013, +0.003]）；对 MAG-B AUC@20 +0.113 [+0.073, +0.156]、Final +0.004 n.s. |
| T1-T0D1-post | validate → aggregate → paper_tables refresh → README 变更日志 → 配对比较 | CPU | `table1/results/`、`paper_tables/`（172 + 441 项检查通过）、`table1/reports/V3_T0_D1_vs_T1_MAG.md`、`exp6_v3/figures/early_phase_vs_magician_t0.*`、`V3_T0_D1_MODEL_ARCHITECTURE.md` §7.2 | ✅ 20:50 |

