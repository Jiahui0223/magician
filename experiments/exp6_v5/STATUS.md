# EXP6 V5 — 状态板（每次启动 / 结束一个运行就更新）

**最近更新：2026-11-01 20:30（两组正式训练运行中；V5 已过 Bridge 2k 进入 Joint；评估端 belief 全链路冒烟通过）**
标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 未开始

## 0. 此刻在跑什么
| 编号 | 内容 | 节点 / GPU | 作业 | 日志 | 开始 | 状态 |
|---|---|---|---|---|---|---|
| P3-V4C | **V4-Continue12k 正式训练**（12k 步，64 视角 / 8 状态 / 步，lr 1e-5，固定 EMA@12000；`train_arm.sh v4c`，torchrun 4 卡） | gpu115 GPU 0–3 | 3443137 | `work_dirs/exp6_v5/logs/train_v4c.log`、`runs/v4c/{log,eval}.jsonl` | 19:36 | 🔄 步 0 S-c FM 0.0196（= V4 EMA@40k）；0.63 s/步；ETA ≈ 21:50 |
| P3-V5 | **V5-Warm12k 正式训练**（Bridge 2k + Joint 10k，λ_surf 0.37、λ_bridge0 0.096，固定 EMA@12000；`train_arm.sh v5`） | gpu116 GPU 0–3 | 3445003 | `work_dirs/exp6_v5/logs/train_v5.log`、`runs/v5/{log,eval}.jsonl` | 19:42 | 🔄 步 0 S-c FM 0.044（新 decoder 未训练）；0.64 s/步；ETA ≈ 22:30 |

| P4-V4C 链 | `post_arm_v5.sh v4c`：等 EXIT 0 → `select_final.py v4c`（EMA@12000 → `checkpoints/v4c_final.pt`）→ V4C 的 G3（4 worker）→ 覆盖率 → G1 / G2（365 态） | gpu115 | 3443137 | `logs/post_v4c.log` | 19:52 | ⏳ 等训练 |
| P4-V5 链 | `post_arm_v5.sh v5`：同上 + 评估端 belief GT（`eval_belief_gt.py`）+ belief 指标 | gpu116 | 3445003 | `logs/post_v5.log` | 19:52 | ⏳ 等训练 |
| P4-助手 | `post_helper_v5.sh`：V4C 链结束且 v5_final.pt 存在后，gpu115 加入 V5 的 G3 队列（claim 共享） | gpu115 | 3443137 | `logs/post_helper_v5.log` | 19:52 | ⏳ |

**GPU / 作业**
| 作业 | 节点 | 到期 | 用途 |
|---|---|---|---|
| 3443137 `vscode-dev` | gpu115 4 × L40S | 11-04 15:11 | 标签、G0、V4-Continue 正式训练（计划）、G3 |
| 3445003 `exp6-v5-aux` | gpu116 4 × L40S | 11-04 19:14 | 标签（反向）、G0、V5 正式训练（计划）、G3 |

## P0 审计 — ✅
| 编号 | 内容 | 状态 |
|---|---|---|
| P0.1 | 读两份附件（设计 v1.0、实现任务书） | ✅ |
| P0.2 | V4 checkpoint / 形状 / 参数 / H_old 尺度（`scripts/audit_v4_shapes.py` → `reports/audit_shapes.json`） | ✅ sha `1b31afff…`，F0 [1,32,X',Y',32]，F1 [1,64,X'/2,Y'/2,16]，H_old RMS 0.030 |
| P0.3 | 数据 / 坐标 / 规划器 / 资源审计 | ✅ `AUDIT_V5.md` |

## P1 标签 — ✅
| 编号 | 内容 | 状态 |
|---|---|---|
| P1.1 | 合成双墙测试（2 种 jitter） | ✅ 7,296/7,296 射线 2 表面，误差 ≤ 2.4e-6 |
| P1.2 | trimesh 精确多交点对比（pisa，2,000 射线） | ✅ 一致率 1.000 |
| P1.3 | 全量缓存（180 轨迹） | ✅ 19:42：180/180，129,914 视角；≥ 2 表面射线 17.1 %（有表面射线的 55.8 %）；首层 = t* 全部一致 |
| P1.4 | 标签质量报告 + 3 张图（`scripts/label_quality.py`） | ✅ `reports/LABEL_QUALITY.md` |

## P2 代码 — ✅
| 编号 | 内容 | 状态 |
|---|---|---|
| P2.1 | `seen2scene/futureview/v5_model.py`（V5Model、V5RayDecoder、Memory、retrieve、warm-start） | ✅ |
| P2.2 | `scripts/train_v5.py`（v5 / v4c 两组，Bridge / Joint，L_surf / L_bridge，DDP，resume，smoke 模式） | ✅ |
| P2.3 | 规划器接入：`v5_core.py`、`v5_{in,out}_planner.py`、`sim_{in,out}_v5.py`、`run_queue_v5.sh` | ✅ |
| P2.4 | 评估：`eval_g1_g2.py`（含 belief 导出）、`eval_belief_gt.py`、`eval_belief_metrics.py`、`eval_g3.py` | ✅（待权重） |
| P2.5 | 单测 6 个文件 21 项 | ✅ 全部通过（CPU） |

## G0 — ✅
| 门 | 内容 | 状态 |
|---|---|---|
| G0-1 | V4 回放：V5 评估框架（v5_core 已装）跑 V4 final.pt，VALSEL 10 态 + Dev-3 10 态 | ✅ 20/20 分数逐位相同，选点 20/20，B2 回放 20/20 |
| G0-2 | 无泄漏：训练只读 Train-6 分片 / 标签；评估 GT 只在评估后 | ✅ `test_g0_invariants` |
| G0-3 | 几何：格约定 = V4 grid_sample；标签射线 = 训练射线 | ✅ `test_ray_world_geometry` |
| G0-4 | 检索 mask（网格外 / 域外 / 相机后 / 重复；UNKNOWN 保留；无局部 token 时只剩 null + coarse） | ✅ `test_retrieval_masking` |
| G0-5 | 推理无 32 点采样：前向 grid_sample 0 次；规划器加保护 | ✅ 单测 + 规划器冒烟（`V5 Core … grid_sample guard on`） |
| G0-6 | 多表面 GT（合成 3 / 9 墙、trimesh、首层 = t*） | ✅ |
| G0-7 | belief → Flow：只改表面头 → cond 的 64 射线通道改变 | ✅ `test_belief_reaches_flow_condition` |
| G0-8 | warm-start：382 张量逐位复制，父文件 sha / mtime 不变；Teacher H_old = V4 cond 通道 | ✅ `test_teacher_migration` |
| G0-9a | 梯度探针 → λ 冻结 | ✅ FM 0.108 / surf 0.148 / bridge 1.125 → λ_surf 0.37、λ_bridge0 0.096 |
| G0-9b | overfit | ✅ V5 2.011 → 0.062（drop 0.969），表面头无退化（首表面后负例正确率 0.76）；V4C 0.0064 → 0.0040（lr 1e-5） |
| G0-9c | resume（单卡，A 40 vs B 20 + 20，跨 Bridge → Joint） | ✅ model / EMA 最大差 0.0 |
| G0-9d | 4 卡 DDP short（400 步） | ✅ V4C 0.051 → 0.031；V5 0.496 → 0.269（H_new / H_old 余弦 0.83，FM 0.026），4.6 GB/卡，0.83 s/步（与标签共享 GPU） |
| G0-10 | 显存 / 速度 | ✅ V5 0.66 s/步、4.5 GB（单卡 16 视角）；V4C 0.34 s/步、10.5 GB |
| G0-12 | 评估端 belief 全链路冒烟（V5 smoke ckpt：VALSEL 1 ep + Dev-3 1 ep → 导出 256 射线 / 视角 → 网格多表面 GT（MP3D、Dev-3 `dev_eval`）→ 指标） | ✅ 20:26 全部 exit 0；GT 中 ≥ 2 表面射线 0.25 / 0.23；回放检查 PASS（共享 GPU 时每 ep 7–28 min，无竞争时 V4 为 1–3 min） |
| G0-11 | 规划器不变：K 4、n_win 5、b2、同一规划器代码；2 步闭环冒烟（V5 smoke ckpt）室内 / 室外 | ✅ 两域 exit 0 |
| 失败记录 | ① 第一版 L_bridge 按通道标准化 → 近常数通道放大到 3.8e5；改为标量 RMS + 适配器输出尺度 0.05 ② 表面头退化（首表面后负例正确率 1.4 %）→ 加首表面后 / 表面旁负例与三组均衡 ③ 表面旁负例深度可为负 → NaN；查询截断到正数、where 屏蔽 ④ 冒烟轨迹列表随标签生成变化 → 固定列表 | ✅ 均已修复并重测 |

## P3 正式训练 — 🔄
| 编号 | 内容 | 状态 |
|---|---|---|
| P3-V5 | V5-Warm12k（Bridge 2k + Joint 10k），固定 EMA@12000 | 🔄 19:42 启动（gpu116） |
| P3-V4C | V4-Continue12k，固定 EMA@12000 | 🔄 19:36 启动（gpu115） |

## P4 评估 — ⬜
| 编号 | 内容 | 状态 |
|---|---|---|
| P4.1 | G3 Macarons++ 30 × 100、MP3D 10 × 150（V5、V4C） | ⬜ |
| P4.2 | G1 / G2（365 态）+ belief GT / 指标 | ⬜ |
| P4.3 | 统一表、资源、偏差、FINAL_REPORT | ⬜ |
