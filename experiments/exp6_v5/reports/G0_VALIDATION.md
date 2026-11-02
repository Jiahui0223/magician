# G0 验证报告 —— V5 与 V4-Continue12k（2026-11-01 19:00–19:42）

全部 11 门通过。证据：单测输出（`experiments/exp6_v5/tests`，6 个文件 21 项，CPU）、冒烟运行的 `run_*.json` / `log.jsonl`（`work_dirs/exp6_v5/runs/g0/`）、回放比对 `work_dirs/exp6_v5/eval_fixed/g0_replay_check.json`、日志 `work_dirs/exp6_v5/logs/g0_*.log`。
冒烟在 V5 代码首次提交前运行（manifest 记录的 HEAD 为 `cb74c40`，V5 文件当时未跟踪）；冒烟后的修复全部包含在 `ad8dc5c`，两组正式训练都从 `ad8dc5c` 启动。
节点：gpu115（作业 3443137）、gpu116（作业 3445003），L40S。

| 门 | 检查 | 结果 | 证据 |
|---|---|---|---|
| G0-1 V4 回放 | V5 评估框架（`eval_g1_g2.py`，`v5_core.install()` 已生效）以 V4 final.pt 为 V4C 组，跑 VALSEL 第 1 个 episode 与 Dev-3 B0 第 1 个 episode | ✅ 20/20 状态的逐候选 DistP1 分数与 exp6_v4 的 V4 行**逐位相同**，选点 20/20 相同，B2 回放 20/20 逐位相同 | `g0_replay_check.json` |
| G0-2 无泄漏 | 训练只读 `train6_minus_Sc` 分片与 Train-6 标签（标签目录只含 6 个 Train-6 场景）；MP3D 训练样本 0；评估端 belief GT 只在评估后读 Dev-3 / MP3D 网格 | ✅ | `test_g0_invariants::test_training_data_isolation`；`AUDIT_V5.md` §3 |
| G0-3 几何 | V5 检索格约定 = V4 grid_sample（align_corners = False）；检索到的格中心到射线距离 ≤ 半格对角线、τ 与解析值一致；标签射线 = 训练射线（同一 `RY.model_rays_torch`） | ✅ 3/3 | `test_ray_world_geometry` |
| G0-4 检索 | 网格外射线无局部 token；域外格被 mask、UNKNOWN 域内格保留；相机后 / 重复格被 mask；无局部 token 时只剩 null + 8 coarse 且输出有限；无 R × N 稠密矩阵（分层 gather） | ✅ 4/4 | `test_retrieval_masking` |
| G0-5 无 32 点推理 | V5Model 无 `rays` / `ray_xattn`；V5 前向 grid_sample 调用 0 次；规划器加载时 `v5_core` 对预测器前向装 grid_sample 抛错保护，2 步闭环冒烟正常结束 | ✅ | `test_v5_forward_backward::test_no_grid_sample_and_layout`；`logs/g0_smoke_{in,out}_v5.log`（`V5 Core … grid_sample guard on`） |
| G0-6 多表面 GT | 合成 3 / 9 墙（+ 重复面、近裁剪前墙，2 种 jitter）；trimesh 精确多交点 2,000 射线；首层 = t*；训练目标：后方表面为正例、区间外 / 空槽 / 填充射线不监督、首表面后与表面旁负例 | ✅ | `labels/test/test_pisa.json`；`test_multisurface_targets`（5/5）；`reports/LABEL_QUALITY.md` |
| G0-7 belief → Flow | cond 布局 = V4（11 投影 + 64 射线 + 有效列）；只改表面头的偏置 → cond 的 64 射线通道改变（11 投影通道不变）；分块 / 子集与全量逐射线一致；所有参数（含 encoder / unet / dec 全部子模块）收到有限非零梯度 | ✅ 4/4 | `test_v5_forward_backward` |
| G0-8 warm-start | 382 个 encoder / unet 张量与 V4 EMA 逐位相同（24,745,890 参数）；V4 文件 sha / mtime 加载前后不变；不使用的 V4 模块 = {rays, ray_xattn}；Teacher 的 `_ray_context_v4` = V4 cond 的 64 射线通道 | ✅ 2/2 | `test_teacher_migration` |
| G0-9a λ 梯度探针 | 300 步 Bridge（λ 暂 1）后 30 步新模块梯度 L2 中位数：FM 0.108、surf 0.148、bridge 1.125 → λ_surf 0.37、λ_bridge0 0.096（规则写入配置与协议） | ✅ | `runs/g0/v5_gradscale/run_gradscale.json` |
| G0-9b overfit（300 步固定 batch） | V5（Bridge 100 → Joint）：2.011 → 0.062（drop 0.969），FM 0.0106；表面头：首表面召回 0.43、第二+ 0.48、负例正确率 0.84、**首表面后负例 0.76**（修复前 0.014）；V4C（lr 1e-5）：0.0064 → 0.0040（drop 0.37） | ✅ | `runs/g0/{v5,v4c}_overfit/run_overfit.json` |
| G0-9c resume | 单卡 A = 40 步直跑，B = 20 步 + 恢复到 40 步（第 20 步 Bridge → Joint 切换处恢复；numpy / torch / CUDA / 查询生成器 RNG 精确恢复） | ✅ model、EMA 最大差 **0.0** | `runs/g0/resume_check.json` |
| G0-9d DDP | 4 rank torchrun 400 步（每 rank 16 视角 / 2 状态 = 正式 batch） | ✅ V4C 0.051 → 0.031；V5 0.496 → 0.269，H_new / H_old 余弦 0.83，FM 0.026 | `runs/g0/{v5,v4c}_ddp_short/run_short.json` |
| G0-10 显存 / 速度 | 单卡 16 视角：V5 0.66 s/步、峰值 4.5 GB；V4C 0.34 s/步、10.5 GB；4 卡 DDP（与标签进程共享 GPU）V5 0.83 s/步、4.6 GB/卡 | ✅ | `runs/smoke/*`、`runs/g0/*` |
| G0-11 规划器不变 | 冻结配置 K = 4、MP3D n_win = 5；V5 子进程 = V4 子进程 + `v5_core.install()`（task2 规划器 main 不变）；P1 默认 b2（D100 输入、S10 稀疏 DistP1）；scoring / v2_plan_core / predictor 不引用 V5 或 belief；V5 smoke checkpoint 2 步闭环：MP3D（Z6MFQCViBuw L0 S6）与 Macarons++（barts s0）均 exit 0 | ✅ | `test_g0_invariants`；`logs/g0_smoke_{in,out}_v5.log` |

## 冒烟中发现并修复的问题（均在正式训练前）
1. **L_bridge 尺度**：第一版按 Teacher 每通道标准差归一化，近常数通道把损失放大到 3.8e5、梯度 9.6e3。改为标量 RMS 归一化（保留原始幅值匹配），并把适配器输出尺度初始化为 0.05（V4 H_old RMS 0.030）；初始 FM 从 1.61 降到 0.13。
2. **表面头退化**：随机对数深度查询很少落在首表面之后，overfit 中首表面后负例正确率只有 0.014（学成"首表面后都是表面"）。加入首表面后的均匀查询与每个表面旁 Δ..2Δ 的难负例，BCE 在正例 / 首表面前负例 / 首表面后负例三组间均衡；修复后 0.76。
3. **NaN**：表面旁难负例在首表面很近时深度为负，log1p 产生 NaN，乘法屏蔽仍传播。查询截断到 ≥ 1e-4，BCE 用 where 屏蔽。
4. **冒烟数据漂移**：冒烟取"已有标签的前 8 条轨迹"，标签并行生成使 A / B 两次 resume 运行的数据不同（第 1 步损失即不同）。改为配置中固定的冒烟轨迹列表；重跑后 resume 差为 0。
5. **belief 导出形状**：某视角 UNKNOWN 射线不足 128 条导致数组不能堆叠；改为每视角固定 256 条不同射线（UNKNOWN 优先，其余随机补齐）。
