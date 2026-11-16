# Experiment G — 事件记录（只记事实）

- 2026-11-14 21:45 审计开始：V3-T0 训练 = `experiments/exp6_v3/shared/train_arm.sh t0` → `task0_vae/scripts/train.py` → `seen2scene.v3.trainer.main --arm t0`；`configs/train_v3.json`（sha `a076e088…`，与 T0 运行清单一致）、`model_v3.json`；gpu117 GPU 0–3，world 4，git `7de02cc`（dirty），一次跑完无续跑（`train.log` 只有一个 start / `EXIT 0`），选模 = 40,000 步（`final.json`）。`seen2scene/v3/{trainer,losses,data,model,flow_unet,vae,eval_dev,predictor}.py` 自 `7de02cc` 起未改；`futureview/model.py` 之后只改了 V2 的 RayEncoder 拆分与 V4 新类，V3 的 LatentRayEncoder / UNet2D 不受影响。T0 实测 1.35 s/步（中位数；平均 1.41），不是 0.7 s/步。
- 2026-11-14 21:51 第一次 flag-off 检查启动时 shell 中 `$R` 未定义，输出目录变成 `/work_dirs/...`，torchrun 立即退出（无写入，无 GPU 计算）；日志改名 `logs/check_flagoff_flow50_badpath_attempt.log`，以正确路径重启。
- 2026-11-14 21:51 flag-off 检查（`train_g.py --head flow`，gpu114 GPU 0–3，`--override steps=50`）启动。
- 2026-11-14 21:54 冻结 `reports/G0_PROTOCOL.md`（sha256 `4aa44592ce8e3bf5b33243d7ba8273a91f7102746c7169871be5be20d8d0c5be`；21:53 第一次计算 sha 后只把文中的冻结时间从“22:00”改成实际的“21:54”，之后只读）。
- 2026-11-14 21:55 单元测试第 1 次：17/18 通过；`K_copies_identical_in_batch` 的写法把“同一 batch 内的 K 个副本相同”和“与另一 batch 大小的结果相同”混在一起。诊断：同一 batch 内 4 个副本逐位相同（D0 与 flow 都是），只有不同 batch 大小之间有 bf16 级差异（flow 也一样）。测试改为只检查同一 batch 内的副本，batch 大小差异作为信息项记录；第 2 次 18/18 通过（`work_dirs/exp6_v3/experiment_G/checks/unit_tests.json`）。
- 2026-11-14 22:00 flag-off 检查通过：第 0 步评估全部数值、第 1 步日志全部数值与存档 V3-T0 逐位相同；config / common init / data manifest / 参数量相同（gpu114 vs gpu117）。第 50 步 loss 3.599402 vs 3.599306（不逐位，反向不确定，预期内），批次射线数相同。22:01 在第 50 步评估中途按 PID 停掉（该评估在 T0 中没有对应项）。
- 2026-11-14 22:01 D0 训练 smoke（`--head det --override steps=300`）启动，gpu114 GPU 0–3。
- 2026-11-14 22:02 共享队列（gpu115）：`expG_a1_smoke_g1r_liberty_s0`（flow K = 1 回归，对照存档 V3-T0-K1D2）、`expG_a3_smoke_g0_liberty_s0`（G0）。
- 2026-11-14 22:03 闭环回归 `expG_a1`（G1R = flow 模式 K = 1，liberty s0，10 步）：10/10 步分数逐位相同、选择相同、执行位姿与覆盖前缀与存档 V3-T0-K1D2 相同，官方回放一致 → PASS。`expG_a3`（G0，10 步）PASS（官方回放一致；第 0 步分数与 G1 不同，`G_MODE g0` 已生效）。
- 2026-11-14 22:14 D0 smoke 300 步结束（`checks/d0_smoke300`）：`check_train_vs_t0.py --mode det` PASS —— common init / config / data manifest / 参数量与 T0 相同；model_cfg 只多 `head: det`；第 1 步 rec 1.7421808540821075、kl 0.09142005443572998、beta、lr、射线数、rec_diag、posterior、kl_per_ch 与 T0 第 1 步逐位相同；第 0 步评估的 recon_* 逐位相同；评估字段集合与 T0 相同；loss 2.677（第 1 步）→ 0.746（第 300 步），det-MSE 0.935 → 0.277。第 300 步的 EMA（0.999^300 ≈ 0.74 的初始权重）尚不预测命中（pred_hit_rate 0），预期内。
- 2026-11-14 22:14:57 D0 正式训练启动：`train_d0_chain.sh`（chain pid 36296 → run_train.sh 36312 → torchrun 36323 → ranks 36369–36372），gpu114 GPU 0–3，`work_dirs/exp6_v3/experiment_G/d0/`。
- 2026-11-14 22:16 `expG_a2`（D0 smoke 权重 `ema_000300.pt`，liberty s0，10 步）PASS：K = 1 规划器加载 D0 检查点（`G_MODE d0`），官方回放一致，决策 0.91 s/步（G1 1.39 s/步）。`expG_a4` 固定状态测试（conf_liberty_s0，t = 0 / 5，四行同一进程）PASS：G4 的 D1 分数与 EXP9 Table 4 V2 T0 行逐位相同（2/2），四行候选 / 路径与 F 行相同（8/8）；约 7 s / 状态；`g_analyze_fixed.py` 在该输出上跑通。
- 2026-11-14 22:25 D0 正式训练稳定：第 1 步与 smoke 第 1 步逐位相同（loss 2.6770804866239013；rec / kl 同 T0）；第 300 步 loss 0.750；1.32–1.40 s/步（T0 中位数 1.354）；每卡 32.5 GB 峰值；GPU 利用率 84–90 %。预计（与 T0 的 15 h 54 min 同速）11-15 约 14:10 结束（±30 min）。之后：`scripts/post_train.sh`。

## G2（用户 2026-11-15 批准，五项全做）
- 2026-11-15 16:39 冻结 `reports/G2_PROTOCOL.md`（G2-v1，sha256 `83dd7436f00e4b262a1ffb603a4451ad66267c2669332bbfa8c271b0f8b25fea`），在 G2 的任何计算之前。D0 `final.pt` sha256 `aeefe098…`。
- 2026-11-15 16:40 gpu114 上启动 4 个临时队列 worker（`SLURM_JOB_ID=g114x`，pid 16534 / 16538 / 16542 / 16547）；gpu115 的 4 个 worker（3497835）在线。
- 2026-11-15 16:42 室内冒烟（EU6Fwq7SyZv_L0 S34，10 步）：G4R（包装器 + K = 4）与存档 EXP7 OURS 的 10 步分数 / 选择 / 执行位姿 / 覆盖前缀逐位相同 → PASS；G1IN（flow，K = 1）、D0IN（d0，K = 1）各 10 步 PASS（G_MODE 行、K = 1、d1 配置均确认）。决策 3.4 / 2.0 / 1.8 s/步（G4R / G1IN / D0IN）。
- 2026-11-15 16:43–16:45 功能测试：`g2_prob.py`（S-c-out 2 块）与 `g2_pool.py`（conf_liberty_s0，t ≤ 5；eps 0..3 与回放链逐位相同）通过，输出在 `work_dirs/.../g2/test/`（不计入结果）。
- 2026-11-15 16:46 入队：`expG2_b_prob_{in_val,out_sc}`（训练器环境，用于与 eval.jsonl 逐位回归）、`expG2_c_pool_w1..8`、`expG2_d_in_*` 88 个（按 EXP7 OURS 墙钟从长到短，G1IN / D0IN 交错）。
- 2026-11-15 16:47 第 3 项（CPU）完成：`results/g2/temporal/`、`figures/g2/temporal_diff.{pdf,png}`。
- 偏离（记录）：F-pool 的分层分位边界由分析脚本在存储的标签数组上计算（只用标签，与模型输出无关），而不是“在任何模型前向之前写入文件”；验证集的边界按协议在前向之前写入 `edges.json`。
- 2026-11-15 16:54 `expG2_b_prob_{in_val,out_sc}` 完成：G4（噪声块 0）与 T0 `eval.jsonl` 第 40,000 步 `samp_*`、D0 与 D0 `eval.jsonl` 第 40,000 步 `samp_*` 全部逐位相同（brier / false_wall / pred_hit_rate / label_hit_rate / fair ES，两个集）。
- 2026-11-15 17:09 `expG2_c_pool_w1..8` 完成：54 条轨迹，755 个状态中 739 个至少有一个带标签 UNKNOWN 射线的候选视角（另 16 个状态没有，不进入第 1、2、4 项），4,606 个视角，823 万条带标签射线；eps 0..3 与回放链逐位相同（每个状态都检查）。
- 2026-11-15 17:11 `g2_analyze.py`（全部三个集）与 `g2_figs.py` 完成。`g2_figs.py` 只改了绘制（去掉深度面板上的命中轮廓、matplotlib API），选图规则与数据未改。
- 观察（不改规则）：预先声明的 GT-only 歧义分 a(v)（C-edge ∪ D-edge 射线数）偏向 GT 命中掩膜碎片化的视角（UNKNOWN 采样点吸附造成的 HIT / 非 HIT 交替），前两个被选状态（sestino_museum t10、colosseum t10）的 GT 命中掩膜呈椒盐状，可读性差；如需改用别的规则属于事后改动，须作为偏离由用户决定。
- 2026-11-15 17:06 起室内 88 个 episode 开始执行（8 卡）；K = 1 远快于 EXP7 OURS（最长的层 6 分钟走到约 100 步）。
- 2026-11-15 17:16 前 4 个完成的室内 episode（TbHJrupSAjP_L1 S112 / S114、x8F5xyUWy9e_L0 S28）用 `g2_indoor_collect.py` 试验证：V1–V5、同起点、G_MODE / K = 1 全部通过（试运行输出在 `work_dirs/.../g2/test/indoor_partial/`，不计入结果）。每条约 8–9 分钟（EXP7 OURS 同层记录的墙钟 2.9 h 是 K = 4 且当时 GPU 共享）。
- 2026-11-15 17:22 启动分离的守护脚本 `scripts/g2_stop_g114x_when_drained.sh`（pid 51026）：队列中不再有待执行的 `expG2_*` 时 touch `STOP_g114x_gpu0..3`，只做这一件事。此时 8/88 完成，72 个待执行。

## G3（用户 2026-11-15 批准：Ours vs 把 K = 4 个生成图取平均，两种平均，室外 + 室内）
- 2026-11-15 18:58:50 冻结 `reports/G3_PROTOCOL.md`（G3-v1，sha256 `34f5002b25c78868380f835747f66744f153f2abba721d095b54cb5f09da7303`），在 G3 的任何计算之前。
- 2026-11-15 18:59 实现：`scripts/m_combine.py`（噪声钩子 + score_state 钩子，装在 D1 之前；raw / depth / off）、`g3_out_planner.py`、`sim_out_g3.py`、`g3_episode.sh`、`g3_in_planner.py`、`sim_in_g3.py`、`g3_in_episode.sh`、`g3_smoke_check.py`、`g3_queue.py`、`g3_fixed.py`、`g3_analyze_fixed.py`、`g3_collect.py`、`g3_figs.py`、`test_m_combine.py`。冻结文件未改。
- 2026-11-15 18:59 单元测试 8/8 通过（≥ 2/4 规则、2-2 平局算命中、只对命中样本取平均、单个命中回填 t_known、M-raw 的均值与解码规则；`work_dirs/.../g3/checks/unit_m_combine.json`）。
- 2026-11-15 19:00 gpu114 重新启动 4 个临时 worker（g114x，pid 23562 / 23564 / 23568 / 23573）。
- 2026-11-15 19:01 冒烟 6/6 通过：室外 G4R（钩子关闭、K = 4）liberty s0 10 步，与存档 V3-T0-D1 的分数 / 选择 / 执行位姿 / 覆盖前缀逐位相同；室内 G4RIN（EU6Fwq7SyZv_L0 S34）与 EXP7 OURS 逐位相同；M-raw / M-depth 室外（官方回放一致）与室内各 10 步正常，G3_MODE 行与 K = 1 已确认。
- 2026-11-15 19:02 固定状态功能测试（conf_liberty_s0，t ≤ 5）通过：G4 的 D1 分数与 Table 4 T0 行逐位相同；M 行内部的 4 个样本与 G4 的样本逐位相同；G4 / G1 / D0 的 D1 分数与 Experiment G 固定状态行逐位相同。
- 2026-11-15 19:01:52 入队 228 个闭环任务（室外 `expG3_c_*` 140 个，室内 `expG3_d_in_*` 88 个）；19:02:45 入队 8 个固定状态 worker（`expG3_b_fixed_w1..8`）。
- 2026-11-15 19:45 固定状态完成（`expG3_b_fixed_w1..8`）：755/755 个状态五行齐全；G4 的 D1 分数与 Table 4 T0 行逐位相同 755/755；M 行内部的 4 个样本与 G4 的样本逐位相同 1510/1510。`g3_analyze_fixed.py` → `results/g3/fixed_table.*`；`g3_figs.py` → `figures/g3/averaging_*`（选图规则冻结，选中 pantheon t0、bannerman t0、barts t5、colosseum t5；深度色标取 GT 的 2–98 百分位，超出范围的预测被截到两端）。
- 2026-11-15 19:45 闭环：室外完成 7 个、待执行 213 个、0 失败（19:02–19:45 有 8 卡被固定状态任务占用）；已完成的 dunnottar episode 官方回放一致。预计 11-16 约 00:00–00:30 全部完成。
- 2026-11-15 19:07 启动分离守护 `scripts/g3_stop_g114x_when_drained.sh`（pid 33706）：没有待执行的 `expG3_*` 任务时 touch `STOP_g114x_gpu0..3`。
