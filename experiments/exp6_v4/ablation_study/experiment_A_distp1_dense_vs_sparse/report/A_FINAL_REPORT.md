# Experiment A 最终报告 —— V4 的 DistP1 应该用 D100 还是 S10 评分？（2026-11-01 15:50–17:00）

> 结论：**NO-GO —— D100 评分不设为默认；V4 保留 S10 稀疏 P1 作为 DistP1 的历史表面排除依据。** 室外 Macarons++ 闭环明显退化（ΔAUC −0.043 [−0.097, −0.007]、ΔFinal −0.038 [−0.078, −0.008]，0/6 场景），室内 MP3D 只有小幅、不满足"有意义"标准的改善（ΔAUC +0.018 [−0.005, +0.043]、ΔFinal +0.008 [+0.002, +0.016]）。V2-Mix 时期的 Dense/Dense 退化模式在从零用 D100 训练的 V4 上**依然存在**。

## 1. 问题与答案
| # | 问题 | 答案（证据） |
|---|---|---|
| 1 | V4 用 D100 评分后 MP3D 的 AUC / Final 是否改善？ | **小幅改善但不显著 / 不满足阈值**：AUC 0.327 → 0.345（+0.018，CI 含 0，胜 3/5）、Final 0.495 → 0.503（+0.008 [+0.002, +0.016]，5/0 但 < 0.010）、Cov@20 +0.013；停滞 episode 4 → 2（`A_G3_RESULTS.md`） |
| 2 | Macarons++ 的 AUC / Final 是否改善？ | **否，退化**：AUC 0.710 → 0.666（−0.043 [−0.097, −0.007]，1/5）、Final 0.878 → 0.841（−0.038 [−0.078, −0.008]，0/6）、Cov@20 −0.028；colosseum −0.17 AUC、neuschwanstein −0.04 |
| 3 | Dense Scoring 是否明显增加对 Predicted Hit 的排除比例？ | **是**：被判已知的预测 Hit 比例 MP3D 0.788 → 0.833（+0.045 [+0.036, +0.055]，9/9 楼层）、Macarons++ 0.857 → 0.881（+0.024 [+0.004, +0.050]，3/3）；band_dil 体素数 +34 %（MP3D）/ +21 %（Macarons++）；平均 DistP1 分数 −34 % / −30 %（`A_FIXED_STATE.md`） |
| 4 | 这种排除是否导致真实新增 Coverage 增加或减少？ | **固定状态**：MP3D 被选 Camera 的真实增益 0.570 → 0.876 pp（+0.305 [+0.069, +0.648]，6/9 楼层，几乎全部来自 early 阶段 1.33 → 2.09 pp），Top-1 0.500 → 0.567；Macarons++ 中性（1.110 → 1.126，Top-1 不变）；A0 / A1 选择不同的状态 20 %（MP3D）/ 15 %（Macarons++）。**闭环**：MP3D 小幅增加（见 1），Macarons++ 减少（见 2）——固定状态的中性在闭环里变成退化：稠密 BAND 把只被远距离扫过一次的大面积表面判为"已知"，规划器不再回访补全，Final 覆盖率下降（colosseum / neuschwanstein 最明显；与 Task 3 E1 对 V2-Mix 的观察一致） |
| 5 | V2-Mix 的 Dense/Dense 退化在从零 D100 训练的 V4 上是否仍存在？ | **是**：Task 3 E1（冻结 V2-Mix，D100 输入 + D100 评分）MP3D ΔAUC +0.009 / Macarons++ −0.038；本实验（V4，从零 D100）MP3D +0.018 / Macarons++ −0.043。模式相同，与训练数据密度无关，是评分口径（膨胀 BAND 对"部分观测"表面的判定）的性质 |
| 6 | V4 是否有理由把 D100 评分设为新默认？ | **没有**（NO-GO）。若只针对 MP3D zero-shot 做域特定配置，D100 评分有正向但未达阈值的证据（Final 5/0），可作为室内微调阶段的一个预注册候选，而不是默认 |

## 2. 实验事实
- 唯一变量：`caster.band_dil`（`scoring.new_set_from_depth` 的已知表面排除图）= 稀疏孪生（A0，`T3_P1_MODE=b2`）vs 稠密（A1，`T3_P1_MODE=full`）；模型输入 / 投影 / 解码 / K = 4 / 噪声 / 候选 / n_win 5 / z0 / 去重 / 合法性全部相同（`A_IMPLEMENTATION_AUDIT.md`；GPU 测试 `tests/test_single_variable.py` 逐位核对）。两种 P1 都由同一批已执行记录的真实融合流程构建（S10 = 模拟器 10 % 采样点进 BAND；D100 = 全部有效像素），不是对 D100 抽样。GT 只在评估侧。
- G0：A0 3 步冒烟两域逐位复现存档 V4 episode（位姿、覆盖率、每步分数）；固定状态 A0 与存档 V4 行 365 / 365 逐位相同；稠密评分 = 规划器进程内分数（断言）。
- 固定状态：MP3D VALSEL 90 态、Macarons++ Dev-3 275 态；闭环：MP3D 10 × 150、Macarons++ 30 × 100。
- 资源：G0 冒烟 0.1 GPU-h；固定状态 + A1 闭环 + 覆盖率同时在 gpu115 4 × L40S 运行 16:01–16:52 ≈ **3.4 GPU-h**；作业 3443137；代码提交 `0913283`（脚本 / 审计）、`9af750b`（固定状态）及本报告的提交。
- 边界：未训练 U0、未延长 V4、未改 X-Attn、未做 MP3D 微调、未改 n_win、未改 Frontier / CC-DistP1。

## 3. 文件
`A_IMPLEMENTATION_AUDIT.md`（调用链与 G0）、`A_FIXED_STATE.md`（固定状态表，regret 在附录）、`A_G3_RESULTS.md`（闭环表、配对差、逐场景）、`../results/{a_fixed_summary.json, a_g3.json, a_g3.md}`、`../figures/a_g3_coverage_curves.png`；运行输出 `work_dirs/exp6_v4/ablation_study/experiment_A/`（不入库）。
