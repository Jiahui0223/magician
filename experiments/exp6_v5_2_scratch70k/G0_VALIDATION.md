# V5.2-M G0 验证（正式训练前，2026-11-02 02:50–03:39）

结论：G0 通过，没有阻断，于 03:39 启动正式 Scratch70k。

有一项单元测试（two_walls）失败了两次，原因都是测试本身的预期写错（见第 1 节），模型代码没有改动。最终复跑结果见第 1 节。

第三次全量复跑（unbuffered 输出）是在训练开始后、任何评估结论出来之前运行的。

日志位于 `work_dirs/exp6_v5_2_scratch70k/logs/g0_*.log`，运行目录位于 `runs/g0/`。

## 1. 单元测试（CPU，`tests/test_v52m_g0.py`）
| 测试 | 内容 | 结果 |
|---|---|---|
| parameters_and_scratch | 总参数 26,999,143；不同 seed 的权重不同（随机初始化）；训练脚本中不出现任何 V4 / V5 / V5.2 权重路径或 warm-start 函数；模型中没有 Flow U-Net 张量 | PASS |
| forward_backward_all_params | 整幅视角前向和反向；首中 PMF 加 no-hit 归一（误差 1e-5）；每个参数都有有限且非零的梯度 | PASS |
| free_before_hazard_and_two_walls | 两墙 3 m / 9 m（tcap 9.4），FREE 内 1.0 处放一个虚假墙（不透明度 0.95）；FREE 在墙前结束 / 在墙所在区间内结束两种情形 | 预期修正后 PASS（见下） |
| known_crossing_band_truncation | KNOWN 射线：只有含 t_known 的区间得到偏置，交叉之前为 FREE、之后不强制；known_bias 非负；EMPTY 射线在 t_max = 3.0 出网格时截断 | PASS |
| domain_gating | 前一半区间不在 domain 内时，不被强制为 FREE | PASS |
| replay_adapter_equals_v52_aggregation | 回放适配器的 S_vis / P_surf / P_space 与 V5.2 原生聚合一致（误差 1e-6）；世界体素去重、距离权重、新颖性都沿用 V5.2 已测代码 | PASS |
| no_flow_no_distp1_calls | `M.sample`、`G.decode`、`new_set_from_depth`、`new_mask_from_depth` 都被替换为计数器，整幅视角 belief 的调用数为 0 | PASS |

**two_walls 两次失败，原因都是测试预期写错（模型代码未改）**
- **测试设置**：D100 FREE 恰好在墙 A 的位置（2.8）结束，所以墙 A 所在区间有 54.7% 是 FREE。按区间内密度均匀的定义，这个区间的光学深度乘以 (1 − free_frac)，墙 A 不再是完全不透明。
- **第一次运行（03:2x）**：断言“墙 A 质量 > 0.99”失败，实测 0.956，等于解析值 1 − exp(−6.91·0.453)。
- **第二次运行**（同一断言改为解析值之后，09:2x 结束）：断言“墙 B 质量 < 0.01”失败。原因相同：部分 FREE 的墙 A 会透过 4.4%，B 实测 0.0435，与解析值 exp(−a_A)·0.999 一致。
- **最终写法**：测试拆成两个情形。
  - 情形 1：FREE 在 2.6 结束（墙 A 之前）。A = 0.9988，B = 0.0010；按错误顺序（先算透射率、再删 FREE 区间）A 只剩 0.0498。
  - 情形 2：FREE 在墙所在区间内结束。A、B 都与解析值相差 < 5e-3（0.9561 / 0.0435）。
  - 两个情形中 FREE 区间内的首中质量都 < 1e-6，且 PMF 归一。
- 单独运行该测试：PASS。**全部 7 项复跑（2026-11-02 09:26，`logs/g0_unit_tests_v3.log`）：ALL PASS。**

## 2. 训练链 smoke（gpu115 / gpu112，L40S）
| 项目 | 结果 |
|---|---|
| 小数据 overfit（200 步，1 卡） | loss 3.096 → 0.510（−83.5%） |
| short（200 步，1 卡，8 views） | 2.91 → 2.32；0.458 s/step；6.7 GB |
| β 梯度量级（Train-6，最后 20 步中位数） | ‖∇L_surface‖ 2.17，‖∇L_firsthit‖ 6.06，比值 2.79，在 [0.1, 10] 内，β = 1（FROZEN_PROTOCOL §3） |
| 第一次 gradscale | 报错：某个 batch 没有有效表面标签时，L_surface 返回 Python 0.0 而不是张量。已改为返回 `logit.sum()·0`，复跑通过 |
| resume，1 卡 | A 连续 30 步 vs B 15 + 15 步：模型与 EMA 最大差均为 0.0（逐位一致） |
| resume，DDP 2 rank | 同上，最大差 0.0 |
| world size 2 → 4 恢复 | 从 step 30 恢复到 45：模型、优化器、EMA 严格恢复，数据 RNG 重新播种并记录 |
| DDP 4 rank short（gpu115） | 3.01 → 2.12；0.999 s/step（每 rank 16 views）；6.9 GB |
| **跨分配双节点 8 rank**（gpu115 + gpu112，ib0，200 步） | 2.90 → 2.14；**0.574 s/step**（每 rank 8 views）；6.8 GB；相对 4 rank 加速 1.74 倍 |
| 全量数据 4 rank 40 步 | 加载 138 + 42 条轨迹约 2.5 min；holdout 在第 0 / 20 / 40 步各约 25 s；best_train_holdout、标签权重、last、manifest 都正确写出；选择分数包含 KnownPassThrough |
| 采样分层份额（训练首中射线） | 全部：known-FREE 0.229、known-BAND 0.249、UNKNOWN 命中 0.273、UNKNOWN no-hit 0.248、近处命中 0.011、远处命中 0.165；自然比例（均匀射线）：0.319 / 0.278 / 0.198 / 0.206；有第二+表面的标签射线 0.364 |

## 3. 评估链与规划器 smoke
| 项目 | 结果 |
|---|---|
| S-c floor 校准（dry，1 场景） | 正常输出 score 配置 |
| 固定状态评估器 | VALSEL 8194nk5LbLH L0（10 态，89 s）和 Dev-3 barts_s0（10 态，38 s）；每态打分 0.8–1.1 s；首中指标与 belief dump 正常 |
| G1 / G2 分析 | 与 B2 / E2 / V4 / V4C / V5 / 旧 V5.2 的同态行正确合并（B2 回放一致性断言通过） |
| 多表面 belief | 网格 GT（Dev-3 17 s / episode，VALSEL 5 s）→ 区间 belief 指标正常 |
| **规划器 2 步闭环**（30 步权重，λ 0.1） | MP3D VALSEL 8194nk5LbLH L0 S10 和 Macarons++ alhambra s0 都 exit 0；守卫计数（flow_sample、pred_cond、pred_sample、decode_depth、distp1_new_set、distp1_new_mask）全部为 0；分数有限；打分 0.8–1.3 s / 步 |
| D100 / S10 角色 | 固定状态和规划器都沿用 V5.2 已验证的 `b2_geometry`：D100 负责输入、结构、FREE / q_valid，S10 的 band_dil 负责新颖性 |
