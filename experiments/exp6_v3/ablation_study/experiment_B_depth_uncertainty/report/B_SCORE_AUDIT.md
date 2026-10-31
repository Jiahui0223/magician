# Experiment B — 评分器审计（协议 §3；真实数据不变量 + 单元测试）

| 检查 | 结果 |
|---|---|
| 单元测试 `tests/test_b_scorer.py`（熵、HIT/NO-HIT、span 归一化、无 far 代入、候选置换不变性、P2 归一化） | 见 pytest 日志（2 passed） |
| cl/indoor：T1 的 S_depthU 范围 | [0.0000, 0.3464]（368 状态、3050 候选；M_unknown 射线占比均值 0.548，valid 射线占比 1.000；全 0 状态 0） |
| cl/outdoor：T1 的 S_depthU 范围 | [0.0000, 0.4758]（1093 状态、7689 候选；M_unknown 射线占比均值 0.532，valid 射线占比 0.911；全 0 状态 9） |
| fx/indoor：T1 的 S_depthU 范围 | [0.0000, 0.2654]（90 状态、702 候选；M_unknown 射线占比均值 0.564，valid 射线占比 1.000；全 0 状态 1） |
| fx/outdoor：T1 的 S_depthU 范围 | [0.0000, 0.3594]（125 状态、875 候选；M_unknown 射线占比均值 0.424，valid 射线占比 0.855；全 0 状态 0） |
| GT 不参与评分 | `b_scorer.py` 只读 `t, hit_u, valid, n_u, d_lo, d_hi, distp1_k`；`oracle.npz` 只在评估侧（regret、诊断）读取 |
| 跨相机比较 | 无：DepthU 在候选内按射线聚合后才与其他候选比较 |
| NO-HIT 处理 | 不参与深度 std；只进入 U_hit 的 Bernoulli 熵；无效射线排除 |