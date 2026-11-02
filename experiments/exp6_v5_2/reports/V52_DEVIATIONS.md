# EXP6 V5.2 偏差与实现取舍

| # | 时间 | 项 | 任务书 | 实际 | 理由 / 影响 |
|---|---|---|---|---|---|
| D1 | 冻结前 | 目录 | `experiments/exp6_v5_2/` | 附件原放在 `experiments/exp6_v5.2/`（只含该文件），已移到 `exp6_v5_2/` 并删除空目录 | 按任务书与用户消息中的目录名统一 |
| D2 | 冻结前 | 配置格式 | `configs/*.yaml`（6 个） | JSON：`model_v52.json`、`train_v52.json`（Stage H / J 同一文件）、`score_v52.json`（λ ∈ {0, 0.1} 同一文件）；fixed-state / smoke 设置写在 `FROZEN_PROTOCOL.md` 与 job 文件 | 仓库其余实验均用 JSON；任务书允许"不为满足示意目录强制复制" |
| D3 | 冻结前 | 脚本名 | `audit.py`、`eval_g3_smoke.py` 等 | 审计用现有工具 + `AUDIT.md`；`eval_g1_g2.py`、`eval_g3_smoke.py`、`eval_g3_full.py`、`calibrate_floors.py` 等 | — |
| D4 | 冻结前 | 首中区间 | 48–64 个，近表面增强 | 64 个，log1p(t / 16 VS) 均匀（近处约 1 VS，远处相对宽度约 6 %）；打分时每区间 8 个子点分配体素 | 单测：区间数 64 / 128 / 256 的可见分差 < 10 %，δ 极限 ≈ DistP1 |
| D5 | 冻结前 | HazardHead 输入的 b | 输入 b_r(t_i) | b 以**截断梯度**输入 | 防止首中损失把墙后 b 压低（任务书 §3.2 的语义要求） |
| D6 | 冻结前 | 首中标签 | 可信有效量程内的真实首中 / No-Hit；无效 / 截断 Mask | 分片 T1.1 t*（不截断，覆盖整条射线）：t* < tcap → 区间，否则 → no-hit；t* < 近裁剪 → mask | T1.1 的 t* 来自全网格光栅化，量程内是否有命中是确定的，没有截断问题 |
| D7 | 冻结前 | q_valid | "用 P1_D100 已确认 FREE 检查物理冲突（若引入额外掩码，单列报告）" | 落在 D100 EMPTY 体素上的首中质量不计入 S_visible；被去掉的比例单列（G2：VALSEL 39 %、Dev-3 23 %） | 已单列报告 |
| D8 | 冻结前 | P_surf 的距离 / 置信折扣 | 对远遮挡表面施加折扣 | κ = exp(−max(0, t − t̄) / 128 VS)，并乘 min(1, Δt / Δ) 与 w_dist | 128 VS 取自 V5 评估中远遮挡（≥ 128 VS）召回只有 0.11–0.16 |
| D9 | 冻结前 | P_space 的可达性 | 与真实 FREE 的可达边界关联；无合法路径 = 0 | frontier = 26 邻域内有 ≥ 25 % 已确认 FREE 的 UNKNOWN 粗格（权重 1），其余 UNKNOWN 0.1；被预测遮挡由 T 自然压低；**未做运动可达性 / 路径推断** | 任务书 §8 允许的保守近似 |
| D10 | 冻结前 | 归一化 | 从 Train-6 或同一候选集合估计尺度 | 每项除以 max(候选集最大值, floor_k)，floor_k = 0.05 × S-c 60 态每态最大值的中位数（训练后、G2 / G3 前标定一次：vis 29.1、surf 153.8、space 15.7） | 两者结合：候选集内归一 + Train-6 下限防止放大噪声 |
| D11 | 冻结前 | G3 smoke 的 MP3D 楼层 | MP3D VALSEL 2 楼层 × 1 起点 | VALSEL episode 列表前两个（8194nk5LbLH L0 S10、EU6Fwq7SyZv L0 S67）；这两层没有既有闭环，所以同时跑了 V5 / V4C 参照 | 预先冻结；VALSEL 两层较简单（Final 0.96–0.99） |
| D12 | 冻结前 | 全量 G3 放行规则 | "结果有信号再做全量" | 预注册：(a) 某 λ 的 G2 增益在某一域 ≥ 0.9 × V5，或 (b) smoke 平均 AUC ≥ V5 − 0.01；结果 (a) 在 MP3D 成立（0.565 ≥ 0.437），(b) 不成立 → 按规则运行全量 G3 | 规则在看到任何结果前写入 `FROZEN_PROTOCOL.md`（提交 `c34cb02`） |
| D13 | 冻结后 | 分析脚本格式化 | — | `analyze_g1_g2.py` 第一次运行时 Markdown 格式化报错（JSON 已写出），修正格式化后重跑 | 不影响任何数值 |
