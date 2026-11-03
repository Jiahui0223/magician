# exp_model_diagnostics —— 架构与训练协议审计（2026-11-02，只读）

为后续 Model Diagnostics A/B Study 做准备。本轮**没有**训练，没有运行 G3，也没有修改任何模型或评估代码。

| 文件 | 内容 |
|---|---|
| `REFRESH_AUDIT.md` | Task 0：P1、几何、3D 编码、区域、噪声的刷新频率与真实代码路径（所有模型都是 R10，每处带行号） |
| `ARCHITECTURE_AUDIT.md` | Task 2：8³ 粗网格、F0 / F1、32 点射线、Conv1D → 64 维；改块大小或采样点数需要改什么、是否需要训练 |
| `TRAINING_COMPARISON.md` | Task 1：V2 Original → V2-Mix / B2 → E2-D100 谱系；B2 室外 / E2 室内的已证实差异与假设；V3 / V4 / V5 与 B2 的训练条件差异 |
| `ABLATION_PLAN.md` | 建议的 R10 / R5 / R2 / R1 与 8³ / Ray32 A/B 矩阵，**待确认** |
| `refresh_log_evidence.json` | 历史闭环日志中刷新事件的实测统计（8 个模型 × 室内 / 室外） |

标注约定：【代码】代码确认（路径:行号）·【日志】/【实测】实测·【报告】历史报告·【推导】计算得出·【假设】未验证。

附带更正：`experiments/exp6_v2/E2/README.md` 和 `E2/configs/e2.json` 中 E2 的选择规则改为“按协议固定 EMA@5k”。此前误写为“ES 比值最小”，见 TRAINING_COMPARISON §1。
