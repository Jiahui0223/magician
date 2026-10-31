# EXP6 V3 Ablation — 资源记录（持续追加）

| 日期 | 部分 | 资源 | 备注 |
|---|---|---|---|
| 2026-10-31 | 审计 + 协议 | 0 GPU-h（CPU） | |
| 2026-10-31 | Experiment A 打分（`ab_score.py`：Pool FX 215 + Pool CL 1,461 状态，B2 / E2D / T0 / T1 K=4 + GTD；含三路冒烟与 6 状态重跑） | 8 × L40S（gpu117 + gpu115），09:43 → 10:22 ≈ **4.8 GPU-h**（每状态约 0.5 s 打分 + 融合重放；峰值 MP3D 18 GB、室外 8 GB） | 缓存 4.0 GB（原始 K=4 样本 + 射线字段），`work_dirs/exp6_v3/ablation_study/cache/` |
| 2026-10-31 | Experiment A / B 分析（`a_analyze.py`、`b_analyze.py`） | CPU，各 2–5 min | |
| **合计** | **Ablation A + B** | **≈ 4.8 GPU-h（L40S）+ CPU**；无训练；无新 Slurm 申请（gpu117 3429774、gpu115 3435033 既有分配） | 2026-10-31 09:30 → 10:45 |
| 2026-10-31 | Experiment C G0（两轮冒烟：室内 C1 ×2 / C2 / B2 回放 5 步，室外 C1 / C2 ×2 / B2 回放 3 步 + 覆盖率回放） | 4 × L40S（gpu117），2 × 2 min ≈ 0.3 GPU-h | `work_dirs/exp6_v3/ablation_study/experiment_C/g0/` |
| 2026-10-31 | Experiment C 正式闭环：C1 40 条（室内 384 s / 条、室外 243 s / 条，含每步诊断）+ C2 40 条（112 / 196 s / 条）+ 室外覆盖率回放 60 条 | 8 × L40S（gpu117 3429774 + gpu115 3435033），15:45 → 16:27 ≈ **5.97 GPU-h**（C1 11,333 s、C2 7,132 s、覆盖率 3,027 s）；子进程峰值 13.8 GB（室内 C1）/ 8.7 GB（室外 C1） | 输出 11 GB（svc / cq / ckpt / 日志），`experiment_C/eval/`；不提交 |
| 2026-10-31 | Experiment C 分析（`c_analyze.py`） | CPU 2 min | |
| **合计** | **Ablation A + B + C** | **≈ 11.1 GPU-h（L40S）+ CPU**；无训练；无新 Slurm 申请 | 2026-10-31 09:30 → 16:45 |
