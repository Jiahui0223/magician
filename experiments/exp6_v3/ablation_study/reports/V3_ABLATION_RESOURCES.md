# EXP6 V3 Ablation — 资源记录（持续追加）

| 日期 | 部分 | 资源 | 备注 |
|---|---|---|---|
| 2026-10-31 | 审计 + 协议 | 0 GPU-h（CPU） | |
| 2026-10-31 | Experiment A 打分（`ab_score.py`：Pool FX 215 + Pool CL 1,461 状态，B2 / E2D / T0 / T1 K=4 + GTD；含三路冒烟与 6 状态重跑） | 8 × L40S（gpu117 + gpu115），09:43 → 10:22 ≈ **4.8 GPU-h**（每状态约 0.5 s 打分 + 融合重放；峰值 MP3D 18 GB、室外 8 GB） | 缓存 4.0 GB（原始 K=4 样本 + 射线字段），`work_dirs/exp6_v3/ablation_study/cache/` |
| 2026-10-31 | Experiment A / B 分析（`a_analyze.py`、`b_analyze.py`） | CPU，各 2–5 min | |
| **合计** | **Ablation A + B** | **≈ 4.8 GPU-h（L40S）+ CPU**；无训练；无新 Slurm 申请（gpu117 3429774、gpu115 3435033 既有分配） | 2026-10-31 09:30 → 10:45 |
