# Experiment E — 稳定性比较与 colosseum 低覆盖平台诊断（2026-11-04）

**协议**：`PROTOCOL.md`（E-v1；偏差记录在末尾）。**结论**：`FINAL_REPORT.md`。

| 文件 | 内容 |
|---|---|
| `STABILITY_COMPARISON.md` | Task 1：D1 vs YBEST-H3，逐分组 / 逐场景 / 逐起点，集中度、退化事件、2 × 2 |
| `COLOSSEUM_DIAGNOSIS.md` | Task 2：平台的表现、局部增益、预测审计、风险项、前瞻、分支续跑、噪声重复与结论 |
| `results/` | 自动生成的表（`STABILITY_TABLES.md`、`PLATEAU_TABLES.md`、`REFUTE_TABLES.md`）与 csv / json |
| `FIGURES/` | 稳定性图，colosseum 曲线、轨迹、高度、空间对比、预测与真实增益 |

**`scripts/` 中的脚本**：

| 脚本 | 用途 |
|---|---|
| `stability.py` | Task 1 全部表与图（CPU，只读 Table 1） |
| `e_sim.py` | 一条闭环：rep / 分支根目录；冻结模拟器与规划器子进程不变 |
| `e_branch.py` | 用源 episode 的前 T 步准备分支根目录（只复制） |
| `e_queue.sh` | 闭环 worker |
| `e_oracle.py` | 评估侧 GT 反事实（MAGICIAN 环境） |
| `e_oracle_queue.sh` | GT 反事实 worker |
| `e_predict.py` | 离线重建规划器预测并与 GT 比对（seen2scene 环境） |
| `e_predict_queue.sh` | 预测重建 worker |
| `plateau_analysis.py` | colosseum 表与图 |
| `figures_spatial.py` | 成功 / 失败的空间对比图 |
| `occlusion_check.py` | 视线遮挡检验 |
| `refute_analysis.py` | 反证记忆离线重打分 |
| `occlusion_rescore.py` | 已观测表面不透明的离线重打分 |
| `chain_after.sh` | 接力启动 |

**运行输出**：`work_dirs/exp6_v3/ablation_study/experiment_E_stability_plateau/`（`runs/`、`oracle/`、`logs/`、`jobs/`、`EVENTS.jsonl`）。

**复现**（在仓库根目录运行）：
- 分析：`python <E>/scripts/{stability,plateau_analysis,refute_analysis,occlusion_check,occlusion_rescore}.py`
- 闭环：`bash <E>/scripts/e_queue.sh <GPU> <jobfile>`
- GT 反事实：`bash <E>/scripts/e_oracle_queue.sh <GPU> <jobfile>`
- 预测重建：`bash <E>/scripts/e_predict_queue.sh <GPU> <jobfile>`
- 任务文件在 `work_dirs/.../jobs/`。
