# Experiment C — GT-Depth Closed-Loop Evaluation（EXP6 V3 Ablation）

任务书：用户 2026-10-31 指令（V4 暂停；A / B 冻结不改）。协议：`protocol.md` v1.0。状态：`../STATUS.md` C 段。报告：`report/`。

| 文件 | 内容 |
|---|---|
| `scripts/c_common.py` | 环境无关定义：路径、组、episode 列表、cq 旁路文件读写与键白名单、状态哈希、oracle 摘要、区域框 |
| `scripts/c_gt.py` | GT 查询（magician env）：`IndoorGT`（t\* 渲染 = EXP5 S1；C_obs 单步增益 = T0.12-C；区域诊断）、`OutdoorGT`（t\*；T7a 快照 / 恢复的单调 + 官方增益；可达诊断） |
| `scripts/c_in_sim.py` | MP3D 闭环（t23 循环体 + GT 旁路 + 诊断 + 执行一致性检查）；组 C1 / C2 / B2replay |
| `scripts/c_in_planner.py` | MP3D 子进程（seen2scene env）：t06 区域 / 稀疏 P1，无模型；C1 = GTD DistP1，C2 = 读增益；`nv_choice` |
| `scripts/c_out_sim.py` | Macarons++ 闭环：`v2_sim.run` 不改 + `EpisodeSimC`（在线单调覆盖）+ `PlannerC`（GT 旁路 + 诊断）；组 C1 / C2 / V2M（B2 回放） |
| `scripts/c_out_planner.py` | Macarons++ 子进程：v2_planner 协议 / 日志 / resume，无模型；C1 = GTD DistP1，C2 = 读增益；`m_score.choose` |
| `scripts/c_scoring.py` | `gtd_scores`（= Experiment A GTD 规则）、`p1_hash` |
| `scripts/c_out_cov.py` | `t25_out_cov.py` 不改，根目录指向 Experiment C |
| `scripts/run_queue_c.sh` | 单卡 worker 队列（claim 跨节点共享） |
| `scripts/c_analyze.py` | 9 组（MAG-G / MAG-B / B0 / B2 / E2D / T0 / T1 / C1 / C2）配对统计、曲线、停滞与诊断汇总 → `results/`、`figures/`、`report/` |
| `tests/` | `test_c_rules.py`（CPU 单测）、`g0_run.sh`（GPU 冒烟）、`g0_check.py`（G0 判定 → `report/C_G0_VALIDATION.md`） |

大文件（episode、svc、cq、frames、metrics）在 `work_dirs/exp6_v3/ablation_study/experiment_C/`，不提交。
