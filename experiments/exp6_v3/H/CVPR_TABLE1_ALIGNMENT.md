# CVPR_TABLE1_ALIGNMENT — H 如何接入 CVPR Table 1

1. **注册**：`experiments/cvpr_results/table1/MODEL_REGISTRY.yaml` 新增 V3-H1 / V3-H2 / V3-H3（role h）。
   - H0 是原 V3 行的**别名**，没有重复登记，也没有重新计算数值。
   - 每条注册都包含：checkpoint sha（与 V3 行相同）、策略代码 / 冻结配置的路径，以及推理设置（b2：D100 输入 / S10 评分，R10，K4，Euler10）。
2. **运行**：由 Table 1 自己的队列 `scripts/t1_queue.sh` 运行。
   - 新增的 `run_sim` 分支调用 `exp6_v3/H/scripts/sim_out_h.py --policy h{1,2,3}`；
   - 模拟器（`t24_out_sim`）与规划器链（`v3_out_planner` 的同一套补丁）都没有改动，只在规划器进程内替换 `score_state` 的打分。
3. **覆盖与导入**：每个 episode 由 `t1_out_cov.py`（即 `t25_out_cov`，官方覆盖重放，必须与模拟器存档的曲线逐位一致）处理，再由 `import_episodes.py --new` 写入规范记录 `work_dirs/cvpr_results/table1/episodes/V3-H*/`。H 的轨迹副本即为同一份 g3 目录下的 episode、规划器日志与服务记录。
4. **汇总**：Table 1 的 `validate.py`、`aggregate.py`（官方 + 单调）、`plot.py`、`results_analysis.py` 自动生成新增的行与图，不手工修改任何旧数值。
5. **H0 一致性**：H 运行器以 h0 策略重跑 barts_s0，与 Table 1 V3 记录逐位一致（`reports/G0_REGRESSION.md`）。所以 H0 直接使用 Table 1 V3 行的 70 条记录，未重跑其余 69 条。
6. **本地分析**：`exp6_v3/H/scripts/aggregate_local.py` 只读取 Table 1 的规范记录，额外计算 H 的逐级配对（H1 − H0、H2 − H1、H3 − H2 等）与决策统计。
