# TABLE1_INTEGRATION — H 结果进入 CVPR Table 1 的证据

| 项 | 结果 |
|---|---|
| 注册 | `MODEL_REGISTRY.yaml` 新增 V3-H1 / V3-H2 / V3-H3（role h）；H0 = 原 V3 行（别名，没有新增重复行） |
| 运行 | Table 1 的 `t1_queue.sh`：主队列 195 个 episode，再单独运行 dunnottar 的 15 个，共 210 / 210，0 次失败 |
| 官方覆盖 | 每个 episode 都由 `t1_out_cov.py` 重放，与模拟器存档逐位一致后才导入（`import_episodes.py --new`） |
| 完整性 | Table 1 `validate.py`：V3-H1 / H2 / H3 各 70 个有效，缺失的只有 sofia_church（PENDING），bad 0 |
| 汇总 | Table 1 的 `aggregate.py`（官方 + 单调）、`plot.py`、`results_analysis.py`、`compute_cost.py` 自动生成了 TABLE1*.md、TABLE1_LATEX.tex、results/*、figures/* 中新增的行；旧数值没有改动 |
| H0 | Table 1 V3 行的 70 个记录。H 运行器以 h0 策略重跑 barts_s0，逐位一致；受控计时中的 h0 重跑也逐位一致 |
| 可复现性 | 受控计时中重跑的 H1 / H2 / H3（barts_s0）与 Table 1 记录逐位一致（`RUNTIME_CONTROLLED.md`） |
| 新增 / 复用 / 缺失 | 新增 210 个（H1–H3），复用 70 个（H0 = V3），缺失 0 个（All-14） |
