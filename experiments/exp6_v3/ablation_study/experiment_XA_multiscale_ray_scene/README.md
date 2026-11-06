# Experiment XA — Multi-Scale Geometry-Aware Ray–Scene Attention

- 任务书：`experiments/exp6_v3/XA_Experiment_Protocol.md`
- 冻结协议（变体定义、训练、评估、完成标准、偏差）：`FROZEN_PROTOCOL.md`
- 模型代码：`seen2scene/v3/xa/`
- 进度：`experiments/exp6_v3/STATUS.md` 的 P12；事件与偏差：`logs/EVENTS.md`；链日志：`logs/chain_<节点>.log`
- 最终报告：`reports/FINAL_REPORT.md`（全部变体完成后生成）

| 路径 | 内容 |
|---|---|
| `configs/` | 六个变体的模型配置和训练配置（JSON） |
| `scripts/train_xa.py` | 用不改动的 V3 训练器训练一个变体 |
| `scripts/run_chain.sh` | 一个节点上的链：训练 → 选模 → 室外 70 条 + 室内 6 条 → 聚合 |
| `scripts/xa_sim_out.py`、`xa_out_planner.py` | 室外闭环（冻结模拟器 + D1） |
| `scripts/xa_sim_in.py`、`xa_in_planner.py` | 室内闭环（冻结模拟器 + H0） |
| `scripts/xa_eval_queue.sh`、`xa_out_cov.py` | 闭环 worker、官方覆盖重放 |
| `scripts/select_final.py`、`aggregate.py` | 选模、聚合 |
| `tests/xa_checks.py` | 启动前的工程检查 |
| `manifests/` | 代码 / 配置哈希、划分、基线哈希、CT 暂停清单 |
| `results/<id>/` | 每个变体的表格、逐条结果、状态 |
| `reports/` | 汇总表、逐场景表、曲线图、最终报告 |

大文件（训练状态、checkpoint、闭环原始输出）在 `work_dirs/exp6_v3/ablation_study/experiment_XA_multiscale_ray_scene/`，不入库。
