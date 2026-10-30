# EXP5 Task 2：V2-Mix 联合训练与最终对照（状态表）

> 依据：你 10-26 的任务书。审计：`T2.0_audit.md`；协议：`protocol.md`；偏差：`DEVIATIONS.md`；冻结：`FROZEN_CONFIGS.json`。
> 代码：`scripts/`；配置：`configs/`；运行输出：`work_dirs/exp5_future_view/task2/`（`W2`）；权重：`work_dirs/weights/`（W00–W21）。

| 阶段 | 状态 | 产物 |
|---|---|---|
| T2.0 审计 + 决定 | **DONE**（10-26）：4 项决定见 `../decision_log.md` | `T2.0_audit.md` |
| checkpoint 整理 | **DONE**（10-26）：W00–W21 硬链接归档；删除 64 个冗余文件，共 34.7 GB | `work_dirs/weights/README.md`、`DELETED.json` |
| T2.1a 室内训练数据 | **DONE**（10-27 01:30）：114 个 train 楼层、228 条轨迹、263,358 个视角；选择集 9 个楼层、10,473 个视角 | `W2/mp3d/`、`data_summary.json` |
| T2.1b V2-Mix 微调 | **DONE**（10-27 17:12）：20k 步，选中第 17,500 步（sha256 `343f41ab…`）；室内选择集 ES 15.5 → 6.0，室外 S-c 3.08 → 3.15。两次节点掉线，从 checkpoint 续训，见偏差日志 | `W2/training/mix_v1/`、W21 |
| T2.2 共享协议与适配 | **冒烟通过**（室内 V2O；室外 V2O 与 Task 1 E1 逐位一致；室外 MAG-G；室外单调评估器） | `scripts/t23_*`、`t24_*`、`t25_*` |
| T2.3–T2.6 正式评估与报告 | **DONE**（10-27）：160 / 160 条 episode，审核通过（帧数、合法性、guard、checkpoint、复现）；表格、图、动图和最终报告已完成 | `reports/TASK2_FINAL_REPORT.md`、`tables/`、`W2/figures/` |
