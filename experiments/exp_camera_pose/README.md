# Camera Pose Phase 1 —— Candidate Pose Optimization（C0 / C1 / C2）

本阶段不训练模型。B2、E2-D100、V4、V5 的 Future Depth 和 DistP1 保持不变，只研究候选相机位姿的选择。
- 审计：`reports/AUDIT.md`
- 预注册协议：`protocol.md`（v1.0，在任何 C1 结果之前冻结）
- 结果：`reports/RESULTS.md`、`reports/FINAL_REPORT.md`

| 目录 | 内容 |
|---|---|
| `configs/` | `c1_lambda.json`（Stage B 按预注册规则选出的 λ）、pilot / G3 的 job 列表 |
| `scripts/` | `frontier.py`（FrontierPrior）、`c1_patch.py`（score_state 钩子）、`c1_{in,out}_planner.py` / `c1_{in,out}_sim.py`（规划器子进程与模拟器包装，原 Task 3 入口不变）、`fixed_prior.py` + `analyze_fixed.py`（Stage B）、`run_queue.sh`（闭环 worker）、分析脚本 |
| `C0/`、`C1/`、`C2/` | 各方法的说明与结果摘要 |
| `reports/` | AUDIT、fixed_state、RESULTS、FINAL_REPORT |

运行产物（日志、固定状态先验、闭环 episode、覆盖率）都在 `work_dirs/exp_camera_pose/`，不进 git。

## 变更记录
- 2026-11-02：完成审计与协议；Stage B 固定状态分析完成，λ*（B2 0.1、E2-D100 0.1、V4 0、V5 0）；开始 Stage A 回归和 Stage C pilot。
