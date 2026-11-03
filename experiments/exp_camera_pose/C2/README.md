# C2 —— Smooth & Continuous Camera Pose（MP3D 室内）

不训练，不改模型、Flow、K、DistP1，只研究候选位姿的分辨率。协议见 `protocol.md`（v1.0，在任何 C2 结果之前冻结），结论见 `FINAL_REPORT.md`。

| 文件 | 内容 |
|---|---|
| `scripts/c2_poses.py` | 细格（1/4 格、5°）、C0 / C2a / C2b / C2c 候选族、合法性（越界 + GT 线段碰撞）、动作成本 |
| `scripts/c2_oracle.py` | Stage A：VALSEL 90 态的一步 oracle 收益（与 e2_valsel_oracle 同口径；C0 与历史逐值一致） |
| `scripts/c2_analyze_oracle.py` | Stage A 分析与预注册门槛 → `reports/stage_a.{md,json}` |
| `scripts/c2_viz_poses.py` | 相机位姿分布图 → `reports/figures/pose_distribution.png` |
| `scripts/c2_offgrid_smoke.py` | Task 0 网格外 smoke：渲染 → 服务记录 → P1 更新 → B2 DistP1 |
| `configs/` | 族定义写在 `c2_poses.py` 中（冻结），本目录不另存副本 |

运行产物：`work_dirs/exp_camera_pose/C2/{oracle, smoke, logs}`。
