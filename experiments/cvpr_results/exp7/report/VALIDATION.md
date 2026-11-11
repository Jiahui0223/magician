# EXP7 验证记录

## MP3D-150（`scripts/collect_mp3d150.py`，2026-11-11 00:40）
176 / 176 条 episode 有效：OURS 44、MAG-B 44、RANDOM 44、OURS-H0 44，其中 30 条复用存档。每条做了以下检查：

| 编号 | 检查 | 结果 |
|---|---|---|
| V1 | 151 个有限覆盖值；150 步；151 个执行位姿；C 不下降 | 176 / 176 |
| V2 | C_obs 重放：用楼层冻结的评价器缓存（30 万 GT 点、可见集、eps）和存储的 svc_t0..150 的 10% 采样点，从头重算曲线 | 176 / 176，最大差值 0.0（逐位相同） |
| V3 | 合法性：每个 executed[t+1] 都在 svc_t 的 `legal_idx` 中；`pose_idx` 与 executed 一致 | 176 / 176 |
| V4 | 同一层同一起点下，各方法起始位姿相同 | 176 / 176 |
| V5 | 来源核对：OURS / OURS-H0 的 ckpt sha 为 7ed0ddd7…，P1 模式为 b2；OURS 的 d_policy 为 d1，用 EXP7 配置，stderr 有 D_POLICY 行，每 10 帧的合法 R / T 与 D1 位姿表逐位相同 | 176 / 176 |

**启动前检查：**
- 22 层位姿表与历史服务记录逐位一致（2,523 个合法位姿）；邻居规则一致（89,520 个位姿）。见 `scripts/check_mp3d_pose_tables.py`。
- Round-1 smoke：MAG-B 与 RANDOM 的前 5 步覆盖值和动作与存档逐位相同。

**结论：** MP3D-150 的结果可以进入 CVPR 表格（室内面板），同时须注明 §4 的来源和可见/留出状态（见 `MP3D_150_RESULTS.md`）。

## UAVScenes（UAV-v0）
- **E-M2 规则：** 对 14 个 Macarons++ 场景逐一复现冻结表的 k_M 和窗口数（`uav_hooks.em2_rule`）。
- **批量线段判定：** 每个切块抽查 300 条边，与官方 `line_segment_mesh_intersection` 的结果一致。
- **smoke：** Ours 与 Random 在 uav_HKairport_00 起点 0 各跑 2 步，返回码均为 0。Ours 的 D1 生效：每步 `policy_s` 约 3.2 s，规划器显存峰值 8.8 GB。
- 正式运行进行中。闭环全部结束后，再做与 Table 1 相同的官方覆盖率重放核对，并单独写入本节。
