# EXP7 缺失与未完成项（2026-11-11 00:45）

| 项 | 状态 | 原因 / 下一步 |
|---|---|---|
| MP3D-150：Ours / MAGICIAN / Random / H0 | ✅ 完成，176 / 176 | — |
| MP3D-150：MAG-G、MACARONS、SCONE、FisherRF | ⬜ 未运行 | 任务书只要求 MAGICIAN beam 与 Random。MACARONS 在室内没有现成的网格适配；SCONE 和 FisherRF 在室外就是 BLOCKED |
| MP3D 官方 5 个场景、5,000 步 | ⬜ 不在范围内 | 150 步是本研究的短预算协议，论文 MP3D-5000 的数字不填入本表 |
| UAVScenes Ours / Random（6 块 × 10 起点） | 🔄 运行中 | 结果出来后补 `UAVSCENES_RESULTS` |
| UAVScenes MAGICIAN | ⬜ 未开始 | 按用户安排放在 Ours / Random 之后；需要把 `t24_mag_out_planner.py` 写死的数据根目录重定向 |
| UAVScenes AMvalley 空洞 | ⚠️ 已知局限 | 约 9% 的 mesh 空洞（水面等）可能让相机钻到地形下方；起点已要求头顶开阔，结果中将单独统计 |
| Pool B 第二次申请（3494406） | ✖ 已取消 | 测试显示必须抢占他人作业才能启动，违反"不抢占他人 GPU"；作业从未启动 |
