# C2 —— Multi-Scale / Coarse-to-Fine Candidates：未运行
- **室外 NOT_SUPPORTED**：MAGICIAN 环境只执行 settings.json 网格上的 `pose_space` 索引（`update_camera(idx)`），碰撞和覆盖率都在该网格上。改变网格就是修改官方环境。
- **室内只能作为新动作协议**：t12 网格模拟器可以渲染任意位姿、检测任意线段。但改步长或角度档需要：新协议下的 C0 重跑、网格外候选的新 oracle、路径成本记账。
- 按 `../protocol.md` §2 的预注册，Phase 1 不做，因此没有任何 C2 结论。
