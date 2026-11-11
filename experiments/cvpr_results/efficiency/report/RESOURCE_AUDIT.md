# EXP8 资源与代码审计（2026-11-11 01:40）

## 1. Slurm 与 GPU（启动 EXP8 之前）
| 作业 | 状态 | 节点 / GPU | 用途 | EXP8 是否使用 |
|---|---|---|---|---|
| 3489889 `vscode-dev` | RUNNING | gpu114，4 × NVIDIA L40S（46 GB） | EXP7 UAVScenes 闭环（12 个 worker，4 卡利用率 100%，各卡 11–23 GB）；MP3D-150 已于 00:40 完成 | 否（禁止占用） |
| 3491984 `vscode-dev` | RUNNING | gpu107，1 × P100 | 另一个会话 | 否 |

- **配额：** 每用户 GPU 上限 8 张（QOS gpu）。已占 5 张，最多还能申请 3 张。
- **空闲资源：** gpu116 空闲 3 张，gpu117 空闲 2 张（GRES 均标为 L40）。
- **EXP8 新申请：** 3 个 1 卡作业。每个提交前都用 `sbatch --test-only` 确认可以立即启动、不抢占他人作业。

| 作业 | 节点 | 用途 |
|---|---|---|
| 3494525 `eff_A_ours` | gpu117（GPU 1） | Ours：主延迟、模块拆分、K / 深度缩放 |
| 3494526 `eff_B1_mag` | gpu117（GPU 2） | MAGICIAN beam，4 条轨迹 |
| 3494528 `eff_B2_mag` | gpu116（GPU 0） | MAGICIAN beam，3 条轨迹，另加 2 条 MAGICIAN 自己的轨迹 |

- **独占：** 每个作业独占 1 张 GPU，作业内各次运行顺序执行，同时只有一个规划器进程。
- **实际分到的 GPU：** `nvidia-smi` 显示均为 NVIDIA L40S 46 GB（Slurm GRES 标签写作 L40）：3494525 用 gpu117 GPU 1，3494526 用 gpu117 GPU 2，3494528 用 gpu116 GPU 0。型号与主实验的 gpu114 相同。同节点的其他 GPU 上有别人的作业在跑，共享 CPU、PCIe 和内存带宽，但不共享我们的 GPU。
- **与 Table 1 生产耗时的区别：** Table 1 日志里的耗时是每卡同时跑 2–3 个进程时测的；EXP8 是独占测量。两者不能直接比较。

## 2. 代码入口（全部冻结，EXP8 未修改任何一个文件）
| 方法 | 入口 | 内部阶段（已有的同步计时） |
|---|---|---|
| RAYSCOUT（V3-T0@40K + D1） | `experiments/cvpr_results/scripts/table3/kd_out_planner.py`（K = 4，D-v1 策略 d1；Table 3 已回归验证与 Table 1 V3-T0-D1 逐位相同）→ `exp5_future_view/task1/scripts/v2_planner.py` → `v2_plan_core.score_state` | `p1_s`（读记录并融合进 P1，即地图更新）；`geometry_s`（规划几何构建）；refresh 的 `encode_s`（t % 10 == 0 时执行 coarse 聚合 + VAE 编码 + 噪声）；`projection_s` / `cond_s` / `sampling_s` / `decode_score_s`（第一层候选）；`policy_s`（D1 前瞻，含第二层候选的生成）；`decision_s`（从读取记录到选出动作）。各阶段都用 `torch.cuda.synchronize` 同步 |
| MAGICIAN beam（10 × 10） | `exp5_future_view/task2/scripts/t24_mag_out_planner.py --method MAG-B` → `task0/magician_comparison/adapters/magician_core.py`（官方 e697009 核心；E1 测试逐位等价） | `observe_s`（地图更新：proxy / surface 填充、雕刻）；`occ_s`（MACARONS/SCONE 占用场，即 imagined Gaussians 的来源）；`novelty_s`（渲染所有历史位姿）；`beam_s`（每个展开的候选渲染一次 Gaussian 深度和图像）；`decision_s` = observe + plan。计时用 `time.time()`，没有显式同步；但阶段边界上的布尔索引和 `.item()` 会隐式同步。EXP8 以驱动端测得的往返墙钟时间为准 |

## 3. 已有可用状态
- Table 1 的规范 episode 与服务记录：`work_dirs/cvpr_results/table1/g3/closed_loop/outdoor/<model>/<group>/svc/<tag>/svc_<scene>_s<k>_t<t>.npz`。每步包含 zbuf、mask、10% 采样点、合法候选和执行历史，正是规划器在线时收到的全部输入。
- 已有记录：V3-T0-D1 的 14 × 5 条全部存在；MAG-B 只有一部分场景有记录（bannerman、dunnottar、eiffel、fushimi、liberty、pisa、redeemer、sestino_museum）。
- 生产日志中的每步耗时（`decision_s`）是在 3 个 worker 共享一张 GPU 时测的，只用作参考：Ours 每条均值 4.9–9.4 s，MAG-B 15.9–20.4 s。

## 4. 测量方法（详见 BENCHMARK_PROTOCOL.md）
- **离线重放：** `scripts/eff_replay.py` 把一条真实轨迹的服务记录逐步喂给冻结的规划器子进程，协议与闭环模拟器完全相同，不做仿真、渲染或覆盖率评估。
- **严格配对：** 两种方法在同一组轨迹上运行（7 条 Ours 轨迹），每一步的输入观测完全相同，但各自维护自己的内部地图。
- **确定性核对：** 规划器在自己的轨迹上重放时，选出的动作应与当时的记录一致（Ours 7 条、MAG-B 2 条）。
