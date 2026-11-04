# PROVENANCE — EXP6 V3 ablation D

| 项 | 值 |
|---|---|
| 模型 | 冻结的 V3-T1 `final.pt`，sha256 `639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb`（不使用 Y 的权重） |
| 参照 | H3：`experiments/exp6_v3/H`（H-v1，只读）；Table 1 的 V3-H3 行 |
| 代码 | `scripts/d_policy.py`，导入 `H/scripts/h_policy.py`（只读，sha 记录在每个规划器日志的 D_POLICY 行） |
| 配置 | `configs/d_config.yaml`（D-v1） |
| 位姿表 | `work_dirs/exp6_v3/H/pose_tables`（H 中已逐位核对） |
| 代码提交 | 见 git log（D 的设置提交） |

## 资源
- **D 的评估节点**：gpu115，Slurm 作业 3443137（4 × L40S，24 个 CPU，EndTime 2026-11-04 15:11）。
- **运行方式**：Table 1 的 `t1_queue.sh` worker，每张卡最多 2 个；episode 各自独立运行。
- **之后**：Y 在 50K 停止后，gpu114 的 4 张卡也会加入评估。

GPU：
  - gpu115 GPU 0: NVIDIA L40S (UUID: GPU-a3308452-0200-07b5-ba1f-1b29c570e3d7)
  - gpu115 GPU 1: NVIDIA L40S (UUID: GPU-52c3e31f-26b2-2a83-7004-184f2c81f3fa)
  - gpu115 GPU 2: NVIDIA L40S (UUID: GPU-1194e737-9b1c-567e-6785-54b9b9eb2441)
  - gpu115 GPU 3: NVIDIA L40S (UUID: GPU-eb9dbd00-04ba-9a1e-8e90-884c58be23f9)
