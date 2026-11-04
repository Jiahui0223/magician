# PROVENANCE — EXP6 V3 / Y

## 父状态与代码
| 项 | 值 |
|---|---|
| 父训练状态 | `work_dirs/exp6_v3/task1_xattn/runs/main/ckpt_last.pt`，sha256 `3d81978206c88068c5dd5eda9cb221c8023948854549184b8aa35274c5941ce9`；step 40000；包含 model、EMA、AdamW（394 项）、RNG |
| 父 EMA（Table 1 V3） | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt`，sha256 `639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb`；与上面 ckpt_last 中的 EMA 逐张量相同（394 个张量） |
| 架构 | V3-T1：`experiments/exp6_v3/configs/model_v3.json`，xattn 开启；没有改动 |
| 训练器 | `seen2scene/v3/trainer.py`（没有改动），由 `Y/scripts/train_y.py` 替换 CFG_DIR 与 lr_at |
| 配置 | `Y/configs/train_y.json`（Y-v1） |
| 代码提交 | 6468076（Y 设置）；之后的提交见 git log |

## 资源
- **Y 的训练节点**：gpu114，Slurm 作业 3461041（4 × L40S，60 个 CPU，EndTime 2026-11-08 18:30）。
- **启动方式**：在该分配内用 torchrun 启动，单节点，c10d rendezvous 使用独立端口（Y_PORT）。
- **QOS 限制**：账户最多同时 8 张 GPU，已被 3443137 与 3461041 两个分配占满，所以不再提交新的 sbatch。

GPU：
  - gpu114 GPU 0: NVIDIA L40S (UUID: GPU-9252f57c-7309-f086-e4c4-7f74d8f8e4e0)
  - gpu114 GPU 1: NVIDIA L40S (UUID: GPU-09c71672-be5d-d04f-ebc7-b3eddbfbb44d)
  - gpu114 GPU 2: NVIDIA L40S (UUID: GPU-396dbac1-1908-fe3f-66cc-e060b9969455)
  - gpu114 GPU 3: NVIDIA L40S (UUID: GPU-24c12117-447f-bf9d-e554-f371248c7165)

## 冒烟
- **续训冒烟**：`work_dirs/exp6_v3/Y/smoke/resume`。从父状态续训到 40030（每 10 步保存一次），再续到 40060，用来验证训练器能从 Y 自己写出的 ckpt_last 继续。结果见 `reports/00_STATUS_AND_RESUME.md`。
