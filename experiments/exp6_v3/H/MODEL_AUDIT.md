# MODEL_AUDIT — 唯一冻结模型 V3-T1

| 项 | 值 | 来源 |
|---|---|---|
| checkpoint | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt`，与 `runs/main/ema_040000.pt` 字节相同 | Table 1 `MODEL_REGISTRY.yaml`（V3 行） |
| sha256 | `639c9d3ae3f3c069dc0399dc4cdd44102be2bf20e70d666a6a529c6affc62bdb`；H 的每次运行中，规划器报告的 sha 都与此相同 | `reports/G0_REGRESSION.md`；episode 的 `planner_hello.ckpt_sha256` |
| 结构 | V3-T1 = 自训练稠密 3D VAE + Scene Cross-Attention（xattn 开）+ 2D rectified flow，23,730,986 个参数 | `exp6_v3/configs/model_v3.json`；`seen2scene.v3` |
| 训练 | 从零开始的混合训练 40k 步（MP3D 228 条轨迹 + Train-6 去 S-c 138 条，D100 + TSDF 缓存，每步 32 + 32 个视角）；按冻结的选择规则选出 EMA@40k | Table 1 注册表；`exp6_v3/configs/train_v3.json` |
| 加载 | `seen2scene.v3.predictor.load_core(ckpt, device, steps, expect_sha)`，经由 `planner_v3.V3Core`，加载时检查 sha | `exp6_v3/shared/planner_v3.py` |
| 本实验中的训练 | **0 次**；没有重新选择 checkpoint | — |
| VAE 说明 | V3 的 VAE 是从零训练的稠密 Conv3D VAE，不是预训练的 Seen2Scene Sparse VAE | `exp6_v5/F1/V3_HISTORY.md` |
