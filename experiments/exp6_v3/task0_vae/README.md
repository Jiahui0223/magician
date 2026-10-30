# Task 0 — V3-VAE（T0）

**模型**：`seen2scene.v3.model.V3Model`，`xattn.enabled = false`。
- 输入：4× 聚合的 Partial TSDF `[sdf, band, free]` + valid（`seen2scene.v3.partial_tsdf`；缓存 `work_dirs/exp6_v3/shared_cache/tsdf4`）。
- 场景编码器：`VAEEncoder`（stem 32 @4× → Res3D → stride-2 64 @8× → 2 × Res3D → 1×1 → μ / logvar，16 通道）；训练期 `VAEDecoder` 重建 `[sdf_hat, 3 类 logits]`。
- 条件：`LatentRayEncoder`（V2 的 Conv1d 结构）沿候选射线 32 点采样 `[μ, valid8]` → 64 通道；加 11 通道确定性投影 → 2D Rectified-Flow U-Net（V2 结构）。
- 损失：`L_FM + λ_rec (类别平衡 CE + BAND 上 sdf L1) + β(step) KL`。

**接口**（与 Task 1 完全相同，Task 1 只多一处 Cross-Attention）：
| 调用 | 输入 | 输出 |
|---|---|---|
| `encode(x, valid)` | `[1, 3, X4, Y4, Z4]`、`[1, 1, X4, Y4, Z4]` | `mu, logvar [1, 16, X4/2, Y4/2, Z4/2]`、`valid8`、`band8`、`free8`、`ext` |
| `view_cond(h, o, d, tcap, ch)` | `[V, 3]`、`[V, 7296, 3]`、`[V, 7296]`、`[V, 11, 7296]` | `cond [V, 76, 64, 128]` |
| `velocity(x, tau, cond)` | `[V, 2, 64, 128]` | `[V, 2, 64, 128]` |
| `decode(z)` | `[B, 16, ...]` | `[B, 4, X4, Y4, Z4]` |

**入口**：`scripts/train.py`（= 共享训练器 `--arm t0`）；评估脚本在 `../shared/`。配置：`../configs/{model_v3,train_v3}.json`。

**报告**（`report/`）：`T0_AUDIT.md`、`T0_G0_TESTS.md`、`T0_TRAINING.md`、`T0_G12.md`、`T0_G3.md`。
