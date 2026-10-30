# T1 — 审计（2026-10-29 / 30；与 T0 共享 `../../task0_vae/report/T0_AUDIT.md` 的全部内容，这里只记录差异）

## 1. 与 T0 的唯一差别
`configs/model_v3.json` 的 `xattn.enabled = true` → `FlowUNet.xattn = CrossAttnBlock(256, 66, 96, heads 4)`，插在瓶颈自注意力之后（`seen2scene/v3/flow_unet.py`）。其余模块、损失、数据、采样、优化完全相同。

## 2. 形状
| 张量 | 形状 | 说明 |
|---|---|---|
| 场景记忆特征 | `[N, 18]` = μ (16) + band8 + free8 | N = 有观测支持（band8 > 0 或 free8 > 0）且域内的潜变量格 |
| 记忆位置编码 | `[N, 48]` | Fourier（8 频 × sin/cos × 3 轴）of 2·(格中心 / 网格范围) − 1 |
| 每候选 token | `[V, 512, 66]` + key mask `[V, 512]` | 视锥选择；不足 512 用 mask |
| Query PE | `[V, 128, 96]` | 8 × 16 格的平均射线方向 Fourier (48) ⊕ 相机位置 Fourier (48) |
| Q / K / V | `[V, 4, 128, 64]` / `[V, 4, 512, 64]` | 4 头 × 64 维 |
| 新参数 | 见 `T1_G0_TESTS.md`（param_counts） | `W_o` 零初始化 |

## 3. 正确性要点（G0 测试覆盖）
- Q 随候选相机变化：query PE 含该候选的射线方向与相机位置；瓶颈特征本身也依赖候选的 cond。
- K/V 只来自当前历史地图：记忆由 `encode(x_t)` 的 μ 构造；UNKNOWN / 域外格不是 token；不同状态不混用。
- 梯度：训练时记忆 token 由 μ 直接 gather（无 detach）；`W_o` 零初始化使 T1 在第 0 步与 T0 输出相同，`W_o` 先获得梯度，之后 Q/K/V 获得梯度（单元测试验证）。
- 缓存：同一状态的记忆一次构造；每候选的 token 选择在 K = 4 样本与 10 个 Euler 步间复用（`V3.sample` 以 `index_mem` 按视角索引）。

## 4. 失败与修复
见 T0 审计 §6（共享代码）。
