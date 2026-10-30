# Task 1 — V3-VAE-XAttn（T1）

与 Task 0 相同的模型、损失、数据与训练配置（`xattn.enabled = true` 是唯一差别），从零独立训练，不读取 T0 权重。

**Cross-Attention**（`seen2scene.v3.cross_attention`，插在 Flow U-Net 瓶颈自注意力之后）：
| 项 | 定义 | 形状 |
|---|---|---|
| Query | 瓶颈特征 token（64×128 图在 8×16 格上，256 通道）+ `Linear(Fourier(每格平均射线方向) ⊕ Fourier(相机位置 / 网格范围))`，经 LayerNorm → `W_q` | `[V, 128, 256]` |
| Key / Value | 场景记忆 token：有观测支持的潜变量格 `[μ (16), band8, free8] ⊕ Fourier(格中心 / 网格范围, 8 频)`；Key 经 LayerNorm → `W_k`，Value → `W_v` | `[V, ≤ 512, 66]` → `[V, 512, 256]`（不足处用 key mask） |
| 选择 | 每候选确定性视锥选择：格中心在视轴 ±(FOV/2·√2 + 10°) 内且距离 ≤ 该视角 tcap；超过预算时按距离排序等距抽样 | 每候选 ≤ 512 |
| 输出 | `H' = H + W_o · Attn(Q, K, V)`，`W_o` 零初始化（初始时 T1 ≡ T0） | `[V, 256, 8, 16]` |

- 同一状态的记忆（token 特征、坐标）只算一次；每候选的选择结果在 K = 4 个样本和 10 个 Euler 步之间复用。
- UNKNOWN / 域外格不进入记忆；mask 全空的视角退化为无注意力。
- 不加 adapter / gate / 额外 head；DistP1 与规划器不变。

**入口**：`scripts/train.py`（= 共享训练器 `--arm t1`）。

**报告**（`report/`）：`T1_AUDIT.md`、`T1_G0_TESTS.md`、`T1_TRAINING.md`、`T1_G12.md`、`T1_G3.md`。
