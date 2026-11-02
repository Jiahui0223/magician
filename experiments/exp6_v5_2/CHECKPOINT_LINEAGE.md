# EXP6 V5.2 —— Checkpoint 谱系

| 名称 | 文件 | 父节点 / 来源 | 步数 | 规则 | sha256 |
|---|---|---|---|---|---|
| V5-Warm12k（父） | `work_dirs/exp6_v5/checkpoints/v5_final.pt` | V4@40k → V5 Bridge 2k + Joint 10k（exp6_v5，只读） | 12,000 | 固定 EMA@12000 | `5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543` |
| V5.2 冒烟初始 | `work_dirs/exp6_v5_2/checkpoints/smoke_v52_init.pt` | V5 encoder / dec + 未训练 HazardHead（seed 0） | 0 | **只用于 G0 冒烟** | — |
| **V5.2-Pilot** | `work_dirs/exp6_v5_2/checkpoints/v52_final.pt`（发布自 `runs/v52/ema_006000.pt`） | V5 encoder + V5RayDecoder 逐位复制（Flow U-Net 不加载）；HazardHead 新；Stage H 2k + Stage J 4k | 6,000 | 固定 EMA@6000（训练前冻结） | `4982bc6673687f6968d84222a56a6b4d8c9a11f443c9f719bf3791273e3b9297`（训练 01:13 EXIT 0，无塌缩） |

`λ = 0`（V52L0）与 `λ = 0.1`（V52L01）共用同一个 `v52_final.pt`，只在推理打分时不同；只训练一次。
