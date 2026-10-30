# T0 — G0 测试（2026-10-30；持续追加，数值全部来自真实运行）

## 1. 清单
| # | 测试 | 命令 / 文件 | 结果 |
|---|---|---|---|
| 1 | 静态检查：语法、导入（V3 包 10 个模块 + 实验脚本） | `python -c "import ast..."`，`pytest` 收集 | 通过 |
| 2 | 三态语义、填充、blob 往返、重建目标（合成融合数组） | `tests/test_partial_tsdf.py` | 通过（2 / 2，06:10） |
| 3 | 模型形状、FM → μ 梯度、Recon / KL → μ 与 logvar 梯度、T0/T1 公共初始化哈希一致、`W_o` 零初始化、Query 随相机变化 | `tests/test_model_shapes.py` | 通过（4 / 4，06:10） |
| 4 | 缓存 = 实时聚合（室外 bannerman t = 0, 1, 10, 55, 99）；`structure_lean` = `fu.structure()`；V2 粗网格 = E2 分片 | `tests/g0_real_data_check.py` | **全部 True**（5 个状态逐位相同） |
| 5 | 缓存构建时每个状态的粗网格核对（同一历史、无未来记录） | `shared/build_tsdf_cache.py`（STOP 规则） | 通过：417 / 417 条轨迹（62,550 个状态）全部逐位通过（第一版 `structure_lean` 的边界 bug 被该检查发现并修复）；缓存 49 GB |
| 6 | 真实数据批：T0 / T1 前后向有限；梯度到 VAE 编码器（μ 头 6.15、logvar 头 3.23）、解码器、U-Net | `tests/g0_real_data_check.py` | 通过；FM 对射线编码器 / Cross-Attn 的梯度在第 0 步为 0（U-Net 输出卷积与 `W_o` 零初始化，V2 同样），第 10 步非零（见 #7） |
| 7 | 单卡 tiny overfit（固定批 8 + 8 视角，200 步，lr 1e-4，β ≈ 0） | `train.py --smoke overfit` | **通过**：loss 3.67 → 0.62（−83%）；FM 1.81 → 0.34；Recon 1.86 → 0.21（三类准确率 0.27/0.25/0.39 → 0.97/0.98/0.98，sdf L1 0.66 → 0.15）；KL/unit 0.088 → 0.31（无压力时上升）；第 10 步梯度范数 vae_enc 2.18 / vae_dec 3.61 / rays 0.014 / unet 10.5，全部有限非零 |
| 8 | 4 卡 DDP profile（100 步，与 13 个缓存构建进程共享 GPU / CPU，非干净计时） | `torchrun --nproc_per_node 4 ... --smoke profile` | 通过：1.45 s/步（p90 1.64），每卡峰值 16.3 GB；参数 23,539,622（推理 22,954,434）；DDP 下 loss 有限、梯度全局求和正常。**干净计时见 #10** |
| 10 | 干净 4 卡 profile（缓存构建结束后，100 步） | `g0_after_cache.sh` → `--smoke profile` | 通过：T0 0.94 s/步（p90 1.08），每卡峰值 15.8 GB，2.62 h / 10k 步；**注意**：profile 的室外子集只含 bannerman（最小场景之一），含 dunnottar 等大场景的平均步时见 #12 |
| 11 | DDP resume 测试（A 60 步 vs B 30 + 30 步） | `g0_resume_test.sh` | ⟨运行中⟩ |
| 12 | 大场景状态的单步开销（单进程，1 MP3D + 1 室外状态，8 + 8 视角，fwd + bwd，T0） | `bench_big.py` → `logs/bench_big_scenes.json` | bannerman 0.81 s / 12.9 GB；pisa 0.86 s / 13.5 GB；fushimi 0.70 s / 12.8 GB；**dunnottar 1.83 s / 27.4 GB**（278 × 398 × 64 格）。DDP 步时 = 各 rank 最大值，dunnottar 占室外状态 1/6 → 估计平均 1.3–1.4 s/步，仍在 46 GB 内 |
| 9 | 射线与网格范围 | `tests/g0_real_data_check.py` | 射线采样点在网格内的比例：MP3D 31%（区域 13.7 m 裁剪，与 V2 相同）、Macarons++ 72%；室外相机可在场景框外（与 V2 几何一致） |

## 2. 实测规模
| 量 | 值 |
|---|---|
| 4× 网格 | MP3D 266 × 246 × 64 / 242 × 216 × 64（按区域）；bannerman 100 × 116 × 64 |
| 潜变量 | MP3D 133 × 123 × 32；bannerman 50 × 58 × 32 |
| 缓存大小 | MP3D 150 状态 103 MB（0.69 MB / 状态）；bannerman 100 状态 24.6 MB |
| 前后向峰值显存（2 状态 16 视角，单进程） | 14.5 GB（真实数据检查，无激活检查点关闭）；训练循环 11.2 GB |
| 单卡每步（每卡批，缓存构建并行占用下） | 0.78–0.83 s |

## 3. 校准依据（协议 v1.0 冻结值，2026-10-30 06:30）
- 初始损失尺度：FM 1.81–1.97、Recon 1.74–1.86（CE 1.2 + L1 0.6–0.7）、KL/unit 0.08；无压力时 200 步内升到 0.31 → **λ_rec = 1.0，β_max = 0.02（5k 步预热）**：KL 项约为重建项的 1–4%，保持真实后验且不压制条件信息。
- 步时：干净 profile 0.94–0.96 s/步（小场景子集），含 dunnottar 后估计 1.3–1.4 s/步 → **预算 40,000 步**（每组约 15 h；两组顺序 + G1–G3 可在当前分配到期前完成并留余量；60k 无余量，未采用）。
- lr 1e-4（预热 1k，余弦到 0）：overfit / profile 下梯度范数 5–15、损失单调下降、无 NaN。
- 配置哈希：`configs/FROZEN_SHA256.json`。
