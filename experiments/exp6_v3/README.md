# EXP6 V3 — 从零训练的 3D VAE 场景编码器 + Camera-Ray Cross-Attention

> 2026-10-29 建立。任务书：`EXP6_V3_PLAN.md`（工程与执行规范）、`EXP6_V3_FROM_SCRATCH_PLAN.md`（科学定义）、`EXP6_V3_REPORT_TEMPLATE.md`（主报告模板）、`EXP6_V3_AI_HANDOFF.md`。
> 冻结规则：`protocol.md`。进度与作业：`STATUS.md`。偏差：`reports/V3_DEVIATIONS.md`。

## 1. 研究问题
- **V2 的瓶颈**（EXP5）：V2 / V2-Mix 的 3D 编码器只看 P1 的 8³ 块统计量（EMPTY / BAND 比例），全深度 P1 对冻结模型是分布外输入；E2 微调只在室内带来约 +0.02 AUC。
- **V3 的问题**：
  1. 直接以**真实 Partial TSDF**（历史观测融合的 100% BAND P1：截断有符号距离、BAND / EMPTY / UNKNOWN 三态）为输入、从零训练一个简单 3D 卷积 VAE 作为唯一场景编码器，能否改善未知区域的未来深度预测和闭环 NBV？
  2. 在同一从零训练预算下，在 Flow U-Net 瓶颈处加**一处** Camera-Ray → 场景潜变量的 Cross-Attention，是否有可测的额外收益？
- **不做**：Indoor-only / Outdoor-only 消融、AE-only（β = 0）、改 DistP1、改规划器、从 V2 / E2 权重微调。

## 2. 架构（两组共用，T1 只多一处 Cross-Attention）
```
真实历史深度 (C0,D0..Ct,Dt) --冻结的 D100 P1 融合--> 细网格 sum/cnt/empty/domain (VS = 0.011)
   --状态感知 4x 聚合--> X_t = [sdf, band, free] + valid  (int8/uint8 缓存, work_dirs/exp6_v3/shared_cache/tsdf4)
   --Simple 3D VAE Encoder (stem 32 @4x, stride-2, 64 @8x)--> mu, logvar  (16 ch, 每格 = 8 个细体素)
        |-- 训练: z = mu + sigma*eps -> 3D Decoder -> [sdf_hat, 3 类 logits]  (只重建当前观测状态)
        `-- 条件: Z = mu --> 候选射线 32 点 grid_sample --> Ray Encoder (V2 结构) --> 64 ch/射线
   11 通道确定性射线投影 (cls, t_known, d_lo, d_hi, L_u, n_seg, ...) + 射线上下文 --> 2D Rectified-Flow U-Net (V2 结构, K = 4, 10 步 Euler)
   T1: U-Net 瓶颈 (8 x 16 tokens) 处 H' = H + CrossAttn(LN(H + 射线方向/相机位置 PE), 场景记忆 tokens)
       场景记忆 = 有观测支持的潜变量格 [mu, band8, free8] + 世界坐标 Fourier PE; 每个候选按视锥选 <= 512 个 token
   --解码到命中/深度--> DistP1 (原始 10% 稀疏 P1) --> Greedy-NV
```
- 损失：`L = L_FM + λ_rec · (类别平衡 CE + BAND 上的 sdf L1) + β(step) · KL`（每有效潜变量单元与通道归一）。
- 代码：`seen2scene/v3/`（`partial_tsdf`、`vae`、`cross_attention`、`flow_unet`、`model`、`data`、`losses`、`trainer`、`predictor`、`eval_dev`）；复用 `seen2scene/futureview/` 的射线几何、投影通道、标签、解码、Res 块与 U-Net。

## 3. 数据
- **训练**：MP3D 228 条轨迹（114 个 train 楼层，61 个 train 房屋；随机 / GT 一步贪心各半）+ Macarons++ Train-6 去掉 S-c 的 138 条；**样本 = (scene, trajectory, t, candidate)**。
- **射线通道与标签**：直接复用 E2 的 D100 分片（`work_dirs/exp5_future_view/task3_full_depth_p1/e2/{w2_d100,out_views_d100}`，由冻结构建器在 P1_100 上计算，候选 t\* 只作标签）。
- **VAE 输入**：`shared/build_tsdf_cache.py` 用同一冻结融合重放记录，每个状态存 4× 聚合网格；每个状态都与 D100 分片的粗网格逐位核对（保证同一历史、无未来记录）。
- **验证 / 选择**：MP3D VALSEL（6 个 val 房屋 9 个楼层）、Macarons++ S-c（同训练场景的留出起点，不是未见场景）。
- **评估**：G2 开发集固定状态 = E2 的 VALSEL 90 个 + Dev-3 275 个；G3 = Task 2 / Task 3 的闭环协议（MP3D 5 个楼层 × 2 起点、Macarons++ Test-3 + Dev-3 × 5 起点）。Dev-3 / Test-3 都已在以前的实验中被查看过，不是盲集。

## 4. 目录
| 路径 | 内容 |
|---|---|
| `protocol.md` | 冻结规则（数据、输入、模型、损失、训练预算、选择规则、G0–G3） |
| `configs/model_v3.json`、`configs/train_v3.json` | 两组共用的模型 / 训练配置（T1 只把 `xattn.enabled` 置 true） |
| `shared/` | `build_tsdf_cache.py`（4× TSDF 缓存）、运行脚本 |
| `tests/` | 跨任务测试：TSDF 三态与填充、模型形状与梯度、T0/T1 公共初始化一致 |
| `task0_vae/` | T0：配置、入口脚本、测试、`report/`（T0_AUDIT / G0_TESTS / TRAINING / G12 / G3） |
| `task1_xattn/` | T1：同上，`report/`（T1_*） |
| `reports/` | `V3_COMBINED_REPORT.md`、`V3_COMPARISON_TABLES.md`、`V3_DEVIATIONS.md`、`V3_RESOURCES.md` |
| `work_dirs/exp6_v3/` | 缓存、runs、checkpoints、eval、figures、logs（不进 git） |

## 5. 运行入口（seen2scene 环境；`work_dirs/exp6_v3/run_py.sh GPU <脚本>` 设置环境）
- 缓存：`python experiments/exp6_v3/shared/build_tsdf_cache.py mp3d|outdoor <i> <n>`
- 测试：`pytest experiments/exp6_v3/tests experiments/exp6_v3/task*/tests`
- 训练：`torchrun --nproc_per_node 4 experiments/exp6_v3/task0_vae/scripts/train.py --out <run dir>`（T1 同样，`task1_xattn/scripts/train.py`）
- 评估：`task*/scripts/eval_g12.py`、`eval_g3.py`

## 6. 状态
见 `STATUS.md`。

## 变更记录
- 2026-10-29：创建；Phase 1（V2 归档推送 `efd7e14`）完成；目录、`partial_tsdf` / VAE / Cross-Attn / 模型代码与单元测试、缓存构建器建立。
