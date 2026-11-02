# G0 验证 —— V5.2（2026-11-02 00:35–00:56，训练前全部通过）

单测：`experiments/exp6_v5_2/tests`（5 个文件 21 项，CPU，seen2scene env），命令：
```bash
cd experiments/exp6_v5_2/tests
for t in test_*.py; do CUDA_VISIBLE_DEVICES="" LD_PRELOAD=$S2E/lib/libtbb.so PYTHONPATH=$REPO $S2E/bin/python $t; done
```
冒烟证据：`work_dirs/exp6_v5_2/runs/g0/`、`eval_fixed/dev3_g0_init/`、`eval_smoke/g0_init/`、`logs/g0_*.log`。代码提交 `c34cb02`（训练与冻结前）。

| 门 | 检查 | 结果 | 证据 |
|---|---|---|---|
| 父权重 / EMA | `v5_final.pt` sha `5ef30ac4…` = `ema_012000.pt`；encoder + dec 126 张量逐位复制；unet 不加载；父文件 sha / mtime 加载前后不变 | ✅ | `test_no_gt_leak_no_flow::test_no_flow_in_model_and_exact_warm_start` |
| 标签缓存 | 复用 exp6_v5 的 Train-6 多表面标签（180 轨迹）；`train_v5.Labels` 逐轨迹核对行号 | ✅ | 训练日志 `label store 8` |
| D100 / S10 分离 | 一次回放同时维护 D100（网络输入、结构）与 S10（band_dil）两张图，规则复制 `p1_full._patch_geometry_b2`；用该几何重算 V5 的 DistP1：Dev-3 barts_s0 10/10 状态与 exp6_v5 V5 行**逐位相同** | ✅ | `eval_fixed/dev3_g0_init/B0/barts_s0.jsonl` |
| 坐标 / 量程 | 区间 [tcap · 0.5 / 70, tcap]（近裁剪 / 传感器 z 量程）；单调、正宽度 | ✅ | `test_firsthit_distribution::test_constant_hazard_independent_of_bins` |
| 首中分布归一 | Σ p_first + p_nohit = 1（误差 ≤ 1e-6），ρΔt ∈ {0, 1e-9, 0.5, 5, 100, 1e6} 无 NaN / Inf；常数 hazard 的 p_nohit 与解析解一致且与区间数（32 / 64 / 128 / 512）无关；真实模型（随机 hazard）与无局部 token 的射线（网格外）归一、有限 | ✅ 3/3 | `test_firsthit_distribution` |
| 两墙 3 / 9（MP3D 尺度 2.8 / 8.4 模型单位） | 不透明 A 已知时 B 的可见 < 1e-2，B 留在 P_surf（远遮挡折扣后仍为只有 A 时的 > 10 倍）；A 透明度 0.3 时 B 的可见 > 10 倍；可见全部落在首墙；已知 BAND 的不透明墙仍遮挡 | ✅ | `test_visibility_and_occlusion`（6/6） |
| 无命中 / 超量程 / 已确认 FREE | 无墙或墙在 tcap 之外：S_vis = 0、p_nohit = 1；墙所在体素是 D100 FREE：可见质量全部被 q_valid 去掉并计入 `mass_free` | ✅ | 同上 |
| 同体素多射线 / 距离 | 8 条相同射线 = 1 条（max 去重）；w_dist = min(1, (z / z0)²)（z0 加倍 → 0.25）；δ 首中极限下 S_visible 与冻结 DistP1（`new_set_from_depth`）相差 < 5 %；区间数 64 / 128 / 256 的 S_visible 相差 < 10 % | ✅ 4/4 | `test_voxel_dedup_and_distance` |
| EMPTY / UNKNOWN 与空间潜力 | frontier（旁有已确认 FREE）> 同样 UNKNOWN 格但无 FREE（> 2 倍）；全部已观测 → 0；不透明预测墙之后 < 0.25 × 通畅；改变 b（0.05 ↔ 0.95）P_space 不变（不是 1 − b） | ✅ 4/4 | `test_space_potential` |
| 无 GT 泄漏 / 无 Flow | 模型无 unet / velocity；打分 API 无 GT 参数；打分时 Flow 采样、深度解码、DistP1 调用 0 次；P1 张量哈希不变；规划器守卫在 V5.2 进程中对这些调用计数并抛错；训练只读 Train-6 | ✅ 4/4 | `test_no_gt_leak_no_flow` |
| 规划器闭环 | 未训练 V5.2（λ 0.1）2 步闭环：MP3D VALSEL 8194nk5LbLH L0 S10、Macarons++ alhambra s0 均 exit 0；每步守卫计数 {flow_sample, pred_cond, pred_sample, decode_depth, distp1_new_set, distp1_new_mask} 全 0；分数有限 | ✅ | `eval_smoke/g0_init/*/v52_steps*.jsonl` |
| 训练 overfit | 300 步固定 batch（H 100 → J）：总损失 4.01 → 1.59；首中 NLL 3.60 → 1.18、最高区间准确率 0.26 → 0.64、众数误差 192 → 11 vox；表面召回 0.74 → 0.77、第二+ 0.61 → 0.69（不塌缩） | ✅ | `runs/g0/overfit/run_overfit.json` |
| β | 联合设置 20 步共享主干梯度中位数：L_surface 0.536、L_firsthit 1.631，比值 3.04 ∈ [0.2, 5] → β = 1 | ✅ | `runs/g0/gradscale/run_gradscale.json` |
| resume | 单卡 A 40 步 vs B 20 + 20（H → J 切换处恢复）：model / EMA 最大差 0.0（第一次失败：生成器状态被加载到 GPU，修正为 `.cpu()` 后通过） | ✅ | `runs/g0/resume_check.json` |
| DDP | 4 rank 300 步：3.65 → 2.06；Stage H 0.095 s/步、1.2 GB/卡 | ✅ | `runs/g0/ddp_short/run_short.json` |

冻结前修复的问题：① `log h` 的运算符优先级（−expm1 先 clamp 再取负）→ NaN，已修正并加单测；② 评估器未在 no_grad 下导出张量；③ resume 生成器状态设备；④ 两个单测预期本身写错（远遮挡折扣下 B 的 P_surf 量级、frontier 对照设计），修正测试而不是打分代码。
