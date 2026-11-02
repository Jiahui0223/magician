# EXP6 V5 审计（2026-11-01，训练前）

所有数字来自本仓库实际运行：`scripts/audit_v4_shapes.py` → `reports/audit_shapes.json`（gpu115，L40S），标签测试 `work_dirs/exp6_v5/labels/test/test_pisa.json`。

## 1. V4 起点（只读，不覆写）
| 项 | 值 |
|---|---|
| checkpoint | `work_dirs/exp6_v4/checkpoints/final.pt`，sha256 `1b31afffbe4da2d94344e0a509b711b30561a34bc4aa521773f1595aafac00d4`（与 V4 报告一致），step 40000，EMA@40k，arm v4 |
| 内容 | `{ema, step, model_cfg, arm}`，无优化器；V4 的 `runs/main/ckpt_last.pt` 学习率已余弦退火到 0，V5 / V4-Continue 都**不**沿用该调度器状态（新优化器、新调度，见 `FROZEN_PROTOCOL.md`） |
| 参数 | encoder 2,412,000 / rays（32 点 RayEncoder）76,224 / ray_xattn 25,430 / unet 22,333,890，合计 24,847,544 |
| 实测形状（bannerman cg-A-0001 t = 85，7 视角） | 粗网格 uint8 [3, 52, 60, 32] → cin [1, 4, 52, 60, 32]；F0 [1, 32, 52, 60, 32]（格 8 VS = 0.088）；F1 [1, 64, 26, 30, 16]（格 0.176）；32 点特征 [51,072, 97, 32]（网格内比例 0.32）；H_old [51,072, 64]；cond [7, 76, 64, 128]；velocity [7, 2, 64, 128] |
| H_old 尺度 | 均值 0.0002，标准差 0.0304，RMS 0.0304（V5 适配器输出尺度初值因此取 0.05，初始 H_new RMS 0.027） |
| 网格范围 | 该状态 4.58 × 5.28 × 2.82 模型单位；tcap 3.2–5.0 模型单位；7,296 射线 / 视角（64 × 114） |

## 2. V5 实现核对
| 项 | 值 |
|---|---|
| 模型 | `seen2scene/futureview/v5_model.py` V5Model；继承 encoder + unet（382 个张量逐名逐形状复制，24,745,890 参数，`tests/test_teacher_migration.py`） |
| 新模块 | V5RayDecoder 242,757 参数（ffn 66,176、q_mlp 27,520、f_proj 20,864、geo_mlp 19,712、wq/wk/wv/wo/s_z 各 16,512、adapter 10,624、s_out 8,321、sum_mlp 4,256、s_t 1,920 等） |
| 同一状态上的 V5 前向 | 每射线 token 上限 45（null 1 + F0 24 + F1 12 + coarse 8），有效 token 均值 17.6、最小 9；z [51,072, 128]；belief 探针 [51,072, 16]；cond [7, 76, 64, 128] |
| 旧采样 | V5Model 无 `rays` / `ray_xattn` 属性；前向中 grid_sample 调用数 0（`tests/test_v5_forward_backward.py`）；规划器侧 `v5_core` 对预测器前向加 grid_sample 抛错保护 |

## 3. 数据与隔离
| 项 | 值 |
|---|---|
| 训练 | Macarons++ Train-6（bannerman, dunnottar, fushimi, liberty, pisa, redeemer）去 S-c：138 条轨迹（98,887 视角，V4 同一列表 `v2_data.load_split("train6_minus_Sc")`），E2 D100 视角分片 `work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100` |
| 监控 | S-c：同 6 场景 42 条留出轨迹（31,027 视角）；只监控，不选权重 |
| 多表面标签 | 只从 Train-6 网格生成（`t0_common.assert_geometry_allowed(scene, "train")`），180 条轨迹 = 训练 138 + S-c 42 |
| 不进反传 | MP3D（任何楼层）、Macarons++ Dev-3 / Test-3 网格；评估端 belief GT 只在评估后计算指标（Dev-3 用 `dev_eval` 权限，MP3D VALSEL 网格），不参与训练或选权重 |
| MP3D 训练样本 | 0 |

## 4. 坐标与单位
- Macarons++ 运行时 = y-up，OBJ × 10（非公制）；模型坐标 = SceneFrame（origin_z, k_M），t_model = zbuf / dz / k_M。
- MP3D 运行时 = y-up，1 m = 7 运行时单位；模型单位 = 运行时 / k_I（k_I = 楼层高度 / 2.816）；10 步窗口（n_win = 5 的 Level 区域）每块重建 P1 区域，模型坐标随区域更新（V2 / V4 原逻辑，V5 不改）。
- 训练射线、标签射线、规划器射线是同一函数 `RY.model_rays_torch`（标签相机 `RY.label_intrinsics(jitter)` 把 64 × 114 像素中心正好放在规划器射线上）。
- V5 的检索格约定 = V4 grid_sample（align_corners = False）的格约定（`tests/test_ray_world_geometry.py`）。
- 度量尺度：V2 / V4 的全部几何都在模型单位里，Macarons++ 的模型单位不是米；V5 的深度编码用模型单位 / 体素单位（S0 = 8 VS），与 V4 一致（偏差说明 D1）。

## 5. 多表面标签可行性（关键风险）
| 检查 | 结果 |
|---|---|
| 合成双墙（墙 A 3 运行时单位，墙 B 9，重复面，近裁剪前一面墙；两种 jitter） | 7,296/7,296 射线恰好 2 个表面；深度误差 ≤ 2.4e-6；近裁剪前的墙不出现 → PASS |
| 与 trimesh 精确多交点（`multiple_hits=True`）对比，pisa cg-A-0001 4 视角 2,000 条真实射线 | 一致率 1.000（0 条不一致） |
| 第 1 层 = T1.1 已存 t*（同相机同光栅器） | 每条轨迹 first == t* 比例 1.0000，最大相对误差 6e-8 |
| K 的选择 | K = 8 时 liberty 层数用满 22 %；改为剥离 16 层、存 ≤ 8 个表面，用满比例降到 4.6 %；用满（K 层后未解析）的射线把监督区间截到第 16 层，第 8 个之后的表面截在两表面中点，**不当作负例** |
| 结论 | 标签可靠，满足多表面监督的前提（详细统计见 `reports/LABEL_QUALITY.md`） |

## 6. 规划器（冻结）
`v2_plan_core.Core` + `predictor.V2Predictor` + `scoring` + 原候选生成器；K = 4、10 步 Euler；MP3D n_win = 5；P1 模式 b2（网络输入 D100，DistP1 新颖性用 S10 稀疏 P1）。V5 只替换 Core 内的模型类（`scripts/v5_core.py`），闭环脚本 = V4 的 sim 包装 + V5 规划器子进程（`sim_{in,out}_v5.py`、`v5_{in,out}_planner.py`）。Experiment A 的 D100 评分**不**使用。

## 7. 资源
| 作业 | 节点 | GPU | 时限 |
|---|---|---|---|
| 3443137 vscode-dev | gpu115 | 4 × L40S，24 CPU，500 GB | 至 2026-11-04 15:11 |
| 3445003 exp6-v5-aux（本轮申请，`-x gpu118`） | gpu116 | 4 × L40S，32 CPU，240 GB | 2026-11-01 19:14 → 11-04 19:14 |
冒烟实测：V5 每步 0.66 s（单卡 16 视角，峰值显存 4.5 GB），V4-Continue 0.34 s（10.5 GB）。
