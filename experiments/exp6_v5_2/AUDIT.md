# EXP6 V5.2 源码 / 权重审计（2026-11-02，训练前）

全部以仓库实际代码与文件为准。

| 项 | 审计结果 | 证据 |
|---|---|---|
| 父权重 | `work_dirs/exp6_v5/checkpoints/v5_final.pt` sha `5ef30ac4fd0b4a175d1ddd72d78c504c82b31b40b56da68b1d9454533918d543`，与 `runs/v5/ema_012000.pt` 相同；内容 `{ema, step 12000, model_cfg, arm v5, lineage "V4 Outdoor EMA@40k -> v5 +12000"}`；ema 434 张量：encoder 2,412,000、unet 22,333,890、dec 242,757 | sha256sum；`torch.load` |
| V5.2 继承 | encoder + dec 126 张量 2,654,757 参数逐位复制；unet（Flow）不加载；HazardHead 13,953 参数新初始化；dec 中 Flow 条件部分（sum_mlp、adapter、out_scale）14,944 参数冻结不用 | `load_v5_warmstart`、`tests/test_no_gt_leak_no_flow.py` |
| 多表面标签 | 复用 `work_dirs/exp6_v5/labels/v1`（180 条 Train-6 轨迹，16 层剥离、≤ 8 表面，首层 = t* 全部一致；exp6_v5 `LABEL_QUALITY.md`）；`train_v5.Labels` 逐轨迹核对行号 | `train_v5_2.py` |
| 首中真值 | E2 D100 分片自带 `tstar [views, 7296]`（T1.1 标签相机 = 训练射线，float32，模型单位，**不截断到量程**，pisa 例最大 22.6，finite 70 %）；V5.2 规则：t* < tcap → 区间，否则 no-hit | 分片键列表 |
| 射线量程 | `RY.model_rays_torch`：tcap = 70 / dz / k_M（传感器 z 量程 70 运行时单位）；近裁剪 0.5 运行时 z → t_near = tcap · 0.5 / 70 | `seen2scene/futureview/rays.py` |
| DistP1（原评分，供可比） | `scoring.new_set_from_depth`：命中点 o + t d → 细体素 floor(p / VS)（VS = 0.011）→ 场景域内且不在 `caster.band_dil` 中为新；权重 min(1, (z / z0)²)，z = t · k_M · dz（运行时 z 深度）；同体素取 max；DistP1 = 权重和 | `seen2scene/futureview/scoring.py` |
| z0 | Macarons++：MAGICIAN scene_scale / 2（`abl_common.z0_of(scene)`，settings.json 场景盒 × 10）；MP3D：楼层卡片 `z0_rt` | `t24_v2_out_planner.py`、`t06_planner.py` |
| D100 / S10 | 闭环 B2 路径：`p1_full.P1FusionB2` = D100 融合（`input_volume`、`structure`、粗网格 → 网络输入）+ 同一记录的 S10 稀疏融合；`_patch_geometry_b2` 只把 `caster.band_dil` 换成 S10 的（|v| < 0.999 trunc ∧ 域内，3³ max-pool）。V5.2 固定状态评估在一次回放里同时维护两张图并复制同一规则（`v52_common.b2_geometry`） | `p1_full.py`；V5 回放逐位一致（下行） |
| 平台核对 | 用 V5.2 的评估器（两张图同时回放）重算 V5 的 DistP1：Dev-3 barts_s0 10/10 状态与 exp6_v5 的 V5 行**逐位相同** | `work_dirs/exp6_v5_2/eval_fixed/dev3_g0_init` |
| 规划器挂点 | 室外 `v2_planner.py`、室内 `t06_planner.py` 都只调用 `v2_plan_core.score_state(core, geo, h, legal_R, legal_T, eps, z0)` 并用 `out["distp1"]` 选点（室外 `m_score.choose`，室内 argmax + prefer-unvisited）；编码器每 10 步刷新（R10）。V5.2 只替换这一个函数 | 源码 |
| 结构常量 | `geometry.EMPTY, UNKNOWN, BAND = 0, 1, 2`；粗网格 = (EMPTY, BAND, 域内) 比例 × 255，块 8 | `geometry.py` |
| 评估集 | G2：VALSEL 90、Dev-3 B0 150 + B2 125 态（与 V5 相同）；oracle `gain`（候选真实一步覆盖增益）与 `tstar` 只在评估端读取 | `e2_fixed.states`、`analyze_g12_v4.rows_to_recs` |
| G3 smoke 可行性 | VALSEL 楼层（8194nk5LbLH_L0、EU6Fwq7SyZv_L0）有场景卡片，`sim_in` 可直接闭环（2 步冒烟通过） | `eval_smoke/g0_init` |
| S-c 标定可行性 | Train-6 轨迹服务记录（`data/macaronspp_belief/v1/trajectories/<scene>/<traj>/records`）含合法候选 `legal_R/T`，可用真实回放标定归一化下限 | `calibrate_floors.py` |
| 资源 | gpu115（作业 3443137，至 11-04 15:11）+ gpu112（作业 3447847，本轮申请，72 h）；QOS 每用户 8 GPU | `squeue`、`sacctmgr` |
