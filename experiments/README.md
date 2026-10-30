# Experiments（CVPR 2027）

总体思路、阶段划分、止损规则和排期见仓库根目录的 `plan.md`。本文件写实验细节和进度。

## 约定

**目录**
- `experiments/` 下只有实验文件夹：`exp_vis`、`exp1_infra` 到 `exp8_joint_semvoxflow`（2026-10-03 重新编号，见下面的第二张对照表），**不建 task 文件夹**。
- 【2026-10-08】在 exp3 和 exp4 之间插入 `exp3b_macaronspp_transfer`（EXP3b — MACARONS++ Transfer Baseline），其余编号不变（见第三张对照表）。
- 【2026-10-03】例外：EXP3 下按算法版本分子目录（`pre_vhd/`、`vhd_v1/`、`vhd_v1_1/`、`vhd_v2/` …）；VHD 的新版本不再占用新的 EXP 编号。

**编号**
- task 在每个实验内部单独编号，引用时写 `exp3-task2`。

**复述流程**（每个实验开始前）
1. 我在对话中完整复述这个实验：每个 task 的步骤、参数、产出、通过标准，以及每一步的理由。
2. 你检查后没有修改，就按复述的内容执行。
3. 你有修改，就先把修改写回本 README，再以修改后的版本执行。
4. 不一次改动太多已有的信息。

**进度**
- 每完成一个 task，或者状态有变化，**立即**更新本文件最后的进度表。

**结果**
- 汇总表和结论写在 `experiments/<exp>/results.md`，实验开始时创建。
- 【2026-09-27】每个实验完成后，写一份工作说明 `experiments/<exp>/<exp 编号>.md`（例如 `exp1_infra/exp1.md`），说明每个 task 做了哪些操作和修改、目的和结果。
- 本 README 只写实验内容、修改和进度，不写细节。完成的部分标 (done)；task 有修改时同步更新本文件。
- 原始输出（日志、TensorBoard、网格、checkpoint、csv）放在 `work_dirs/<exp>/task<k>_<短名>/{runs,eval,figures}`。
- 运行时用 `--run-root work_dirs/<exp>/task<k>_<短名>/runs`。

**运行**
- 生成器和 ControlNet 推理必须加 `--precision bf16-mixed`。
- 用 `CUDA_VISIBLE_DEVICES` 控制卡数。
- 算力：2 个节点（A、B），每个节点 4 张 L40S，作为两个独立的 4 卡作业使用。

### 原编号 → 新编号（2026-09-26 调整）

| 原编号 | 新编号 | 说明 |
|---|---|---|
| exp1 task1–5 | exp1-task1–5 | 不变 |
| exp4 task13（模拟器） | exp1-task6 | 挪到 exp1，因为基线就要用 |
| – | exp1-task7 | 新增：NBV episode 框架和基准标定 |
| – | **exp2_baseline** task1–7 | 新增：来自 `exp0.md` |
| exp2 task6（官方基线） | exp2-task3/4 和 exp3-task1 | 观测比例和步数的扫描并入 exp2；原始权重 vs EMA、编码采样 vs 均值并入 exp3-task1 |
| exp2 task7、8 | exp3-task1、2 | |
| exp3 task9–12 | exp4-task1–4 | B5 和 B6 即 exp2 的 S1 和 S2 |
| exp4 task14、15 | exp5-task1、2 | Random、Frontier、S1、S2、Oracle 已在 exp2，task1 只补 B3 和 B4 |
| exp5 task16–19 | exp6-task1–4 | |
| exp6 task20–22 | exp7-task1–3 | |
| exp7 task23–26 | exp8-task1–4 | |
| exp8 task27–29 | exp9-task1–3 | |
| exp9 task30 | exp10-task1 | |

### 第二次编号调整（2026-10-03，你的决定）

| 原来 | 现在 | 目录 |
|---|---|---|
| pre_vhd（原顶层一节） | **exp3-T1** pre_vhd（D1、D2） | `experiments/exp3_geometry_vhd/pre_vhd/` |
| vhd_v1（原顶层一节） | **exp3-T2** vhd_v1（预注册、冒烟测试） | `experiments/exp3_geometry_vhd/vhd_v1/` |
| – | **exp3-T3** 结构性退化诊断（新） | `experiments/exp3_geometry_vhd/vhd_v1_1/` |
| exp3_geometry_hypotheses、exp4_geometry_vhd、exp5_closed_loop_nbv（空目录，未开始） | 合并为 **exp3_geometry_vhd**（原设计作为背景保留在 exp3 一节末尾） | `experiments/exp3_geometry_vhd/` |
| exp6_consistency | exp4_consistency | `experiments/exp4_consistency/` |
| exp7_active_geometry | exp5_active_geometry | `experiments/exp5_active_geometry/` |
| exp8_semantics | exp6_semantics | `experiments/exp6_semantics/` |
| exp9_final_eval | exp7_final_eval | `experiments/exp7_final_eval/` |
| exp10_joint_semvoxflow | exp8_joint_semvoxflow | `experiments/exp8_joint_semvoxflow/` |

- **原始输出**：`work_dirs/pre_vhd/` 和 `work_dirs/vhd_v1/` 作为已完成实验的来源记录保留在原处、不移动；从 v1.1 开始的新结果放在 `work_dirs/exp3_geometry_vhd/…`。`work_dirs/` 下原来空的 exp3–exp10 目录按同样的对照改名。
- **旧文字中的编号**：2026-10-03 之前写的文字（exp1、exp2、exp_vis 各节，exp3-T1/T2，各 `results.md`、`exp1.md`、`exp2.md`）保留原编号，按本表理解，例如"留到 exp9"即现在的 exp7。尚未开始的计划部分已改为新编号。
- **整理前的 checkpoint**：`work_dirs/checkpoints/2026-10-03_pre_exp3_restructure/`（已跟踪文件的 diff、全部未跟踪文件的快照、git 状态）。`experiments/` 从未被 git 跟踪，所以用普通 `mv`（`git mv` 只适用于已跟踪文件）；`seen2scene/` 没有任何修改。

### 第三次路线调整（2026-10-08，你的规划）

| 你的路线 | 对应的实验 | 说明 |
|---|---|---|
| EXP3：3D-FRONT 基线 / VHD 诊断 ✅ | exp2_baseline、exp3_geometry_vhd | 已完成：S2 > S1 ≫ VHD；VHD-v1/v1.1 降为分析项 |
| **EXP3b：MACARONS++ Transfer Baseline** | **exp3b_macaronspp_transfer**（新） | R 协议：跨数据域迁移（T1–T4）；M 协议：MAGICIAN 原生基准（T5–T7） |
| EXP4：Consistency | exp4_consistency | 编号不变；目标从 VHD 改为 S2，开始前复述时重写通过标准 |
| EXP5：Active Geometry 训练 | exp5_active_geometry | 编号不变；即以后的 ActiveGeometry-S2，评测改为 S2 闭环 + 两个数据域 |
| EXP6：两个数据域重新评测 | exp7_final_eval | 编号不变；扩展为 3D-FRONT test200 + MACARONS++（R 协议 held-out、M 协议） |
| Final：与 MAGICIAN 比较 | exp7_final_eval 中 M 协议的表 | 使用 EXP3b 冻结的测试循环 |
| （不在新路线中） | exp6_semantics | 暂不动，等你决定是否保留 |
| （可选） | exp8_joint_semvoxflow | 不变 |

- exp4–exp8 不重新编号，避免第三次改号；你路线中的 "EXP6" 就是这里的 exp7_final_eval。

### 修改记录（实验内容的变化，细节见各实验的说明文件）

| 日期 | 实验 | 修改 |
|---|---|---|
| 2026-09-26 | 全部 | 调整编号；加入 exp2_baseline；task 在实验内部编号 |
| 2026-09-26 | exp1-task1 | 数据单独存放在 `/bsushare`，用 tmux 下载 |
| 2026-09-26 | exp1-task2 | 网格指标的采样改为 30 万对 200 万；部分输入直接从 VDB 裁剪 |
| 2026-09-26 | exp1-task4 | 改用自写的光线步进；判据改为 3D 距离加非边缘深度；patch 外遮挡采用方案 A（区域主动检视） |
| 2026-09-27 | exp1-task5 | 可复现性标准：数值逐位相同，网格 ≤ 1×10⁻⁶ m |
| 2026-09-27 | exp1-task1/2 | patch 原点对齐到体素网格，重新导出 GT |
| 2026-09-27 | exp1-task7 | 候选合法性规则；冻结 N=64、T=10，主指标 AUC@10 |
| 2026-09-27 | exp1-task1 | ff2e6ead 按原流程重选；dev/test patch 列表冻结 |
| 2026-09-27 | exp2 | 同步 exp1 的结果：VAE 检查的参照改为 F@1 0.995；task3 需要处理 p_0.01 的空输入；N 和 T 已冻结 |
| 2026-09-27 | exp2 | 复述后定稿：task1 的 CD 范围只作预警；task2 同时报告绝对和相对下降；task3 空输入分开报告失败率；task4 加 G_cov 排序一致性检查；task5 世界模型接口严格等价、无表面时先尝试生成；S2 与第一版 VHD 都用 K=4；冻结 Frontier 定义；规划器用 dev100；Holm 只用于 AUC@10 主比较；task2/3/4 完成后停下汇报 |
| 2026-09-28 | exp2-task1 | 可复现性检查发现：S0-O 的世界会随同一进程内先处理了哪些 patch 而变（CLIP 物体名嵌入缓存）。修复：`--task.world-seeding` 开启时，每个 patch 开始前清空这个缓存（官方默认行为不变）。修复后世界只取决于 (patch, 种子)。S0-F 不受影响。dev50 用修复后的代码重跑 |
| 2026-09-28 | exp2-task1 | 倒序测试又发现第二个来源：默认 GPU 算法中的浮点原子加法，求和顺序随显存布局而变。修复：`--task.world-seeding` 开启时，同时启用 PyTorch 确定性模式和固定的 cuBLAS 工作区（约慢 4%，网格也能逐位复现）。dev50 用最终管线重跑（`dev50_v3`） |
| 2026-09-28 | exp2 | 你的决定：task1 通过；确定性 world seeding 用于所有公平比较的生成实验；task3/4 默认不出图；每张卡一个生成进程 |
| 2026-09-28 | exp2-task2/3 | task2 通过；诊断 A/B/C 显示提升主要来自去掉 CFG。task3：空输入到不了模型，记为推理失败；新增可选的跳过渲染和失败记录参数 |
| 2026-09-29 | exp2-task4 | task3、task4 通过；工作点固定为 S=10；默认配置冻结为 S0-F + 无布局 + guidance 1.0 + S=10 + 确定性 world seeding |
| 2026-09-29 | exp_vis | 新增 EXP_VIS（你的 vis.md）：S2 在 48/100 暂停，先做 VIS1-A；复述后定稿（S2 第 1 步状态、预先登记的 4 个场景、与 S2 日志逐项核对）；你补充：保留原始 G_cov 和 A3 的连续分歧统计 |
| 2026-09-29 | exp_vis | VIS1-A 完成：A3 的 view_low 在 4 个场景都退化（光线看不到任何表面），补充 view_low_seen（≥20% 光线命中表面），预先登记的结果原样保留，采用哪个待你决定 |
| 2026-09-29 | exp_vis | 你的修改：图 1 和 overview 的灰色改为固定的观测证据（命中点），所有面板逐点相同（硬性检查 4/4 通过）；原画法保留为诊断图；图 2 的定义和数值不变 |
| 2026-09-29 | exp_vis | 你的决定：VIS1 改为 task1 vis_xworld（A 3D-FRONT、B MACARONS++），新增 task2 vis_step（采样轨迹，A 以 x̂₀ 预测为主行）；3D-FRONT 结果移到 `work_dirs/vis/vis_xworld/front3d/`，代码分到 `vis_xworld/`、`vis_step/` |
| 2026-09-29 | exp_vis | 新增 task3 vis_stepcov（采样步数 S10↔S50 的影响随覆盖率的变化）：L0 单视角（构造）+ p=0.01/0.05/0.1/0.2，12 个场景做曲线（4 个做图），加入损失 R 和 step/seed 比值；你的调整已写入 |
| 2026-09-30 | exp_vis | task1-B 定稿：只做适配器扩展、3D-FRONT 回归门槛、场景 sestino_museum/barts/pantheon（colosseum 为备选）、运行时尺度不缩放、patch/观测/候选都从位姿网格按规则冻结、叫 full-scan reference 而不叫 GT；A3 主结果改用 view_low_seen（task1-A 同样按此解读） |
| 2026-09-30 | exp_vis | task1-B 完成：两道门槛通过（适配器逐位等价、3D-FRONT 回归无变化）；选择规则运行前写定；3 个场景推理成功、硬性检查和复现全部通过；补全带 3D-FRONT 地面和房间先验，Top-1 全部一致；A3 的 20% 门槛在 MACARONS++ 上几乎失效（照实报告，未改规则） |
| 2026-09-30 | exp_vis | task1-B 补充带纹理的上下文图（整个场景、周围、patch 本身）；据此改正 sestino（石碑）和 barts（房间角落的大物体）的内容描述，数值不变 |
| 2026-09-30 | exp_vis | 你的决定：task1-B（k=1）保留为不缩放的跨域基线；新增 task1-B2 尺度变体 k=2、3（配对设计：只改全局尺度，观测、候选、种子都不变），加入 eiffel（事先指定的塔腿位置），另记观测密度 |
| 2026-09-30 | exp_vis | task1-B2 完成：k=2、3 下观测变连续，结构占比和规划分歧接近 3D-FRONT，与 GT 的一致程度没有系统性变好；eiffel 在 k=1 输入无效（8/67 个合格视角），按规则停止 |
| 2026-10-01 | exp2-task6/7 | task6 通过（你的确认）；task7 改为整理已有 S1/S2 成本，并在 8 个固定状态上补测 K=1/K=4 的延迟和显存（allocated、reserved、进程显存分开）；S=50 引用 task4，K=8 不再单测 |
| 2026-10-01 | exp2 之后 | 你的决定：task7 后先做 task6 事后诊断（初始分歧、离群抑制），再冻结 VHD-v1（K=4、命中/未命中 + 首次命中深度分歧、不训练、不按 GT 调权），先冒烟测试再跑 dev100；核心比较 VHD vs S2 |
| 2026-10-01 | exp2-task7 | 完成：K=1/K=4 延迟、显存三个量分开；加了 1 个 episode 的重放诊断，确认 39–43 GB 是缓存；更正 task5 的 GPU 时（S1 单独 6.3，全部 8.4）；exp2 结束，写 `exp2.md` |
| 2026-10-01 | exp2 → pre_vhd | 你的决定：task7 通过，exp2 结束（不加 task8）；VHD 复用 S2 的 K=4 个世界，暂不优化生成器；新增 pre_vhd（D1 初始分歧、D2 离群偏好抑制），只做描述性分析 |
| 2026-10-01 | pre_vhd | 复述确认；你的补充：D1 三组分层、D2 第 2 步后称 k=0 counterfactual、整数体素数判断、假设执行不可变性硬性检查、random-world-following advantage（等权不去重）、解释边界和两张汇总表 |
| 2026-10-01 | pre_vhd | D1、D2 完成：S2 的一步优势与平均掉单个世界的离群偏好一致，但这种抑制不加区分（3-1 中 37% 离群偏好更好）；第 1 步分歧大小与收益无可见关系 |
| 2026-10-01 | pre_vhd → vhd_v1 | 你的决定：D1/D2 通过，不再做事后挖掘；VHD 不用全局分歧、Top-1 个数、多数强度，不用多数投票，不给少数世界降权；起草 VHD-v1 预注册（与 S2 只差规划分数），暂不恢复原 exp4 的开环研究 |
| 2026-10-02 | vhd_v1 | 预注册冻结为 v1：主掩码 B（新观测，复用 G_cov 规则），A 记为 VHD_raw 诊断；分母固定 16384；全 0 退回 Random；冒烟测试前 16 个 episode，只查实现 |
| 2026-10-03 | vhd_v1 | 冒烟测试完成：实现检查全部通过；发现相机被生成几何包住的退化视角（≥ 21/158 个决策），等你决定是否升级 v1.1 |
| 2026-10-03 | exp3 | 你的决定：VHD-v1 实现检查 PASS，但因结构性退化不进入 dev100；新建 EXP3（合并原 exp3/4/5、pre_vhd、vhd_v1），后续实验改为 exp4–exp8，VHD 新版本用 vhd_v1_1/vhd_v2… 子目录；定义 T3 结构性退化诊断（4 种门控，不跑 dev100） |
| 2026-10-03 | exp3-T3 | 完成：观测自由门控几乎无效；0.1 m 候选可行性门控把病态选择从 30/158 降到 15/158，正常状态全部保留，无新退化；剩下的是 0.1–0.3 m 的近场生成表面 |
| 2026-10-03 | exp3-T3/T4 | 你的决定：T3 通过；VHD-v1.1 只加 0.1 m 候选可行性门（固定除以 6，其余与 v1 相同），不用并集，0.3 m 只作诊断；起草 T4 预注册；核对发现原探测渲染器有 0.01 m 近裁剪面，T3 没有，冻结前待定 |
| 2026-10-03 | exp3-T4 | 预注册冻结为 VHD-v1.1：近裁剪面选 A（z_near = 0.01 m 按 z 深度），B 只作 T3 复现诊断；开始前 16 个 episode 的冒烟测试 |
| 2026-10-03 | exp3-T4 | 冒烟测试完成：实现检查全部通过；全 0 退回 0；剩下的 19/158 个诊断病态都是 0.1–0.3 m 近场情况；没有新的方法级退化；未进入 dev100 |
| 2026-10-03 | exp3-T5 | 你的决定：v1.1 冒烟测试 PASS；VHD-v1 和 VHD-v1.1 冻结后跑 dev100；主比较 v1.1 − S2 的 AUC@10；不得事后修改 |
| 2026-10-04 | exp3-T5 | dev100 完成：VHD-v1.1 − S2 = −0.183 [−0.202, −0.163]（98/100 更差），VHD-v1.1 ≈ Random；v1.1 − v1 = +0.015；VHD 分数与实际增量弱负相关。如实记录，不修改 v1.1 |
| 2026-09-30 | exp_vis | eiffel 补跑 k=2、3（观测和候选在 k=2 抽取，k=3 复用），单独出图 `compare_eiffel_rows.png`；另算了只用 1/2 个观测视角时的覆盖率，供决定是否做低观测档 |
| 2026-10-08 | exp3 | 你的决定：接受 T5 的结论；VHD-v1/v1.1 降为分析项，不再为它做新版本；主线改为多世界期望覆盖（S2）+ 更好的世界模型 |
| 2026-10-08 | 路线 | 你的新路线：插入 EXP3b（MACARONS++ 迁移基线），之后是 EXP4 一致性 → EXP5 Active Geometry → 两个数据域重新评测（exp7）→ 与 MAGICIAN 比较；exp4–exp8 不改编号 |
| 2026-10-08 | exp3b | 设计稿：R 协议（区域检视）做 transfer gate，M 协议（MAGICIAN 全场景协议）做基准表；MAGICIAN 协议的核对材料写入 `exp3b_macaronspp_transfer/magician_protocol_facts.md` |
| 2026-10-08 | exp3b | 你的规范化要求：统一命名为 `exp3b_macaronspp_transfer`（EXP3b — MACARONS++ Transfer Baseline）；明确 R 协议（our pipeline + MACARONS++）和 M 协议（MAGICIAN pipeline + 被评测的方法，不等于 MAGICIAN 规划器）；加入数据链接（真实目录是 `MAGICIAN/data/Macarons++`，未改名）；RQ1–RQ3；T3 阈值取自冻结的 3D-FRONT 参考；T4 只排除灾难性失败；T5 失败即 STOP、不事后搜索；T7 分 All-14 / Unseen-6；**每个 task 单独审批**；README 中的 task 表合并为一张，细节移到 `preregistration.md`（草稿，待你确认） |
| 2026-10-08 | exp3b | 你的确认：preregistration 冻结为 v1，加入三个冻结点：T1 必须核清 M 协议的深度信息流（核清之前不冻结传感器信息假设）；T4a/T4b/T4c 分别审批；T5 的 E-M2 后端只按复现标准选（都达标用官方滑窗，都不达标就 STOP）。另将数据路径集中到 `exp3b_paths.py`（`MACARONS_PLUSPLUS_ROOT`） |
| 2026-10-08 | exp3b-T1 | 完成，判定 NEED-DISCUSSION：10% 点同时用于评测和规划器的表面地图（不是只用于评测），完整深度只在 proxy 点处用于 carving 和可见性 → v0 的全分辨率设计不保留，对等方案等你选（推荐 P1 分层对等）；MAGICIAN 工作区有未提交的修改（HEAD 9521d69 + diff），固定用哪个状态等你决定。E-M2 窗口表、报告方式冻结 |
| 2026-10-10 | exp3b | 你的决定：先做 T4c 的 VHD 迁移诊断，暂不进入 M 协议；T4x 归档（E2 4b7ea6be…）；固定三层区分（World Generation Quality / World Utility / Planner Effectiveness）；Table A Full-40 已生成；T4c 审计：20 个 episode = p0–p3，VHD 两版的校验和一致；发现 VHD-v1.1 `planner.py` 自己绑定了 `camera_rays` 和 `RandomPlanner`，作用域管不到（待确认）；T4c 定位建议为 pre-specified descriptive、post-hoc 执行；Table B 只定格式。未运行 |
| 2026-10-10 | exp3b | T4c VHD-20 完成（你批准；`t4c_run.py`；冻结 7ba25d0b…）：单元检查全部通过（零偏移逐位等价、生产模式射线配对、强制 fallback、作用域）；41 条轨迹全部 exit 0，16 项完整性检查全部通过，确定性重跑逐位相同 → 执行 VALID。Table B（七方法，同样 20 个 episode）：VHD-v1.1 0.853 ≈ S2 0.864 ≈ Frontier 0.856 ≫ Random 0.680；VHD-v1.1 − VHD-v1 +0.035（5/15/0）；fallback 0；冻结分类规则不满足（不是"跨数据域一致的失败"），与 3D-FRONT 的方向不同，但受饱和等条件限制，只作描述；归档 b1bd7c55…；STOP，T5 和 M 协议未开始 |
| 2026-10-10 | exp3b | 你的确认：T4c 结论固定（不归为"跨数据域一致的失败"，也不能据此声称 VHD 有效）；三层解读固定；EXP3b-R 阶段暂时关闭（R_STAGE_CLOSED_SHA256SUMS 079be972…）。T5 预注册审计已提交（16 项，逐项标注【冻结】/【事实】/【待确认】）。核对出的事实：官方滑窗的步长是 204、最后一个窗口贴边，gate 场景的窗口数仍是 34 个，与 em2 表一致；官方加载器的 max_scene_volume=70 m³ 会跳过 colosseum、sestino 和 pantheon；t=0 时的空窗口没有定义；MAGICIAN 在 beam 第一层用 GT 做碰撞过滤，与冻结的"所有方法共用同一过滤"等价。估计约 23–30 GPU 时。未运行 |
| 2026-10-10 | exp3b | 你的决定：原则上批准 T5 作为单独授权的 feasibility branch，只批准文档工作和只读代码审查。T5 修正案 v1.3 草稿（A–H）已写入 preregistration §7：窗口差异审计（倍数 8、步长 200；gate 窗口数仍是 34；建议 B1）；P1 自由空间语义（按 MAGICIAN 实际代码，F-a / F-b 待选）；空窗口的四层定义；多窗口评分、去重、并列和退回规则；H1–H4 和全部 gate 的准确规则；受限的实现验证 V0–V3（不算任何 gate 指标）；历史轨迹的使用与白名单。更正上一份审计：步长不是 204；max_scene_volume 的说法过强；空窗口不报错；Random walk 的 bug 比 T1 记录的严重。未运行 |
| 2026-10-10 | exp3b | 你确认 v1.3（B1、F-a、空窗口的四层定义、单一网格 + 来源标签、统计口径），选择沿用 MAGICIAN 反投影约定，并批准 V0–V3。新代码：m_common / m_magician_replay / m_lmdb_prefix / m_p1 / m_world / m_score / t5_validate。冻结前修正 5 处，execution-repair 2 次（导入路径；V2.4 重跑 OOM，第一次尝试归档到 attempt1_v2_oom）。第二次尝试（冻结 4479789b…）V0–V3 全部通过：回放逐位一致、像素约定、P1 证据隔离、官方多窗口推理（空窗口不报错，colosseum 每个世界约 256 s / 27 GB）、新进程重跑逐位相同、重叠去重、来源标签、完整链路。V1.6 发现 2 个整场景空输入状态（eiffel/2/t0、colosseum/2/t0）。STOP |
| 2026-10-10 | exp3b | 你调整了 M 协议的定位：核心问题是在 MAGICIAN 原生约束下，世界假设能否改善合法动作的选择。V0–V3 判 VALID；原 T5 PASS 不可达。完成两个空输入状态的只读根因审计（colosseum/2/t0：没有 z < 70 的命中；eiffel/2/t0：487 个样本全部在场景框外；MAGICIAN 的覆盖率同样为 0）。提交了 T5-Preflight 方案（g 评估器、历史一致性、评价量、H1–H4 和 INVALID amendment、体素边界）、T5-Diagnostic 设计（73 + 2 个状态，Q1–Q6 / U1–U6，约 12 GPU 时）和 T7 公平性复核。均未运行 |
| 2026-10-10 | exp3b | T5-Preflight B–F（你批准）：新代码 m_gain_eval / m_h4_neighbors / m_metrics / t5_preflight，m_score 加了 dtype 选项；冻结 be7aa00d…。结果：B1–B3（恢复逐位精确、与动作顺序无关、确定）、C1（75 个执行动作的 g 逐位等于覆盖差，含 2 个负值和 5 个零值）、C2、D1–D5、E1–E4（H2 平均 \|Δ\| 0.0017）、F1–F3（float32 = float64）全部通过；B4 的辅助时钟条件 as-run 不通过（时钟偏差约 2.6 s，三个实质隔离条件都满足）→ STOP。归档 ec1e5d97…；约 0.5 GPU 时 |
| 2026-10-10 | exp3b | B4 execution-check repair（你批准）：新文件 t5_preflight_b4fix.py（原 t5_preflight.py 未改），只用归档产物。R1–R8 全部通过：证明时间基准不同（文件服务器时钟比本地快 2.56 s）、同一时钟下分数早于 g 21 s、哈希不变、静态 / 白名单 / 1319 个服务记录 / 动态追踪都没有发现泄漏、5 个负例全部检出、没有运行 MAGICIAN 也没有生成世界。最终：T5-Preflight PASS after separately authorized execution-check repair。STOP |
| 2026-10-10 | exp3b | 你正式接受 T5-Preflight PASS after separately authorized execution-check repair，Preflight 阶段结束。T5-Diagnostic 最终设计冻结审查完成（§7 冻结协议）：状态清单和可用性分类、冻结 E-M2 的配置和种子、共用合法列表、Frontier-M 定义（exp2，观测权限相同，裁剪到表示域盒子，全零取最小下标，约 60 行代码）、Q1–Q5 和利用价值 / 单步决策指标、分母表、分组重采样、go/no-go 参考、运行顺序和防泄漏、有效性检查、STOP、预算和表格结构。未运行 |
| 2026-10-10 | exp3b | 你正式批准 T5-Diagnostic（75 个固定状态的下一步动作价值探索性评估）。三项补充约束写入 §7。新代码：m_frontier / m_quality / t5_diagnostic / t5_diagnostic_pretest。冻结前修改两处（surface_voxels 阈值与 vis_common 一致；P1 测试数据），均已记录在 §12。正式 pretest P1–P8 通过，冻结 d182e411，正式运行已启动 |
| 2026-10-11 | exp3b | T5-Diagnostic 运行完成并 STOP。五部分表格见 results.md 与 `diagnostic/evalside/analysis/tables.md`。完整性全部 PASS。探索性结论：冻结的世界假设在单步选点上优于 Random（4/5 个场景），但没有显示多世界（S2 vs S1）优势，相对 Frontier 的优势在 CI 内不确定；未观测结构质量较低 |
| 2026-10-11 | exp3b | 你调整后续计划为 Baseline Benchmark v2（Baseline Establishment，探索性，不覆盖原 T6/T7）。完成 T6 运行前只读审计，提交 v2 amendment（T6 = gate-5 × 5 × 10 步闭环，8 个方法加参考行；D1–D4 待定）。未运行 |
| 2026-10-11 | exp3b | 你正式批准 T6 Native 10-Step Closed-Loop Baseline（D1 映射；D2 literal transplant；D3 共享 10%；D4 权重符号链接）。实现 T6 闭环代码，pretest P1–P10 通过，冻结 ac45b6f8，正式运行启动 |
| 2026-10-11 | exp3b | T6 Native 10-Step Closed-Loop Baseline 完成并 STOP：450 个 episode，完整性全部 PASS；Baseline Table 见 results.md 和 `task6_baseline_v2/analysis/t6_tables.md`；归档 d654fc77 |
| 2026-10-11 | exp3b | 你接受 T6 PASS（execution valid）。提交 T7a 方案：非结果场景选择规则、100 步协议、R=10 刷新、代码审计、三档预算、smoke 计划。未运行 |
| 2026-10-11 | exp3b | 你正式批准 T7a 方案 B。实现 T7 代码（R=10 刷新、原生 20 步重填、预算暂停、全卡显存采样），pretest H1–H4 通过，冻结 c685164c，正式运行启动 |
| 2026-10-12 | exp3b | T7a 第 1 轮 STOP：S2 colosseum/4 在 t=60 的刷新中 OOM（fvdb）；全卡峰值 45.1–45.5 GB 超门槛；as-run 全部保留；没有修复或续跑，等你决定 |
| 2026-10-12 | exp3b | 你批准方案 1（执行环境修正：colosseum 生成式任务双卡、expandable_segments、生成时 P1 移到 CPU）并要求完整重跑。第 2 轮 pretest H1–H5 通过（H5 复现第 1 轮最重刷新，世界哈希逐位相同），冻结 6af1a691，17:29 启动 |
| 2026-10-13 | exp3b | T7a 第 2 轮被基础设施中断：交互作业 3330776 于 02:34 被 SIGTERM 终止；已完成 10 个 colosseum S1/S2 episode，4 个 S2 pantheon 中断 |
| 2026-10-14 | exp3b | 你决定改用 sbatch 续跑（保留 10 个、重跑中断的 4 个、补跑 181 个）。续跑外壳 t7a_resume*（冻结代码不改）；环境测试与 T6 逐位相同；185 个数组任务全部 exit 0 并逐任务核验；finish（冻结 post + analyze）exit 0，完整性 12 项全部 PASS。AUC@100：MAGICIAN 0.697 > Frontier 0.631 ≈ MACARONS 0.628 > S1 0.530 ≈ S2 0.520 > Random 0.366（Oracle 0.768）；S2 − S1 −0.010 [−0.045, +0.025]。episode 63.3 + post 2.4 GPU 时；归档 c637a6d3。T7a 完成（探索性），STOP；T7b 未批准 |
| 2026-10-14 | exp3b | 你的决定：T7a = PASS（Execution Valid，Exploratory）；结论只针对当前 3 个按成本选取的场景；T7b 不运行，也不作为 EXP4 的前置条件；冻结文件和归档不变。只读补充（不用 GPU）：同场景前 10 步 195 个 episode 与 T6 逐位相同，排序变化主要出现在第 11–100 步；补记 Seen2Scene 权重的 sha256（`readonly_same_scene_first10/`） |
| 2026-10-14 | exp4 | 你正式建立新的 EXP4（Learning Reliable Generative Multi-World Beliefs for Active 3D Mapping）：Task 1 多世界生成器，Task 2 信念更新 C0/C1/C2（含 Fresh Regeneration 对照，R=1 / R=10），Task 3 多世界 NBV 规划器（P1 / P2）。输入只用 Depth + Pose。旧 exp4_consistency / exp5_active_geometry 的文字保留作背景，与新 EXP4 的对应关系见 `exp4_multiworld_belief/exp4_plan_proposal.md`。本轮只做 EXP4-P0 设计审查（只读，不训练） |
| 2026-10-14 | exp4 | 你新增 Task 0（Training Dataset Construction & Validation），放在 Task 1 之前；划分原则上采用 O3，sofia_church 改为 Reserved。T0-0 只做设计和 CPU 只读审计：15 个场景的 OBJ 哈希两两不同；Train-6 共 69 个窗口，2 个无表面；表面内容 dunnottar 41%、pisa 33%；EXP3b held-out 修正案草案 A-EXP4-1 未生效。GPU pilot 未批准 |
| 2026-10-14 | exp4 | 你接受 T0-0 = PASS（Design and CPU Audit Completed）；批准 Dataset Protocol v1.0（Train-6 / Dev-5 / Test-3 / Reserved = sofia_church）和修正案 A-EXP4-1（追加式）。T0-1A：清单 v1 冻结（e3cf35d7…）；Π* 改用 MAGICIAN 精确规则（145,250 个位姿，比启发式多出 13 个可达位置）；EMPTY-only（G / F / U）方案；token mask 核对；采样注水法；Gate v1（【硬】/【目标】分开）；smoke 计划在现有 3377066 四卡分配内，一次一张卡。没有运行 GPU |
| 2026-10-22 | exp5 | 你提出新 EXP5（任务书 `EXP4_NextStage_Task0_Task1_V2_Research_Plan.md`），目录 `exp5_future_view/`，删除空的 `exp5_active_geometry/`。Task 0（MP3D 低覆盖诊断）因没有 MP3D 授权 BLOCKED，规划草案保留；先做 Task 1（V2）的详细规划审查，不跑代码。git 提交格式改为 `[expN][类型] 描述`，EXP4 belief 代码的 3 个提交连同此前 15 个提交已 push |
| 2026-10-09 | exp3b-T4x-E2 | **探索性**：其余 35 个 episode × S1/S2（tmux，17:08–19:49），E1 已归档未重跑。40 个 episode 执行 VALID（完整性全部通过，确定性 2/2，0 次 fallback）。AUC@10 S1 0.850、S2 0.852、Frontier 0.850、Random 0.633、Oracle 0.908；S2 − S1 +0.002 [−0.006, +0.009]（17/7/16）；S2 − Random +0.219（37/0/3）；S2 − Frontier +0.002（20/0/20）。只读诊断：选中视角预测新表面平均 46–50% 在补出的地面层（实际约 4%）；192/400 个 S2 步 4 个世界 Top-1 一致，这些步的地面层占比更高（共同偏差）；sestino 的若干 episode 和 colosseum/p1 第 1 步因地面先验选错视角；但所有步合在一起看时，与收益损失的关联在很大程度上被饱和混杂。STOP |
| 2026-10-09 | exp3b-T4x-E1 | **探索性**（不是 T4b）：冻结（7c62ede1）→ fallback adapter 单元检查全部通过 → 5 个 p0 × S1/S2 加 1 次重跑，每条在独立进程中运行。执行 VALID：完整性全部通过，与 T3 第 1 步逐位一致 10/10，确定性一致，0 次 fallback。AUC@10 均值 S1 0.838、S2 0.834、Frontier 0.827、Random 0.660、Oracle 0.920；S2 − S1 −0.003（2/0/3），S2 − Random +0.174（5/5），S2 − Frontier +0.007（1/0/4）。地面先验：sestino 第 1 步 S1/S2 选中的视角预测新表面 99.98% 在补出的地面层，实际增量只有 0.021（Oracle 0.811）。STOP |
| 2026-10-09 | exp3b-T4a | 你确认 T4a = FAIL（有效的科学结果）：保留全部 gate 结果；解释为 early saturation 和动态范围压缩，但 R 协议并非完全无法区分规划器；T4b confirmatory BLOCKED；方向原则上选 B，只提 post-hoc exploratory 的 S1/S2 方案、不运行；暂不进入 M 协议（以后若转入，须记录 T4b 前提未满足，并作为独立的 feasibility investigation）。T4a 输出已归档（sha256，只读） |
| 2026-10-09 | exp3b-T4a | 完成，**FAIL**（gate 3）：严格按冻结预注册和 §12 的执行细节运行；120 个主 episode 加 6 次重跑，完整性检查全部通过，确定性 6/6。AUC@10：Random 0.633、Frontier 0.850、Oracle 0.908（gate 1 PASS）；Cov@10 Oracle − Random +0.205（gate 2 PASS）；**Oracle Cov@3 0.925 > 0.90（gate 3 FAIL，饱和；28/40 个 episode ≥ 0.90，不只是高 c0 的 episode）**；gate 5：6 个 episode 的 AUC@10 高于 greedy one-step Oracle（标记）。STOP，T4b 未运行 |
| 2026-10-09 | exp3b-T3 | 完成，**PASS**：严格按冻结预注册（阈值、种子、场景、patch 都不变；v1.2 射线语义）。40 个 v1.2 参考已建；5 个 p0 的 W1–W4 推理全部成功；(b) 观测复现中位 1.000 / 0.939 / 0.955 / 0.902 / 0.868（≥ 0.786）；(c) 已观测空闲空间表面占比中位 0.0000 / 0.0105 / 0.0006 / 0.0147 / 0.0005（≤ 0.0786）；(d) 结构占比 0.526 / 0.594 / 0.438 / 0.675 / 0.422（≥ 0.2，但低于 3D-FRONT 的 0.68–0.87）；(e) 初始覆盖中位 0.347（≤ 0.508；6/40 ≥ 0.8）。只描述：pantheon/eiffel/colosseum 的世界在底部第 2–4 层铺满地面而参考几何为 0（3D-FRONT 地面先验）；sestino 争议率 0.107；F（τ1/τ3）0.13–0.49 / 0.39–0.62。作图：空观测网格的显示处理 + 本地拼 overview（A3 在 pantheon/colosseum 无合格候选）。STOP，T4a 未开始 |
| 2026-10-09 | exp3b | 你确认 R5，关闭历史射线敏感性问题（R4 gate FAIL 与 R5 PASS 都保留；表述限定为“fixed-trajectory replay 下配对结论稳健”，不扩大为“规划对射线约定不敏感”）；不做 full replanning sensitivity；R 协议措辞澄清（冻结部分 + 目标渲染器 sample-0 adapter；不声称逐位相同）；批准 T3（严格按冻结预注册，完成后 STOP） |
| 2026-10-09 | exp3b-T2 | R5 完成：冻结（13 个代码 + 701 个数据文件；第一次冻结在导入阶段崩溃、尚无结果，修一行后重新冻结）→ 4 个 GPU 分片重放 → tmux 链自动 analyze。像素中心重放逐位复现 7700/7700 个日志状态和 exp2/exp3 冻结表；各方法 ΔAUC@10 = +0.0002 至 +0.0009；S2 − S1 = +0.01589 [+0.00743, +0.02460] → +0.01601 [+0.00754, +0.02473]，ΔD = +0.00012 → **主门槛 PASS**；次要对比符号和 CI 都保持（\|ΔD\| ≤ 0.00033）；无排名反转。R4 判定（FAIL）保留。STOP，T3 未开始 |
| 2026-10-09 | exp3b-T2 | 你的决定：R4 判定 FAIL 永久保留；sample-0 只是 counterfactual sensitivity test（官方数据与像素中心一致，不把旧管线定义为错误）；批准 R5 — Fixed-Trajectory Paired Sensitivity：7 个方法的冻结 dev100 轨迹固定重放（不重新规划、不生成世界、不改日志）；主门槛 S2 − S1 AUC@10（sample-0 下差 > 0、CI 下界 > 0、\|ΔD\| ≤ 0.004）；次要对比只报告；运行前冻结代码和脚本的 sha256；完成后 STOP，T3 未开始 |
| 2026-10-09 | exp3b-T2 | 你确认 T2 = PASS after preregistered repair 及两处实现细节；批准 R4/R4-b（只读）→ R4：5 个 dev patch 的冻结状态逐位重建；Δcov_10 = +0.0025 / +0.0014 / +0.00003 / −0.0032 / +0.0004 → max 0.0032 **不满足 < 0.0011**；BAND \|Δtsdf\| 中位 0.054–0.074 体素（满足）；官方 p_0.1 VDB：像素中心一致率 ≥ 0.99992，sample-0 0.991–0.998（官方数据就是像素中心解释）；R4-b：光线翻转 0.131%，候选层面 46.8%，被选中候选 0.104% / 32%。**STOP / NEED-DISCUSSION**；EXP1–EXP3 未改动、未重跑 |
| 2026-10-09 | exp3b-T2 | 你批准 R2+R3 → R2 冻结 26 个代码文件（code_frozen_v12，SHA256SUMS adaecdb4…）→ R3 正式重跑全部通过：C1-geometry 5/5（G2 ≥ 0.999973，G3 ≥ 0.999994），C2a/C2b 5/5，C4 5/5（命中点 1.0000；MC 0.9921–0.9999），C6 PASS（参考用 sample-0 融合重建到 references_v12/），C5 PASS；R3 与 R1 逐字段相同。**T2 as-run = FAIL（保留）；T2-repair-v1.2 = PASS；T2 ✅ PASS after preregistered repair**。R4/R4-b、T3 未开始 |
| 2026-10-08 | exp3b-T2 | T2-repair R0 PASS（零偏移逐位恒等：函数、规划器 S1/S2/VHD/Frontier/Oracle/遮挡、episode；无泄漏，同进程 C5 逐位相同；pyrender 探针 ≤0.0015 px，PyTorch3D 1×/4× ≤1e-4 px）→ R1 pilot PASS（5 场景 C1-geometry：G2 ≥0.99997、G3 ≥0.99999；C2a/C2b 全过；C4 eiffel 命中 1.0000 / MC 0.9921，colosseum 1.0000 / 0.9995）→ STOP，as-run 输出 50/50 未变 |
| 2026-10-08 | exp3b-T2 | 你确认 R0 审计结论并冻结 v1.2：EXP3b-R 从观测到规划全链路统一 MSAA sample-0 射线语义（含区域外遮挡、S1/S2、VHD、Frontier）；选 I1（作用域内替换，泄漏即 STOP）；G1/G2/G3、C4 规则、R4（Δcov_10 < 0.0011 且 BAND \|Δtsdf\| 中位 < 0.1 体素）和 R4-b（只描述）冻结；gate_patches.json、pantheon p0、seen2scene/、MAGICIAN、阈值、EXP1–EXP3 不改。只批准 R0 → R1 → STOP |
| 2026-10-08 | exp3b-T2 | 你确认修正案总体方案（1–4 项原则上确认）→ R0 射线语义范围审计（只读）：区域外遮挡和 128² 规划器光线（S1/S2 cast、VHD、Frontier）依赖像素→视线语义，第 1 稿“不依赖”有误 → 统一规则扩展到这些模块（I1 作用域内替换 / I2 子类复制，待选）；加入 PyTorch3D 探针；C4 三角形→cell 归属与并列规则固定；R4 定义 Δcov_10（参考 0.011，门槛 0.0011）+ R4-b（只报告）。v1.2 最终草稿待你确认；因发现新语义问题，R0/R1 尚未执行 |
| 2026-10-08 | exp3b-T2 | 你的 repair 指示 → 起草 T2-repair 修正案（preregistration v1.2 草稿，待确认，未运行）：as-run FAIL 保留；射线语义走 F-b（传感器不变，深度→三维点一律用 MSAA sample-0 射线，新文件 `r_sensor_rays.py`，不改 `seen2scene/`）；C1 拆成 C1-geometry（G1 位姿、G2/G3 同一射线，MAGICIAN 4× 精确取样，G3 剔除 [0.5,1) 近距离语义像素）和 C1-renderer semantics（只报告）；C4 只统计 8 角点已观测的 cell；不采用双渲染拼接；pilot 不过即 STOP；pantheon p0 不重选；3D-FRONT 的 MSAA 记为历史诊断关注点，修复后只读诊断。protocol_card 补充 §9（PyTorch3D 按 znear/2=0.5 裁剪） |
| 2026-10-08 | exp3b-T2 | 完成，判定 FAIL → STOP：C1/C2 不过，原因是 pyrender 的 4×MSAA 深度取在子像素采样点、我们按像素中心反投影；C4 不过（命中点同一原因待证；MC 检查有 skimage mask 实现 bug）；MAGICIAN 的 PyTorch3D 不按 znear=1 裁剪；C3/C5/C6 PASS；40 个 gate patch 冻结。偏差 4 条写入 preregistration §12（待批准） |
| 2026-10-08 | exp3b-T1 | 你的决定：(a) 传感器信息冻结为 P1 分层对等（表面只来自同一组 10% 样本；完整深度只写 EMPTY，到“深度 − 截断”为止；披露 proxy 点与 TSDF 的离散化差异），P2 不采用，P3 永久排除；冻结共享观测采样；(b) MAGICIAN 用 `9521d69` 的干净 worktree，不带未提交修改，失败不切回。预注册升为 v1.1，T1 改判 PASS；允许进入 T2 |
| 2026-10-29 | V2 归档；exp6_v3 | 你指示把 V2（EXP5：Task 0–3、消融、E1 / E1.2 / E2）的代码、配置、报告归档并推送到 GitHub（`[V2][类型]` 提交）。`.gitignore` 改为只跟踪 `experiments/README.md`、`exp5_future_view/`、`exp6_v3/`；8 个超过 1 MB 的结果文件、`work_dirs/`、`data/` 仍留本地。新建 `exp6_v3`（V3：从零训练的 3D VAE 场景编码器 + 一处 Camera-Ray Cross-Attention，task0_vae / task1_xattn），见 `exp6_v3/README.md` |

## 通用设定（所有实验共用）

- **数据**：官方公开的 1000 个 3D-FRONT 留出场景，官方训练从未使用过。
  - 按场景 ID 排序后，用种子 2027 打乱，分为 `train_ours` 700 / `dev` 100 / `test` 200；
  - `dev50` 是 dev 的前 50 个场景。
- **评测 patch**：每个 dev/test 场景固定 1 个 256³ patch（2.8 m 见方）。
  - 条件：至少有 3 个非结构物体，并且 p_1.0 中已知体素占比 ≥ 50%；
  - 【2026-09-27】patch 原点已对齐到体素网格，dev/test 列表**冻结**，不因模型结果改动（exp1-task1）；
  - patch 列表写成 CSV，用 `--data.csv-path` 指定；
  - **同时必须传 `--data.scene-names`**，否则不在 split 里的场景会被悄悄换成随机样本。
- **任务设定（2026-09-26 你的决定）**：**区域主动检视**（regional active inspection）。目标 patch 之外的场景视为已经建好图、几何已知，规划器打分时用它遮挡光线（方案 A）；patch 内部是未知的。
  - 判断一条光线是否被遮挡，只看它在进入 patch 之前是否已经打到了几何体，所以只用到 patch 外的几何，不会泄漏 patch 内的内容。
  - 实现在 `PatchRaycaster.outside_block_tmax`。
- **目标的叫法**：p_1.0 称为"**高覆盖目标**"。它是全部官方视角的融合，不是完整的 watertight 场景。指标只在它的已知区域上计算。
- **不泄漏信息**：除 exp2-task1（官方复现）外，所有推理都使用 `--task.drop-bbox`，即不给 GT 布局，不做 CFG。
- **默认参数**：K 和 S 在 exp2 和 exp3 之后定稿，初始为 K=8、S=50（分析用）和 K=4、S=10（NBV 用）。
  - 种子：全局 2027，每个世界的种子为 `seed_world(scene, episode, step, k)`，由 exp1-task5 实现。
- **NBV episode**：
  - 目标区域是一个 patch；
  - 初始观测：该房间官方 p_0.01 相机中，视锥与 patch 相交的那些；如果没有，就用 1 个随机合法视角（固定种子）；
  - 每步从 N=64 个候选视角中选一个，共 T=10 步，直接瞬移。T 和 N 由 exp1-task7 标定，**2026-09-27 冻结为 N=64、T=10**。主指标 AUC@10，辅助指标 AUC@5 和第 10 步覆盖率。
  - 候选视角：高度 1.5–2.0 m，俯仰 ±60°，偏航任意，离障碍物 ≥ 0.1 m，视锥与 patch 相交。这与训练数据的相机分布一致。
    - 【你的决定，2026-09-27】合法性规则：加上 patch 外遮挡后，至少还有 1 条光线能进入 patch；不设可见比例的门槛。
- **统计**：
  - 以 patch 或 episode 为单位做配对 bootstrap（1 万次），报告 95% 置信区间；
  - 多重比较用 Holm 校正；
  - **超参只在 dev 上选，test 只在 exp9 跑一次**（包括 exp2 的基线）。

---

## exp1_infra：基础设施（W1）(done，2026-09-27)

**目的**：在跑任何模型实验之前，把所有实验共用的数据、评测、相机、光线投射、可复现性、模拟器和 NBV 框架搭好并**冻结**。
**详细说明**：每个 task 做了什么、改了哪些代码、结果如何，见 `experiments/exp1_infra/exp1.md`；完整数字见 `experiments/exp1_infra/results.md`。

### task1 固定划分和数据下载 (done)
- **内容**：按种子 2027 划分为 train_ours 700 / dev 100 / test 200，dev50 是 dev 的前 50 个。数据下载到 `/bsushare/yzhang-shared/data/seen2scene-FRONT-3D/`，共 567 GB，在 tmux 中进行。每个 dev/test 场景选 1 个评测 patch：至少 3 个物体，已知比例 ≥ 50%。
- **修改**：
  - 【2026-09-26】数据单独存放，用 tmux 下载；每个 task 完成后先汇报；
  - 【2026-09-27】patch 原点对齐到体素网格（由 task6 发现）；
  - 【2026-09-27】dev 的 ff2e6ead 对齐后只剩 2 个物体，按原流程重新选位置，旧条目存档；
  - 【2026-09-27】dev/test patch 列表**冻结**。

### task2 GT 导出和几何评测 (done)
- **内容**：导出 GT（稠密 TSDF、网格、部分输入）；只在已知区域上计算的指标 `metrics_known.py`；**冻结表面覆盖率的定义**。
- **修改**：
  - 网格指标改为 30 万查询点对 200 万参考点，原来的 F@1 0.979 只是噪声底；
  - 部分输入改为直接从 VDB 裁剪，因为官方 Dataset 会悄悄替换没有表面的输入（dev 在 p_0.01 下有 5 个 patch）；
  - 对齐后重新导出，自检全部通过。

### task3 相机 (done)
- **内容**：反解官方相机，得到完整的旋转矩阵；候选视角生成器（规则与官方一致，顺序固定）。
- **修改**：pyrender 的 DEPTH_ONLY 会剔除单面网格，改用 `FLAT | SKIP_CULL_FACES` 渲染。

### task4 光线投射 (done)
- **内容**：规划器在生成的世界里模拟"看到什么"；**冻结 G_cov**（新看到的表面体素数），G_free 作为辅助。
- **修改**：
  - fvdb 的 `ray_implicit_intersection` 在只含表面带的网格上结果错误，改用自己写的 GPU 光线步进；
  - 【你的决定】判据：命中点到表面的 3D 距离（99.99%）和排除深度边缘后的深度一致率（99.3%）；逐像素的 91.2% 只作诊断；
  - 【你的决定】patch 外遮挡采用方案 A，即"区域主动检视"。

### task5 可复现性和数据缓存 (done)
- **内容**：训练采样种子、按世界设种子 `seed_world`、记录 git 状态、按 epoch 保存 checkpoint、修掉 OOM 和 FID 报错、patch 缓存。都是新增的可选参数，不传时与原版一致。
- **修改**：【你的决定】数值输出必须逐位相同；网格允许 ≤ 1×10⁻⁶ m 的偏差；训练只要求多种子加置信区间。

### task6 模拟器 (done)
- **内容**：按官方源码复现深度渲染和 TSDF 融合。与官方融合的三态一致率 99.997%；端到端补全差异 0.03%，而种子之间的自然差异是 3.45%。
- **修改**：发现 patch 坐标有 ≤ 0.5 体素的偏差，已通过 task1 的对齐修正；对齐后复查结论不变。

### task7 NBV episode 框架和基准标定 (done)
- **内容**：闭环框架，所有规划器只有打分函数不同；Random 和 Oracle 在 dev50 上标定。
- **修改**：
  - 【你的决定】候选合法性规则：加上 patch 外遮挡后，至少还有 1 条光线进入 patch；
  - 【你的决定】**冻结 N=64、T=10**。主指标 AUC@10，辅助指标 AUC@5 和第 10 步覆盖率。
- **结果**：AUC@10 Random 0.455、Oracle 0.770，差距明显；Oracle 在第 10 步达到 0.941，接近饱和。

---

## exp2_baseline：基线和对照组（W1–W2，约 80–180 GPU 时，取决于 task4 选出的 S）

**来源**：`exp0.md`（你的设计原稿）；2026-09-27 复述后，按你的修改定稿（标【你的修改，2026-09-27】）。

**目的**：建立"什么都不改"的 Seen2Scene 对照组，并在同一个生成模型上建立单世界和多世界的覆盖规划器。这样后续每一处提升都能明确归因到具体的改动。

**排除清单**：M_obs、GeoControlNet、多观测比例或 NBV 式重训、观测一致性（TSDF 写回、RePaint）、语义、VHD、BALD、语义富度，全部不进入 exp2。

**公平性原则**：所有规划器共享同一套设置，**只有评分函数不同**。共享的是：
- 初始观测；
- 候选集和候选顺序；
- 世界模型、K 和随机种子；
- 光线投射和可见性定义；
- 模拟器；
- episode 长度和评测代码。

以后 VHD 与 S2 比较时，也必须满足这一条。

**固定不变**：
- 官方三个 checkpoint 冻结，主干用 EMA 权重；
- exp1 冻结的 patch、GT、指标、种子方案 `seed_world`、模拟器、候选规则、光线投射和 G_cov；
- NBV 设置 N=64、T=10，主指标 AUC@10，辅助指标 AUC@5 和第 10 步覆盖率。

**数据**【你的修改，2026-09-27】：
- task1–4 用 dev50，做模型验证和参数选择；
- 规划器主表（S1、S2、Frontier、Random、Oracle）用 dev100；
- test 200 完全不碰，留到 exp9 与最终方法一起跑一次。

**统计**【你的修改，2026-09-27】：
- 配对 bootstrap 1 万次，报告 95% 置信区间；
- Holm 校正**只用于预先定义的 AUC@10 主比较**：S1 vs Random、S2 vs Random、Frontier vs Random、S2 vs S1、S2 vs Frontier；
- 辅助指标不计入这个比较族。

**执行顺序**【你的修改，2026-09-27】，分三层：
1. S0-O → S0-F：确认模型和公平条件；
2. 观测比例 → 采样步数：确认模型在主动建图中是否可用，并定下推理工作点；
3. 世界模型接口 → Random/Oracle/Frontier → S1 → S2 → 运行成本：闭环规划器基线。

每个 task 完成后先汇报再继续；**task2、task3、task4 完成后必须停下**，因为它们分别决定公平性、低观测下的可用性和采样步数工作点。

### task1 官方复现（S0-O）(done，2026-09-28，PASS)
- **目的**：确认官方 Seen2Scene 在我们的代码、checkpoint、环境和数据下能正确运行。
- **设置**：
  - 官方权重，官方布局条件，CFG 3.0，S=50；
  - 输入 p_0.1，目标 p_1.0；
  - 每个 patch 生成 K=2 个世界（用于 TMD），使用 `--task.world-seeding`；
  - 先用 10 个 dev patch 跑通，再扩大到 dev50。
- **每个 patch 保存**：部分 TSDF、目标、补全结果、网格、`metrics.json`、`runtime.json`。
- **记录**：三个 checkpoint 的路径和 sha256。
- **指标**：
  - 已知区域和未观测区域的表面带 IoU、F@1/F@2、Chamfer-L1；
  - 已观测区域误差和观测违反率；
  - 官方脚本算的 CD、L2、TMD，与论文对照（带 bbox：CD 2.05、L2 0.51、TMD 3.33；不带 bbox：CD 1.92、L2 0.53、TMD 1.69；×10⁻²/10⁻⁴/10⁻²；只有 CD 只在 3D-FRONT 上算）。
- **硬性检查**【你的修改，2026-09-27】：
  1. checkpoint、预处理、推理、指标这一整条流程正常；
  2. 输出没有 NaN 或异常；
  3. 视觉结果合理；
  4. 相同种子时数值输出可复现：逐位相同，网格 ≤ 1×10⁻⁶ m；
  5. VAE 检查沿用 exp1-task2 的结果（F@1 0.995）。
- **预警范围**【你的修改，2026-09-27】：官方 CD 在论文值的 0.5–2 倍之内。超出时人工检查，**不作为停止条件**，因为我们的留出场景和 patch 与论文不同。
- 【实现说明，2026-09-28】第一次 dev50 运行后的可复现性检查发现，S0-O 的世界依赖同一进程内的处理顺序。原因是官方 Conditioner 会缓存不在预计算表中的物体名的 CLIP 嵌入，而嵌入数值取决于这个名字第一次和哪些名字一起编码。已在 `world_seeding` 开关下修复：每个 patch 开始前清空缓存，并同时覆盖 ControlNet 放在普通 dict 里的基础生成器。修复后已验证与新进程中单独运行逐位相同。dev50 用修复后的代码重跑；旧结果保留为 `dev50_v1`，用来统计受影响的 patch。S0-F 只使用名字 `" "`，已验证不受影响。
- 【实现说明，2026-09-28】倒序测试又发现第二个来源：默认 GPU 算法中的浮点原子加法，其求和顺序随显存布局而变，所以即使修了缓存，仍有 3/12 个 patch 与处理顺序有关（约 0.01% 的体素）。排查结果：不是未初始化内存，也不是 cuBLAS 工作区。修复：`--task.world-seeding` 开启时，同时启用 `torch.use_deterministic_algorithms(True, warn_only=True)` 和 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，只改变浮点求和顺序，约慢 4%，网格也能逐位复现。task1 的 dev50 用最终管线重跑（`dev50_v3`）。**以后所有生成世界的运行（包括 task5 的世界模型接口）都必须使用这个设置。**
- 【结果，2026-09-28】5 项硬性检查全部通过；最终管线下生成体素和网格都逐位可复现，与处理顺序无关。dev50 上已知区域 F@1 0.806，未观测区域 F@1 0.633；官方 CD 3.22×10⁻²，是论文的 1.57 倍，在预警范围内。详见 `experiments/exp2_baseline/results.md`。
- 【你的决定，2026-09-28】task1 通过。确定性模式固定并入 `--task.world-seeding`，之后所有公平比较的生成实验都使用它；网格判据不放宽；每张卡只放一个生成进程；task3、task4 批量运行默认跳过官方可视化，只对固定的少量代表性 patch 出图。

### task2 公平基线（S0-F）(done，2026-09-28，PASS)
- **目的**：建立主动建图实际使用的基线。GT 布局在主动建图中属于"未来信息"，必须去掉。
- **设置**：与 task1 完全相同，只加 `--task.drop-bbox`。它去掉布局，同时 guidance 变为 1.0，即不做 CFG，这两项无法分开。
- **报告**：
  - S0-O vs S0-F 的补全质量，以及 TMD（K=2）；
  - 【你的修改，2026-09-27】同时给出**绝对下降和相对下降**，例如 0.62 → 0.56 写作 −0.06（−9.7%）。
- **阈值**【你的修改，2026-09-27】：未观测区域 F@1 相对下降 ≤ 10%，表述为"去掉布局后模型仍然较稳健"。这只是**报告和预警用的阈值**：无论下降多少，后续的主动建图规划器都使用 S0-F。
- 完成后**停下汇报**。

- 【结果，2026-09-28】没有失败或空输出；无布局下仍然完全确定（倒序重跑 12/12 逐位相同）。去掉布局后指标反而变好：已知区域 F@1 从 0.806 升到 0.857（+0.051），未观测区域 F@1 从 0.633 升到 0.686（+0.053），置信区间都不含 0；官方 CD 从 3.22 降到 2.20（×10⁻²）；世界之间的差异不变（TMD 1.55→1.61）。混杂因素：S0-F 同时少了 CFG，已观测区域也变好，说明主要可能是 CFG 在起作用。详见 results.md。
- 【你的决定，2026-09-28】task2 通过，后续统一用 S0-F。补跑诊断 B（有布局 + guidance 1.0），用 A→B 看 CFG 的作用、B→C 看布局的作用，只用于解释 task2。
- 【诊断结果，2026-09-28】A（布局 + CFG 3）→ B（布局 + g1）：未观测 F@1 +0.067，说明提升主要来自去掉 CFG。B → C（无布局 + g1）：未观测 F@1 −0.015（不显著），表面带 IoU −0.019（显著）。所以布局本身只有很小的帮助，去掉它的真实代价约为未观测区域 2%。详见 results.md。

### task3 观测比例（done，2026-09-28，PASS，结果冻结）
- **目的**：测量模型在不同观测比例下的退化。NBV 的初始覆盖约为 0.25，与 p_0.01 在同一稀疏区间。
- **设置**：
  - p ∈ {0.01, 0.05, 0.1, 0.2, 0.5}，外加 p=1.0 作为健全性检查；
  - S0-F，S=50，K=1，dev50；
  - p=0.1 直接复用 task2 的 k=1 世界。
- **输出**：
  - Quality(p) 曲线，重点是未观测区域的 F@1/F@2；
  - 每个 patch 实际的观测比例；
  - 注明 p ≠ 0.1 属于官方 ControlNet 训练分布之外。
- **p=0.01 的空输入**【你的修改，2026-09-27】：有些 patch 的输入没有任何表面，官方 Dataset 会悄悄替换它们，所以直接裁剪输入喂给模型。分别报告：
  1. 推理失败率；
  2. 成功样本上的指标；
  3. 如果需要端到端的汇总数字，再额外把失败样本记为 0 算一个总体指标。
- **检查**：p=1.0 时输出基本还原输入；质量随 p 基本单调上升。
- **预警，不是停止条件**：F@2(p=0.01) < 0.5 × F@2(p=0.1) 时预警。低观测下重建质量差，不代表世界之间的分歧没有价值。
- 完成后**停下汇报**。
- 【实现说明，2026-09-28】
  - **空输入**：冒烟测试证实，p=0.01 的 3 个空输入 patch（db70885c、eb1f5725、fd5ffce0）**根本到不了模型**，官方流程在构建输入的稀疏网格时就报错（`Empty tensor (ijk)`）。所以运行时把它们排除在场景列表外（否则 Dataset 会悄悄换成随机场景），直接记为推理失败：失败率 3/50，成功样本 47 个，另外再给一个失败记为 0 的总体指标。
  - **新增可选参数**（默认关闭，官方行为不变）：`--task.skip-render` 和 `--task.render-scenes`：批量运行不出图，只对 8 个固定的代表性 patch 出图，按 task2 中 S0-F 的未观测 F@1 取最好、中位、最差各 2 个，另加 2 个 Chamfer 离群 patch；`--task.record-failures`：生成阶段出错时写 `failure.json` 并继续下一个 patch。
  - **p=0.1**：直接复用 task2 中 S0-F 的第一个世界（同样的种子和输入，确定性管线下逐位相同）。
  - 驱动脚本：`experiments/exp2_baseline/run_task3.sh`。
- 【结果，2026-09-28】已知 F@1 从 p=0.01 的 0.535 上升到 p=0.2 的 0.909，之后饱和在约 0.92。p=0.01 的推理失败率为 6%（3/50）；低观测预警没有触发（未观测 F@2 为 0.489，高于 0.384）。**p=1.0 的健全性检查没有通过**：已知 F@1 为 0.922，低于 0.95。原因不是流程错误，而是即使给出完整观测，模型也不会原样保留它。详见 results.md。
- 【你的决定，2026-09-28】task3 通过，结果冻结。p=1.0 的现象记录为当前模型的观测一致性局限（当前推理配置下实测约 F@1 0.92），不作为停止条件，exp2 中不修改模型或写回。

### task4 采样步数（done，2026-09-29，PASS）
- **目的**：确定推理步数的工作点。每个 NBV 步都要生成 K 个世界，S=50 太贵。
- **设置**：S ∈ {5, 10, 20, 50}，S0-F，p=0.1，K=1，dev50。
- **代码**：新增推理参数 `--task.num-inference-steps`，默认 50；先验证不传时与原版逐位相同。
- **指标**：F@1/F@2（已知区域和未观测区域）、每个世界的耗时、峰值显存。
- **规则**：
  - 取未观测区域 F@1 相对 S=50 的平均下降 ≤ 5% 的最快的 S；
  - 在 20 个 patch 上测 S ∈ {10, 20, 50} 时 K=4 的 TMD，如果比 S=50 下降超过 20%，就退到更大的 S；
  - 可选：在选定的 S 和 S=50 下补测 p=0.01。
- **规划器层面的检查**【你的修改，2026-09-27】：
  - 在 10–20 个 dev patch 上，对 S ∈ {10, 20, 50}，用同一组 64 个候选计算 G_cov；
  - 报告与 S=50 排序的 Spearman 相关系数，可选报告 Top-1 一致率和 Top-5 重合度；
  - 理由：最终关心的是视角的排序，不只是重建的 F 值。
- **不预设 S=10**，工作点完全由 task4 的结果决定。
- 完成后**停下汇报**。
- 【实现说明，2026-09-29】新增可选参数 `--task.num-inference-steps`（默认不改变官方的 50 步），已验证显式传 50 时与不传时逐位相同。
- 【结果，2026-09-29】按预先定的规则选中 **S=10**：p=0.1 时未观测 F@1 +1.6%，TMD −11.7%，排序一致性 Spearman 0.976、Top-1 95%，比 S=50 快 5 倍。代价：p=0.01 时未观测区域下降 3–5%，未观测表面状态不一致下降 11%；S=20 能约减半这两项代价，但成本加倍。K=4 时峰值显存有一张卡达到 44 GB。详见 results.md。
- 【你的决定，2026-09-29】task4 通过，按预设规则固定 **S=10**。S=20 保留为稳健性设置，只在多样性成为瓶颈时再补。显存风险在 task6 通过分批或逐个生成来处理，task7 精确测量。**后续 exp2 默认配置：S0-F + 无布局 + guidance 1.0 + S=10 + 确定性 world seeding。**

### task5 单世界规划器（S1），先做世界模型接口（done，2026-09-29，PASS）
- **世界模型接口 `seen2scene/active/world_model.py`**【你的修改，2026-09-27：严格等价】：
  - 流程：当前融合的 TSDF → world_model.py → Seen2Scene（S0-F），输出稠密的世界 TSDF；
  - 检查：输入同一个官方 fusion_p_0.1 裁剪、使用同一个种子时，与官方测试流程相比：
    - 数值输出（体素、潜变量）逐位相同；
    - 网格拓扑相同；
    - 网格坐标 ≤ 1×10⁻⁶ m（沿用 exp1 的标准）。
  - 只有通过这个检查，闭环中的模型才能被视为与 task1–4 测的是同一个模型。
- **流程**：当前观测 → 1 个世界（种子 `seed_world(scene, 0, t, k=1)`）→ 64 个候选 → 光线投射（含 patch 外遮挡）→ G_cov → 选分数最高的。
- **参数**：K=1，S 取 task4 的结果，N=64，T=10，dev100，episode 0。
- **没有观测到表面时**【你的修改，2026-09-27】：
  - 先正常尝试用 S0-F 生成，因为即使没有表面，观测里也可能有自由空间和 carving 的信息；
  - 只有当世界模型确实无法产生有效世界时，才退回随机选择，并记录 `fallback_count`；
  - 这条规则对 S1、S2 以及以后的 VHD 完全相同。
- **通过**：100 个 episode 全部跑完；抽 1 个 episode 重跑，日志逐位相同。
- 【结果，2026-09-29】世界模型接口严格等价验证通过：10 个 patch × k=0..3 的世界 40/40 与官方测试流程逐位相同，网格 40/40 完全相同，输入 10/10 相同，进程内与顺序无关。k 从 0 开始计数：S1 用 k=0，S2 用 k=0..3。Frontier 已按冻结定义实现。dev100 上的 Random、Oracle、Frontier、S1 正在同一确定性设置下运行。
- 【结果，2026-09-29】dev100（N=64，T=10）上的 AUC@10：Random 0.435，Frontier 0.579，**S1 0.599**，Oracle 0.758。S1 − Random = +0.164 [+0.143, +0.185]（Holm p=0.0002）；S1 − Frontier = +0.019（置信区间跨 0；表述为"与强 Frontier 基线相近、略高"，不说显著优于）。退回随机 4 次，都是初始覆盖为 0 的 episode 的第 1 步。闭环可复现（3 个 episode 重跑，日志逐位相同）。S1 每步约 21 s。详见 results.md。
- 【你的决定，2026-09-29】task5 通过。保留初始观测为空的 episode 和退回规则。进入 task6：S2 必须与 S1 严格对照（S1 的 k=0 就是 S2 的第一个世界，候选、光线投射、模拟器、种子和评测都相同）。允许逐个世界生成、打分、累加、释放，但要先做小规模严格等价检查。

### task6 期望覆盖（S2），以后 VHD 最重要的对照（done，2026-09-30，PASS）
- **流程**：K=4 个世界（k=1…4，其中 k=1 与 S1 的世界相同）→ 每个候选在 4 个世界中分别算 G_cov → 取平均 → 选最大的。
- **参数**：K=4；【你的修改，2026-09-27】第一版 VHD 也必须使用同样的 K=4。
- **理由**：VHD 与 S2 的世界模型、K、候选、种子、光线投射、模拟器和评测完全相同，只有评分函数不同。所以只有 VHD > S2，才能把收益归因到世界之间的分歧，而不是"用了多个世界"。
- **预警**【你的修改，2026-09-27】：如果 S2 的 AUC@10 没有显著高于 Random，在开始后续实验前先讨论。但这**不是 VHD 的停止条件**：期望覆盖无效，不代表世界之间的分歧无效。更危险的情况是 K 个世界在未观测区域几乎没有差异，这由 exp3 的止损规则检查。
- 【实现说明，2026-09-29】流式运行（生成一个世界 → 打分 → 累加 → 释放）。严格等价检查通过：4 个 patch 上与一次生成 4 个世界相比，世界逐位相同，每个世界的 G_cov、平均分和选择完全相同，S1 的世界就是 S2 的第一个世界。改成流式后 S1 的日志不变。每步约 79 s，峰值显存（allocated）11.4 GB。dev100 正在运行。
- 【结果，2026-09-30】dev100 上的 AUC@10：S2 0.615，S1 0.599，Frontier 0.579，Random 0.435，Oracle 0.758。按完整的 5 项比较组做 Holm 校正后全部显著：S2 − S1 = +0.016 [+0.007, +0.025]，S2 − Frontier = +0.035 [+0.014, +0.056]。退回随机 4 次，与 S1 相同。K=4 世界的两两排序 Spearman 为 0.865，只有 30% 的步 4 个世界 Top-1 相同，未观测表面状态不一致 5.6%。每步 80 s，其中生成世界 78 s；PyTorch 分配峰值 11.5 GB，进程峰值 40–44 GB。详见 results.md。

- 【你的决定，2026-10-01】task6 通过。在与 S1 完全相同的世界模型、候选、光线投射、模拟器和评测下，把 1 个世界扩展到 K=4 个世界并对期望覆盖取平均，AUC@10 从 0.599 提升到 0.615（+0.016 [+0.007, +0.025]，Holm 后显著），并显著优于 Frontier：提升不大但稳定，说明多世界信息本身有额外的规划价值。4 个世界的整体排序仍较一致（Spearman 0.865），但只有 30% 的步 Top-1 完全相同，说明世界分歧确实会传到动作偏好上；这只能作为 VHD 的动机，不能据此说 VHD 已经有效。

### task7 运行成本（done，2026-10-01，PASS）
- **测量**：
  - VAE 编码、世界采样、解码、光线投射、评分、总决策延迟、峰值显存；
  - 条件：S=50 和 task4 选出的 S；K ∈ {1, 4, 8}，分批和逐个两种方式；
  - 每项重复 5 次取中位数。
- **理由**：以后如果要说某个 S 和 K 适合主动建图，必须有官方 S=50 作为对照。

- 【你的修改，2026-10-01】不重新跑大规模基准，分两部分：
  1. **整理已有成本**：从 dev100 的 S1（task5）和 S2（task6）日志整理每步决策延迟、生成世界时间、光线投射/打分时间、规划器其余开销、每个世界的摊销成本和 dev100 的总 GPU 时。
  2. **在少量固定状态上补测**：K=1 和 K=4（S=10，流式，与正式运行相同的代码路径）的总决策延迟、生成世界时间、光线投射/打分时间、规划器其余开销、CUDA 峰值 allocated、CUDA 峰值 reserved、nvidia-smi 统计的进程显存、每个世界的摊销成本。
  - **显存要分清**：实际张量占用（allocated）、缓存分配器保留（reserved）和 nvidia-smi 看到的进程显存是三个不同的量，不能把缓存写成实际占用。
  - 原计划中的 S=50 对照直接引用 task4 的实测（每个世界 S=50 为 98.1 s，S=10 为 19.6 s）；K=8 和一次性生成方式按这次的范围不再单独测。
  - task7 完成后 exp2 结束，先停下汇报，不进入 VHD。
- 【实现说明，2026-10-01】固定状态和测量方式（运行前写定）：
  - **状态**：exp_vis task1-A 预先登记的 4 个场景（e8150c6b、e3ae803e、fd316eb3、dd804295，当时只按 S2 日志选出，与成本无关），每个场景取 S2 episode 的第 1 步和第 10 步的决策状态，即最少和最多观测的状态，共 8 个。状态按初始观测加上 S2 日志中已执行的视角重建。
  - **运行**：4 张卡各一个进程，每个进程一个场景，与 dev100 的运行方式相同。每个状态交替运行 K=1 和 K=4 各 5 次，取中位数；进程中的第一次决策单独记为冷启动。之后各加 1 次细分计时（编码、采样、解码等阶段之间加同步），只用来拆分时间。
  - **显存**：每次决策前重置峰值统计，记录 `max_memory_allocated`、`max_memory_reserved`；另用 NVML 每 50 ms 采样本进程的显存（即 nvidia-smi 的进程显存）。细分计时那次之前先清空缓存，得到一次决策从空缓存开始实际需要保留的显存。进程结束前清空缓存，再读一次进程显存，得到权重、CUDA 上下文和渲染器等的固定部分。
  - **正确性检查**：重建状态的覆盖率等于日志；每次 K=4 决策的选择、分数和每个世界的 Top-1 与 S2 日志完全相同；K=1 在第 1 步与 S1 日志相同、在第 10 步与 S2 日志中第 1 个世界的 Top-1 相同；同一状态的重复结果逐位相同。
  - 【补充，2026-10-01】冒烟测试中固定状态的进程显存只有约 13–14 GB，复现不出 dev100 的 40–43 GB；而 dev100 的显存曲线显示，每个进程在第一个 episode 内就从 3 GB 涨到 34–42 GB，之后偶尔回落到 12–15 GB。为了把这部分说清楚，加一个小诊断：在新进程中完整重放一个 S2 episode（dev100 第二段 GPU0 进程的第一个 episode e75c45f8，当时进程显存在这个 episode 内从 3 GB 涨到 41 GB），逐步记录候选生成、打分、执行各阶段的 allocated 峰值、reserved、进程显存和分配器事件（显存不足后清空缓存重试的次数）。同时用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 再重放一次，只改变缓存分配器的策略。两次的每一步日志（覆盖率、选择、分数、分歧统计）都必须与 S2 日志完全相同。这只是诊断，不改变正式配置。
  - 代码：`experiments/exp2_baseline/task7_cost_from_logs.py`、`task7_profile.py`、`task7_replay.py`；输出在 `work_dirs/exp2_baseline/task7_cost/`。
- 【结果，2026-10-01】8 个固定状态的全部检查通过（选择、分数与 S1/S2 日志完全相同）。决策延迟 K=1 为 20.3 s，K=4 为 80.1 s，每多一个世界 +20.0 s，95–96% 是流匹配采样，光线投射和规划器自身合计不到 3%。实际张量峰值（allocated）10.9 GB，与 K 无关；dev100 中 nvidia-smi 看到的 39–43 GB 是缓存分配器保留的空闲缓存（重放诊断复现了上涨，只换成 `expandable_segments` 后进程显存为 11.8 GB，结果逐位不变）。dev100：S1 6.3 GPU 时，S2 22.9 GPU 时（整个运行 23.5），每个世界摊销约 21 s。详见 results.md。
- 【你的决定，2026-10-01】task7 通过，**exp2 正式结束，不再加 task8**。`expandable_segments:True` 只记为已验证不改变结果的工程选项，不修改冻结的基线。
### 规划器主表的其他成员（dev100）
- **B-Random**：从合法候选中随机选。dev50 已有结果，补跑 dev 的另外 50 个场景。
- **B-Oracle**：用 GT 按覆盖贪心，作为上界和健全性检查。同样补跑另外 50 个场景。
- **B-Frontier**【你的修改，2026-09-27：冻结定义】：
  - frontier 体素：已观测为空，并且 6 邻域中至少有一个是 UNKNOWN；
  - 每个候选的分数：它能直接看到的 frontier 体素数，按体素去重；
  - 同样使用 patch 外遮挡；
  - 它完全不使用生成模型，只依赖当前观测，是经典的几何基线。
- 健全性检查：所有规划器都不应高于 Oracle。

### exp2 之后、进入 VHD 之前的顺序【你的决定，2026-10-01】
1. task7 完成后先汇报，不开始 VHD。
2. **task6 事后诊断**（尽量只用已有日志和 GT 候选打分，不重新生成世界）：
   - 初始分歧诊断：只在 S1/S2 状态完全相同的第 1 步，分析世界 Top-1 的多样性、两两 Spearman、表面状态不一致，以及 S1/S2 是否选了不同的相机，与逐 episode 的 AUC_S2 − AUC_S1 的关系。只做描述性关联，不作为 VHD 调参的依据。
   - 离群抑制诊断：在 S2 的同一个规划状态中，比较 4 个世界各自的 Top-1 候选与 S2 平均分选出的候选的 GT 一步收益，检查 S2 的收益是否部分来自平均掉单个幻觉/离群世界的极端偏好。
3. 两个诊断完成后冻结 **VHD-v1**：只用现有 K=4 个世界的未来命中/未命中和首次命中深度的分歧；不训练模型、不加语义、不做 beam search、不根据 GT 调权重。
4. VHD-v1 冻结后先跑 10–20 个 episode 的冒烟测试，只检查实现正确性、分数分布、退化视角、退回、确定性、GT 泄漏和运行时间，不据此判断方法优劣；通过后再跑正式的 dev100。
5. 最关键的比较是 **VHD vs S2**：两者使用完全相同的 K=4 个世界、种子、候选、光线投射和模拟器，只改变规划目标。
（这些步骤与下面 exp4/exp5 原写法的对应关系，在复述时给出新旧对照表。）
- 【你的决定，2026-10-01，来自 task7】K=4 的世界生成本身约 78 s，而 S2 的光线投射和打分只有约 1.7 s，所以 **VHD 复用与 S2 完全相同的 K=4 个生成世界**；暂时不优化生成器，先验证规划目标是否有价值。
- 第 2 步（task6 事后诊断）的定义见下面的 **pre_vhd** 一节。

### 主表（写入 `experiments/exp2_baseline/results.md`）

**表 A：世界模型基线**

| Method | Layout | p | Steps | F@1 | F@2 | Unseen F | Chamfer | TMD | Runtime |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0-O | GT | | 50 | | | | | | |
| S0-F | None | | 50 | | | | | | |

另外有"补全质量 vs 观测比例"曲线（task3），以及"质量、耗时和 G_cov 排序一致性 vs S"表（task4）。

**表 B：规划器基线**（dev100）

| Planner | World Model | K | AUC@10 | AUC@5 | Coverage@10 | Decision Latency | fallback_count |
|---|---|---:|---:|---:|---:|---:|---:|
| Random | – | – | | | | | – |
| Frontier | 当前观测 | – | | | | | – |
| S1 Single Coverage | S0-F | 1 | | | | | |
| S2 Expected Coverage | S0-F | 4 | | | | | |
| Oracle | GT | – | | | | | – |
| VHD（留空，exp4/exp5 填） | S0-F | 4 | | | | | |

**命名**：S0-O = Seen2Scene-Official，S0-F = Seen2Scene-Fair，S1 = S2S-SingleCov，S2 = S2S-ExpectedCov。

**成功标准**：
1. S0-O 能稳定复现；
2. S0-F 的无布局行为明确；
3. 得到观测比例曲线；
4. 定下步数工作点；
5. 世界模型接口严格等价；
6. S1 和 S2 都能完整跑完 dev100；
7. 所有后续方法都能直接复用同一套评测管线。

---

## exp_vis（EXP_VIS）：世界假设可视化（插在 exp2-task5 和 task6 之间，设计见根目录 `vis.md`、`vis_step_experiment.md`）

目的：同一个部分观测下，S0-F 的 K 个世界的差异是表面抖动还是结构不同的假设；这些差异会不会改变候选相机的评分和 Top-1。只做定性检查，不训练、不加模块，不改 EXP2 的代码路径。代码在 `experiments/exp_vis/`，输出在 `work_dirs/vis/`。

【编号和目录调整，2026-09-29，你的决定】

| 原来 | 现在 | 输出目录 |
|---|---|---|
| VIS1 | task1 `vis_xworld`（同一观测、不同种子的多世界对比） | `work_dirs/vis/vis_xworld/` |
| VIS1-A（3D-FRONT） | task1-A | `work_dirs/vis/vis_xworld/front3d/`（原 `work_dirs/vis/front3d/`） |
| VIS1-B（MACARONS++） | task1-B | `work_dirs/vis/vis_xworld/macaronspp/` |
| （新）尺度变体 | task1-B2 | `work_dirs/vis/vis_xworld/macaronspp_scale/` |
| – | task2 `vis_step`（采样轨迹，新） | `work_dirs/vis/vis_step/` |

代码：task1 的脚本在 `experiments/exp_vis/vis_xworld/`（`run_vis1.py` 改名为 `run_xworld.py`），task2 在 `experiments/exp_vis/vis_step/`，共用 `experiments/exp_vis/vis_common.py`。

### task1 vis_xworld：同一观测、不同种子的多世界对比（原 VIS1）

#### task1-A 3D-FRONT（原 VIS1-A；2026-09-29 复述后定稿；done）
- **状态**：S2 第 1 步的状态（初始观测、种子 seed_world(scene, 0, 1, k)、64 个候选），世界用未修改的 S2 代码生成（S0-F、guidance 1.0、S=10、K=4、确定性 world seeding）。与 S2 日志逐项核对，必须完全相等。不用 task4 的 p_0.1 世界作主结果。
- **场景**（预先登记，只用 S2 日志）：S2 暂停时已完成的 45 个有效 episode 中，未观测分歧最小 / 中位数 / 最大，以及不同 Top-1 最多的场景：e8150c6b、e3ae803e、fd316eb3、dd804295。
- **图 1**：观测 + W1–W4（+ GT 仅参考），官方相机和渲染参数，所有面板相同。【你的修改，2026-09-29】灰色 = 固定的观测证据（初始深度观测的命中点，只提取一次，所有面板逐点相同），橙色 = 该世界生成的几何；原来的画法（世界自己的表面按观测截断带着色）保留为诊断图 `worlds_raw_generated.png`，用来看观测一致性。图 2 不变。
- **图 2**：未观测区域（obs == UNKNOWN）的 3D 分歧 D = 4p(1−p)，表面容差 τ=1 体素为主，另报 τ=0/2；结构占比 = 有争议比例(τ=2)/有争议比例(τ=0)。
- **图 3**：同一组 64 个候选在 4 个世界中的 G_cov、期望覆盖、Top-1/Top-5 和 Spearman。【你的补充】`candidate_scores.json` 保留未归一化的原始 G_cov，以便判断 Top-1 差异是大 gap 还是小 gap。
- **A3**：同一候选位姿投进 4 个世界的命中和深度；组合分歧用 10 cm 归一化，只在 ≥20% 光线进入 patch 的候选里选高/低视角。【你的补充】同时保留连续深度分歧和命中/未命中分歧的原始统计，不只保留阈值化的结果。汇报时放在三类核心图之后。
- GT 只作参考，不参与任何场景选择、候选选择或分歧定义。
- **完成标准**：vis.md §5，外加与 S2 日志的核对全部相等，以及新进程重跑一个场景结果完全相同。
- **预警判读**（只用于判读，不是停止条件）：结构占比 < 0.2 的场景 ≥ 3/4 → 预警 1；4 个场景的高分歧视角的分歧都 < 0.05 → 预警 2；出现 ≥2 个不同 Top-1 或两两 Top-5 重合 < 0.6 → 预警 3（只记录为证据，不据此说 VHD 有效）。
- 【结果，2026-09-29】与 S2/S1 日志的核对 4/4 全部相等，新进程重跑逐位相同。结构占比 0.68–0.87（纯抖动 ≤ 0.02），预警 1、2 未触发；2/4 个场景的 Top-1 改变（其中既有小 gap 的重排，也有离群世界造成的大 gap 偏好），预警 3 触发。详见 `experiments/exp_vis/results.md`。
- 【你的决定，2026-09-30】A3 的主结果正式使用 view_low_seen（≥20% 光线在至少一个世界里命中表面，其中分歧最低者）；预先登记的"20% 光线进入 patch"规则在 4 个场景都选出了退化视角，保留为退化诊断。

#### task1-B MACARONS++（原 VIS1-B；2026-09-30 复述后定稿）
- **原则**【你的要求】：只做适配器扩展。MACARONS++ 的坐标、尺度、patch、相机和观测，都在 `experiments/exp_vis/vis_xworld/macaronspp/` 下通过独立的适配器转换成 task1-A 已有的数据接口。
  - 不修改：`seen2scene/active/world_model.py`、EXP2 规划器、`nbv_run`、冻结的候选打分、光线投射和 world seeding，以及 3D-FRONT Dataset 的行为。如果非改不可，先停下来汇报。
  - 画图代码只允许把写死的 front3d 路径改成显式参数，front3d 的默认行为必须保留。
- **回归门槛**：
  - 目录调整后，先在一个已有的 3D-FRONT 场景上做回归测试。输入观测、W1 哈希、64 个 G_cov、`candidate_scores` 和关键可视化 metadata，都必须与 task1-A 逐位或逐项相同。
  - 适配器必须在 3D-FRONT 上逐位复现 task1 的 W1。
  - 任何 task1-A 或 front3d 的数值、哈希发生变化，都先停止 task1-B。
- **场景**：sestino_museum、barts、pantheon。colosseum 是预先登记的备选，只有当 pantheon 因输入或网格的技术原因无法运行时才替换，不能根据生成质量替换。
- **尺度**：保持 MACARONS++ 原来的运行时尺度（OBJ × 10），不做逐场景人工缩放，并注明这不是严格的米制标定。
  - 生成前记录：场景范围、相机到 patch 的距离、有效观测的统计。
  - 如果尺度导致传感器或 patch 无法形成有效输入，就停下汇报，不临时调整尺度。以后如果要做归一化，作为单独的变体。
- **patch**：
  - 候选 patch 必须有足够多的合法位姿网格视角能产生有效的 patch 命中，可见性阈值在运行前冻结；在满足条件的候选里，选非地面表面最多的一个。
  - patch 底部 = 估计的地面 − 3 个体素。
  - 不根据生成结果移动 patch。
- **观测**：用 MACARONS 位姿网格，不用 MAGICIAN 轨迹。在能看到 patch 且有有效命中的视角中，用种子 2027 取 3 个，生成前写入 `selection.json`，不根据覆盖率或生成质量重选。
- **候选**：从同一位姿网格剩下的合法视角中，用种子 2027 取 64 个，提前冻结。
- **参考几何**：
  - 由所有合法视角融合得到的，叫 **full-scan reference**，不叫 GT；
  - 直接用原始网格裁剪得到的，可以叫 GT。
- **A3**【你的决定】：
  - 主结果正式使用 view_low_seen：先取至少 20% 光线在至少一个世界里命中表面的视角，再从中选分歧最低的一个；
  - 原来"20% 光线进入 patch"的规则保留为退化诊断，不作为主结果；
  - task1-A 同样按此解读，数值不变。
- **不变**：S0-F，S=10，K=4，确定性 world seeding；四类可视化 + 跨域记录表（vis.md §3.5）。S2 继续暂停。
- 【回归门槛结果，2026-09-30】通过。适配器在 dd804295 上逐位复现 task1-A 的 W1–W4；整套 task1-A 流程在 dd804295 上重跑，数据、哈希、64×4 个 G_cov、`candidate_scores.json` 和 metadata 各节全部相同，只有 pyrender 的渲染噪声（≤1 个灰度级）。
- 【冻结的选择规则，2026-09-30，运行前写定；细节见 `macaronspp/select_macarons.py`】
  - 合法位置：从相机位置向 26 个方向探测，首个命中都不是背面（相机不在物体内部或背后）。
  - 合法位姿：在合法位置上，沿用 3D-FRONT 候选的避障检查（10×10 探测光线，最近命中 ≥ 0.1）；3D-FRONT 的"在地板上方"和"无背景"两条不适用。
  - 有效 patch 命中：128² 打分光线中至少 1%（164 条）的首个命中在 patch 内。
  - 地面：向上的水平面面积最大的高度层，取该层的面积加权 5% 分位；patch 底部 = 地面 − 3 个体素。
  - 候选 patch：体素格点上每 91 个体素（1.001）取一个 xy 原点，范围在 MACARONS 场景框内。
  - 合格 patch：至少 67 个合法位姿有有效命中（3 个观测 + 64 个候选）；其中选非地面表面面积（高于地面层 0.20 以上）最大的一个。
  - 观测和候选：合格位姿按网格序号排序，种子 [2027, 场景名哈希] 打乱，前 3 个为观测，后 64 个为候选。
  - full-scan reference：该 patch 全部合格位姿融合而成；GT 为原始网格裁剪（只用于显示）。
- 【结果，2026-09-30】3 个场景推理成功，硬性检查和新进程复现全部通过。观测复现 84–99%；补全带明显的 3D-FRONT 先验：sestino、barts 的 8 个世界都在底部铺满地面，pantheon 的柱子被补成开放墙角。结构占比 0.44–0.63。3 个场景的 Top-1 都一致，原因主要是候选离 patch 远、G_cov 由可见范围决定，按预警 4 只记为域偏移。A3 的 20% 门槛只有 3/1/3 个候选满足，barts 的主结果与 view_high_seen 是同一个候选。详见 `experiments/exp_vis/results.md`。

- 【定位，2026-09-30 你的决定】task1-B（k=1）保留为**原始运行时尺度、不做任何缩放**的跨域基线，结果全部保留，不覆盖，用来说明直接迁移时的尺度局限。

#### task1-B2 尺度归一化变体（2026-09-30 你的决定）
- **问题**：模型的固定输入（256³，每体素 1.1 cm，约 2.8 m）对应更大的原场景范围时，跨域生成的结构可读性、世界多样性和规划分歧会不会系统性地变化？不预设 k=3 一定更好。
- **说法**【你的要求】：模型输入没有变大，仍然是 256³、1.1 cm、约 2.8 m。k=2、3 是把整个 MACARONS++ 场景统一缩小 k 倍，所以同一个模型输入对应原场景 5.6 / 8.4 个单位（native extent）。代码和文档里用"尺度因子 k / native extent"，不写成 8.4 m。
- **尺度**：k = 1、2、3，三个都做。k=2 用来判断变化是否随有效上下文逐渐出现，避免只比较两个端点。k=1 的 3 个场景直接复用 task1-B。
- **场景**：sestino_museum、barts、pantheon（与 task1-B 同一个 patch 中心，底部在地面），外加 eiffel。
  - eiffel 的位置事先指定为西南塔腿脚下 (−13.0, −13.5)，是为这次尺度和上下文研究预先固定的位置，不是根据生成结果挑的；生成前写进 selection 的 metadata。
  - eiffel 的 3 个观测和 64 个候选按 task1-B 的规则 R2–R4、R9，在 k=1 的框上抽一次，三个尺度共用。
- **唯一改变的是全局尺度**【你的要求】：
  - 同一场景的 k=1/2/3 用完全相同的原始相机位姿、观测、64 个候选和种子；
  - 网格坐标、相机平移、深度和 patch 坐标统一乘 1/k，旋转不变；
  - 不随 k 重新挑观测或候选。
- **参考几何**：full-scan 和 GT 按每个尺度的新 patch 重新构造，选择本身不依赖生成结果。
- **输出**：
  - 保留 task1-B（k=1）全部结果；
  - 新增 k=2、k=3（eiffel 还有 k=1）；
  - 每个场景一张 k=1/2/3 的配对对比图；
  - 上下文裁剪图，标明 2.8 / 5.6 / 8.4 个 native 单位。
- **指标**：task1-B 的全部指标，另加：
  - 每个尺度的观测密度：观测命中体素数、表面覆盖、观测表面的连通程度（你的要求）；
  - 未观测区域中各世界与 GT（原始网格裁剪）的表面精度和召回（仅参考）。
- **完成后先汇报，不继续改模型。**
- 【eiffel，2026-09-30 你的决定】按建议执行：k=1 在固定位置只有 8 个合格视角（需要 67），记为原始尺度下输入无效；3 个观测和 64 个候选改在 k=2 的框上按 R2–R4、R9 抽取（同一个种子），k=3 复用；单独出一张与 `compare_k3.png` 布局相同的横排图（每行一个 k）。
- 【结果，2026-09-30】3 个场景的 k=2、3 全部跑完，硬性检查、k=1 等价检查和新进程复现都通过。增大 k 后观测变连续（每像素约 1 个体素），结构占比从 0.44–0.63 升到 0.72–0.88，6 组中 5 组出现不同的 Top-1，A3 的 20% 门槛恢复正常；但与 GT 的一致程度没有系统性变好。eiffel 在 k=1 只有 8 个合格视角（需要 67），按规则停止；按你的决定补跑了 k=2、3（观测和候选在 k=2 抽取），k=3 时 4 个世界的 Top-1 分成 2 比 2。详见 `experiments/exp_vis/results.md`。

### task2 vis_step：采样轨迹（新；2026-09-29 复述后定稿）
- **目的**：说明同一个种子的 50 步采样只是一条生成轨迹上的中间状态，多个世界假设来自不同的初始噪声（种子）。
- **固定**：
  - 冻结的 Seen2Scene，S0-F（无布局，guidance 1.0），确定性执行；
  - 观测与 task1 完全相同（S2 第 1 步）；
  - 主轨迹 S=50；种子 A–D = task1 的 W1–W4 的同一批初始噪声，即 seed_world(scene, 0, 1, k)，k=0..3。
- **场景**（先冻结）：e3ae803e、dd804295、fd316eb3。
- **A 单种子轨迹**（种子 A）：
  - 【你的修改】主行是每一步对最终世界的预测 Pred@i = x̂₀(i)，取检查点 1、2、5、10、20、35、50；
  - 状态 State@i = x_i 只作诊断行，标注"Intermediate flow states — not independent world samples"；早期解码为空或没有意义就如实记录。
- **B 多种子**：A–D 各自完成 50 步，只展示 A@50…D@50，这才是真正的多世界假设。
- **C 对比**（核心总结图）：第 1 行是同一种子的预测轨迹，第 2 行是 4 个种子的第 50 步。
- **诊断**：
  - D_traj(i) = D(Ŵ_A^(i), W_A^(50))，看一个假设如何收敛；
  - D_seed(i) = 同一步 4 个种子的平均两两分歧，看假设从哪一步开始分叉；
  - 表面体素数随步数的变化；
  - 同种子 S10↔S50 与不同种子 S50↔S50 的分歧对比，不预设结论；
  - D 的定义与 task1 图 2 相同：未观测区域，容差 τ=1，另报 τ=2，两个世界之间比较。
- **x̂₀ 的计算**：严格按 FlowMatchEulerDiscreteScheduler 的约定，x0 = sample(float32) − σ·v，与调度器自身的定义一致。
- **硬性检查**：
  - 最后一步的预测等于最终状态；
  - 中间某一步满足 Euler 关系，且 t = 1000σ；
  - (a) 观测与 task1 逐位相同；(b) 包装不影响结果；(c) S=10 的种子 A 复现 task1 的 W1；(d) 哈希可复现；
  - 灰色观测证据在所有面板中逐点相同。
- **完成后先汇报**，不直接进入下一个实验；S2 继续暂停。
- 【结果，2026-09-29】全部硬性检查通过。
  - 预测在第 13–18 步（σ ≈ 0.76–0.65）进入 D < 0.1；中间状态在 σ ≥ 0.59 时解码后只有一块地面。
  - 不同种子从第 1 步起就已分开：每个种子第 1 步的预测已更接近自己的最终世界。
  - 同种子 S10↔S50 的平均差异是不同种子的 0.39–0.58 倍，个别种子达到不同种子之间的水平。
  - 详见 `experiments/exp_vis/results.md`。

### task3 vis_stepcov：采样步数的影响随覆盖率如何变化（新；2026-09-29 复述后定稿，含你的调整）
- **目的**：在不同覆盖率下，同一个观测、同一个种子，只把采样步数从 10 换成 50，看世界、候选排序和 Top-1 变多少，以及这种变化相对于换种子有多大。exp2-task4 选 S=10 时只在 p=0.1 上验证过，而 NBV 主要工作在更低的覆盖率。
- **覆盖率档位**（p = 官方完整扫描中所用视角的比例；各档的相机集合是逐级包含的）：
  - L0：我们构造的单视角压力测试条件，不是官方的 p 档。规则：在 p=0.01 中能看到 patch 的相机里，取 patch 内有效深度命中最多的一台；相机编号在生成之前冻结进 `selection.json`。
  - L1：p=0.01，与 task1 的观测相同。
  - L2：p=0.05。
  - L3：p=0.1。
  - L4：p=0.2。
  - p=0.5 只作为可选的检查点，单独记录，不进主曲线和结论；p=1.0 不做。
  - 理由：所有核心指标都只在未观测区域里算，覆盖率太高时未观测区域太小，比例不稳定。
- **场景**：
  - 图只用 task1-A 的 4 个场景：e8150c6b、e3ae803e、fd316eb3、dd804295。
  - 曲线另加 8 个场景：从剩余的有效 episode（task1 选择池的 45 个中去掉这 4 个）里，用固定随机种子 2027 抽取，在任何生成之前写入 `selection.json`。
  - 曲线共用 12 个场景。
- **固定**：
  - S0-F，确定性执行；
  - 种子 A–D = seed_world(scene, 0, 1, k)，在所有档位和两种步数下都相同；
  - 【控制变量】每个场景在所有档位都用 task1 同一组 64 个候选。这是采样步数的敏感性分析，不是在每个覆盖率下重新构造闭环 NBV 状态；所以候选集、光线投射和 G_cov 全部固定。
- **指标**（只看未观测区域，τ=1）：
  1. 同种子世界分歧 D(S10, S50)；
  2. 候选 G_cov 排序的 Spearman；
  3. Top-1 一致率；
  4. S10 决策在 S50 世界下的损失 R = (G50(c*50) − G50(c*10)) / G50(c*50)；
  5. 另记 step/seed 比值 = 同种子 D(S10, S50) / 不同种子 D(S50, S50)。
  以上都同时报告"按世界"和"按 S2 平均分"两种。
- **曲线**：
  - 横轴是实测表面覆盖率，点旁标注 p；
  - 分歧、Spearman、损失：每个场景一条细线，外加总体平均；
  - Top-1 一致率：每个档位把场景×种子汇总成一致率，并给出 bootstrap 置信区间。
- **图**：
  - 图 1：`Observation | S=10 | S=50 | Pairwise Difference | GT`，固定用种子 A，每个档位一行。Difference 只画当前观测的未观测区域，τ=1，分三类：只有 S10 有、只有 S50 有、两者都有。**不沿用** task1 的 D=4p(1−p) 配色。
  - 总览：最低覆盖率档位下的 4 个场景。
  - 补充图：低、高覆盖率下 A–D 四个种子。
  - 每张图里 S=10 和 S=50 都并排出现。
- **硬性检查**：
  - (a) L1 的观测与 task1 逐位相同（另外 8 个场景与 S2 日志核对）；
  - (b) L1 在 S=10 下的世界 = task1 的 W1–W4，在 S=50 下 = task2 的最终世界；
  - (c) L1 在 S=10 下的 G_cov = task1 保存的值；
  - (d) 每一行的灰色证据逐点相同；
  - (e) 哈希可复现，新进程重跑逐位相同。
- **目录**：代码在 `experiments/exp_vis/vis_stepcov/`，输出在 `work_dirs/vis/vis_stepcov/`。S2 继续暂停。
- 【结果，2026-09-30】全部硬性检查通过。
  - 同种子 S10↔S50 的世界分歧在各覆盖率下都约为 0.2，step/seed 比值从 0.41 降到 0.34。
  - 候选排序相关 0.91–0.96，同种子 Top-1 一致率 0.77–0.90，S10 决策的损失 0.03–0.08；换种子分别是 0.75–0.88、0.50–0.68、0.13–0.19。
  - 低覆盖率下 S=10 没有明显变差，只有排序相关略低（0.91）。
  - 详见 `experiments/exp_vis/results.md`。

---

## exp3_geometry_vhd：几何假设与 VHD（2026-10-03 起，合并原 exp3/exp4/exp5 以及 pre_vhd、vhd_v1）

- **目的**：在冻结的 Seen2Scene 和与 S2 完全相同的 K=4 个世界下，检验"主动选择最能区分假设的观测位置"是否比"平均期望覆盖"更有价值（S2 = 平均掉不确定性，VHD = 主动消解不确定性）。
- **目录**：代码 `experiments/exp3_geometry_vhd/`（`README.md` 为本实验的说明），子目录按任务或算法版本：`pre_vhd/`（T1）、`vhd_v1/`（T2）、`vhd_v1_1/`（T3 起）；新输出在 `work_dirs/exp3_geometry_vhd/`。
- **任务**：T1 pre_vhd（done，PASS）；T2 vhd_v1（实现 PASS，结构性退化，不进入 dev100）；T3 结构性退化诊断（进行中）。

### T1 pre_vhd：exp2-task6 的事后诊断（进入 VHD 之前；2026-10-01 你的设计，复述后确认；done，PASS）

**原则**【你的要求】：不重新生成任何世界，也不根据结果调整任何 VHD 参数；只用已保存的 S1/S2 日志、候选打分、世界统计量和 GT 模拟器的量做描述性分析（均值、bootstrap 置信区间、图），不做优化、不搜阈值、不调参。完成后先停下汇报，再预注册并冻结 VHD-v1。

**数据**：dev100，episode 0，N=64，T=10。S1 来自 `work_dirs/exp2_baseline/task5_s1/dev100/s1_N64/`，S2 来自 `work_dirs/exp2_baseline/task6_s2/dev100/s2_N64/`。

**GT 一步收益** G_GT(c | s)：在观测状态 s 下执行候选 c（与 episode 执行步相同的传感器：pyrender 512²、uint16 量化、PatchFusion），冻结的表面覆盖率的增量。

#### D1 初始分歧诊断
- **只用第 1 步**：此时 S1 和 S2 的观测和状态完全相同；只用非退回 episode。
- **每个 episode 记录**：4 个世界中不同 Top-1 候选的个数 N_top1；两两 Spearman 均值；未观测表面状态不一致；S1 和 S2 是否选同一个候选。
- **主结果**：同一状态下的 ΔG_GT = G_GT(c_S2) − G_GT(c_S1)。
- **次要（只是关联）**：ΔAUC@10 = AUC_S2 − AUC_S1；执行第一步后两条轨迹已经不同，所以只作关联描述。
- **预先分组**：N_top1 = 1（4 个世界完全一致）与 N_top1 ≥ 2（至少两个不同的 Top-1）；看 S2−S1 的收益是否与世界分歧有关。不根据结果找阈值。

#### D2 离群偏好抑制诊断
- **范围**：已有 S2 规划状态的每一步（非退回步）。
- **每一步记录**：4 个世界各自的 Top-1 c₁*…c₄* 和 S2 平均分的选择 c*_S2，在同一状态下计算这些候选的 G_GT。
- **预先按 Top-1 票型分组**：4-0 / 3-1 / 2-2 / 2-1-1 / 1-1-1-1。
- **重点**：在 3-1 的情况下，S2 的平均是否经常避开单个世界偏好的、GT 收益较低的离群候选。没有 GT 证据时不叫 hallucination，只叫 **outlier preference**。

#### 复述后的确认和补充【你的修改，2026-10-01】
- **两项核心定义确认**：
  - G_GT 用闭环实际执行的 512² 深度渲染 + uint16 量化 + PatchFusion 后的表面覆盖增量，不用 Oracle 的 128² 命中代理；
  - D2 可以重跑 GT 模拟器评价未执行的候选，但不重新生成世界、不修改规划器，GT 不用于任何 VHD 参数选择。
- **(1) D1 分层**：N_top1 = 1 时第一步必然选同一个候选，ΔG_GT = 0 是数学必然，不作为多世界收益的证据。
  - 除全部 96 个 episode 外，在 N_top1 ≥ 2（49 个）和 S1 ≠ S2（19 个）中单独报告 ΔG_GT、更好/相同/更差的个数和相关性；
  - 分歧与收益的相关性同时给全样本和 N_top1 ≥ 2 子集，避免大量 0 值机械地影响解释。
- **(2) 命名**：D2 中第 1 步可以叫 S1（状态相同）；第 2 步以后统一叫 **k=0 single-world counterfactual on the S2 state**，不叫真实 S1（实际 S1 轨迹已经不同）。
- **(3) 收益同时保存整数和比例**：每个假设候选保存新增覆盖的 GT 表面体素数 ΔN_covered 和归一化的 ΔCov。更好/相同/更差一律按整数体素数判断，百分点只用于汇报，避免浮点误差。
- **(4) 假设执行不改变原状态（硬性检查）**：每个未执行候选都从同一个执行前状态的独立副本开始；评估完所有假设候选后，原状态的观测哈希、覆盖率和已观测体素统计必须与评估前完全相同，然后才执行真正的 S2 动作进入下一步。失败先停止。
- **(5) random-world-following advantage** = G_GT(c*_S2) − ¼ Σ_k G_GT(c_k*)：4 个世界等权，多个世界选同一候选时保留重复权重，不去重，含义是均匀随机相信一个世界。
- **(6) 解释边界**：D1/D2 只分析一步的 GT 效用和关联，用来理解 S2 为什么比 S1 稳定；不据此宣称长时程收益，不据此调整 VHD 的阈值、权重或其他设计；不计算任何候选级的 VHD 分数，VHD-v1 的定义留到诊断完成后单独预注册。
- **汇总表**：
  - D1：N_top1 = 1 / N_top1 ≥ 2 / S1 ≠ S2 三组，报告 N、动作改变数、ΔG_GT 均值、S2 更好/相同/更差；
  - D2：4-0 / 3-1 / 2-2 / 2-1-1 / 1-1-1-1，报告步数、random-world-following advantage、相对 k=0 counterfactual 的收益、相对 4 个 Top-1 中 GT 最好者的差距。
- **3-1 仍是 D2 的核心**：除交叉表外，重点区分离群候选的 GT 收益更低、相同、更高三种情况；即使离群候选经常更好也照实报告（这对区分"平均掉不确定性"和"主动消解不确定性"同样重要）。
- 原硬性检查和 bootstrap 方案全部保留。完成后停下汇报，不修改原 exp4/exp5，不进入 VHD。
- **实现**：代码 `experiments/exp3_geometry_vhd/pre_vhd/`，输出 `work_dirs/pre_vhd/`，结果 `experiments/exp3_geometry_vhd/pre_vhd/results.md`。D1 第 1 步的整数收益取自 D2 的模拟（并与两份日志的覆盖率核对）；不可变性检查对全部 episode 的每一步都做。
- 【结果，2026-10-01】全部硬性检查通过（1000 步重建与日志逐位一致，假设执行不可变性 1000/1000，新进程重跑逐位相同）。D1：S1 ≠ S2 的 19 个 episode 中 S2 14 个更好、5 个更差（+2.63 pp，置信区间刚好跨 0）；N_top1 ≥ 2 为 +1.02 pp [+0.01, +2.13]；分歧大小与收益差在全样本和 N_top1 ≥ 2 子集中都没有可见关系。D2：random-world-following advantage 全部步 +0.34 pp [+0.23, +0.47]，票型越分散越大；3-1 时 S2 90% 跟随多数，离群偏好 63% 更差、37% 更好，避开它平均 +1.24 pp；S2 相对 4 个 Top-1 中 GT 最好者平均少 0.99 pp。详见 `experiments/exp3_geometry_vhd/pre_vhd/results.md`。
- 【你的决定，2026-10-01】D1/D2 通过，pre_vhd 结束，不再做事后挖掘。由此定下：VHD 不直接用全局 3D 分歧、Top-1 个数或多数强度打分；不用多数投票，也不预先给少数世界降权；0.99 pp 的差距只作背景事实。

### T2 vhd_v1：闭环 VHD-v1（预注册 → 冒烟测试 → dev100；2026-10-02 冻结为 v1）

- **预注册全文**：`experiments/exp3_geometry_vhd/vhd_v1/preregistration.md`。确认前不运行任何东西。
- **与 S2 完全相同**【你的要求】：K=4 个世界、世界种子、N=64 个候选、128² 光线投射、patch 外遮挡、episode 状态、模拟器和评测；**唯一改变的是规划分数**。
- **分数**：只用以候选为条件的未来观测分歧，即每个世界从候选相机投射得到的命中/未命中和首次命中深度在 4 个世界之间的分歧。沿用 A3 在 D1/D2 之前冻结的逐光线定义（10 cm 饱和），候选分数的分母固定为整个光线预算（128²），不合法、被 patch 外遮挡的光线贡献 0。
- **不做**：不用 GT、不训练、不加语义、不做 beam search、不学习世界权重、不与期望覆盖加权组合、不调 λ，也不用 D1/D2 的结果调任何阈值。
- **之后**：冒烟测试 10–20 个 episode（只查实现正确性、分数分布、退化视角、退回、确定性、GT 泄漏、与 S2 的世界等价和运行时间，不按 AUC/覆盖率调算法）→ 冻结版本跑 dev100；**最关键的比较是 VHD vs S2**。
- 原 exp4 中 K=8、dev50 的开环相关性研究和止损规则暂不恢复，预注册写完后再决定是否还有必要。
- 【你的决定，2026-10-02】
  - **目的**：不是再验证多个世界是否有用（S2 已回答），而是检验在相同的 K=4 个假设和计算预算下，主动选择最能区分假设的观测位置，是否比平均期望覆盖更有价值。S2 = 平均掉不确定性，VHD = 主动消解不确定性。
  - **主掩码选 B**：正式分数 G_VHD(c) = (1/128²) Σ_r M_valid·M_novel·c(r)，M_novel 严格复用冻结的 G_cov 新表面规则（不用 GT、不加阈值）；选项 A 记为 VHD_raw，每步保存，只作诊断；运行后不能在 A/B 之间切换。
  - **全 0 时**用冻结 Random 在该步的选择（`vhd_zero_fallback`），不退回 S2；与 world_invalid_fallback 分开记录。
  - **冒烟测试**固定用 `dev_patches.csv` 前 16 个 episode，只检查实现（单元测试、与 A3 等价、与 S2 世界等价、GT 泄漏、确定性、分数分布、退回、选中视角的合法/新观测光线比例、运行时间），不输出 VHD vs S2 的 AUC/覆盖率比较。
  - 10 cm、K=4、分母 16384、新观测掩码、argmax 平局规则、Random 全 0 退回在运行前冻结；除实现错误、NaN、非法光线/掩码、结构性退化外不改，必须改时升级为 v1.1 重新预注册。
- 【冒烟测试结果，2026-10-03】实现检查全部通过（单元测试、与 A3 等价、与 S2 世界等价、S2 未改动、GT 泄漏、确定性）；全 0 退回 0 次；每步 80.1 s，与 S2 相同。发现结构性问题：在 60 个高分决策中有 21 个，至少一个世界在相机位置附近生成了几何（一半以上命中在 0.3 m 以内），而真实位姿是自由的。按规则需要你决定是否升级 v1.1。详见 `experiments/exp3_geometry_vhd/vhd_v1/results.md`。

### T3 结构性退化诊断（2026-10-03 你的定义；done，PASS）
- **目标**：找到"生成几何包住合法相机位姿"的最小、不用 GT、可预注册的修复规则。**不实现 v1.1，不跑 dev100，不看 AUC、覆盖率或 GT 收益。**
- **数据**：VHD-v1 冒烟测试的状态（`work_dirs/vhd_v1/smoke/`），世界按同样的种子重新生成。
- **比较的门控**：
  1. **V1 基线**：当前分数，不加一致性条件；
  2. **观测自由一致性**：生成的表面与当前融合观测中明确的 FREE/EMPTY 证据冲突时，不奖励这部分分歧；
  3. **候选可行性一致性**：某个生成世界违反动作空间已冻结的 0.1 m 间隙约束时，这个世界在这个候选上记为不一致；
  4. **并集**：同时用 2 和 3。
- **0.3 m** 继续只作为已预注册的病态诊断阈值，不作为 v1.1 的规划阈值（候选生成器原本提供的自由空间约束是 0.1 m）。
- **只报告**：
  - 21 个已识别的病态高分状态中，各门控消除或降低了多少；
  - 34 个正常高分状态是否被保留；
  - 候选分数和排序的变化；
  - 全 0 和退回的比例；
  - 有效世界数和有效世界对数；
  - 是否出现新的结构性退化。
- **待诊断的 v1.1 形式**（不冻结）：每个候选、每个世界一个一致性门 q_k(c) ∈ {0,1}，c_cons(r) = (1/6) Σ_{i<j} q_i(c) q_j(c) d_ij(r)，仍固定除以 6，不因有效世界数减少而重新归一化。
- 完成后停下汇报；再单独决定 VHD-v1.1 的最终定义和预注册，然后重新跑冒烟测试。VHD-v1 的结果完整保留，不覆盖。
- 【实现说明，2026-10-03，运行前写定】
  - **状态**：VHD-v1 冒烟测试中全部 158 个非退回决策状态。按 VHD-v1 日志重建状态，用同样的种子重新生成 4 个世界；V1 的 64 个分数必须与日志完全相同。
    - 全 0 比例、分数分布、有效世界数等在全部 158 个状态上统计；
    - 病态和正常的分组沿用 0.3 m 诊断：21 个病态、5 个中间、34 个正常。
  - **门控 2（观测自由）**：逐光线判断，不引入阈值。世界 k 在光线 r 上的首次命中体素在当前融合观测中是 EMPTY，就记为冲突。只要两个世界中有一个冲突，这对世界在这条光线上的分歧就不计。
  - **门控 3（候选可行性）**：逐候选、逐世界判断。用与候选生成器完全相同的 10×10 探测光线（60° 视场，像素中心）投进世界 k；只要有一个首次命中的 z 深度小于 0.1 m，就记 q_k(c) = 0。patch 外是真实场景，候选生成器已经检查过。
  - **统一形式**：c_g(r) = (1/6) Σ_{i<j} w_i(r) w_j(r) d_ij(r)，其中 w_k(r) = q_k(c)·[世界 k 在 r 上不冲突]，固定除以 6；分数 = (1/16384) Σ_r M_valid·M_novel^g·c_g(r)。
    - M_novel^g 只看仍参与计算的世界（w_k(r) = 1）是否有新表面命中；另外记录"沿用 V1 新观测掩码"的版本作为敏感性分析。
  - **病态判断**：对每个门控选出的 argmax，用 0.3 m 诊断判断（某个仍参与计算的世界有一半以上的命中在 0.3 m 以内）；另外报告"所有世界"的版本。
  - 不看 AUC、覆盖率或 GT 收益。代码 `experiments/exp3_geometry_vhd/vhd_v1_1/t3_*.py`，输出 `work_dirs/exp3_geometry_vhd/vhd_v1_1/t3_degeneracy/`。
- 【结果，2026-10-03】158 个状态的 V1 分数都与日志完全相同。观测自由门控几乎不起作用（病态 21 → 20）；0.1 m 候选可行性门控消除了 9/21 个病态，全部状态中病态选择从 30 降到 15，34/34 个正常状态不变，0.39% 的世界 × 候选被标记，没有全 0 退回，也没有新的退化；并集与之效果相同。剩下的病态来自相机前 0.10–0.21 m 的生成表面，冻结的 0.1 m 约束不排除它们。详见 `experiments/exp3_geometry_vhd/vhd_v1_1/results.md`。
- 【你的决定，2026-10-03】T3 通过，不再继续搜索额外的门控。VHD-v1.1 只加入已冻结的 0.1 m 候选可行性一致性门；不采用并集（只多减少 1 个退化选择，却改变更多 argmax）；剩下的 0.1–0.3 m 近场情况不由 v1.1 处理（0.3 m 只作诊断，不得事后改成硬门控），若仍影响规划效用，属于 VHD-v2。

### T4 VHD-v1.1 预注册与冒烟测试（2026-10-03 冻结为 v1.1；冒烟测试 done，PASS）
- **预注册全文**：`experiments/exp3_geometry_vhd/vhd_v1_1/preregistration.md`。VHD-v1 保持完全不动，两个版本独立保存；VHD-v1 文件的校验和见 `work_dirs/exp3_geometry_vhd/vhd_v1_1/vhd_v1_checksums_2026-10-03.txt`。
- **唯一改动**：c_v1.1(r) = (1/6) Σ_{i<j} q_i(c)·q_j(c)·d_ij(r)，分母固定为 6；G = (1/16384) Σ M_valid·M_novel·c_v1.1，新观测掩码与 v1 相同；其余全部与 VHD-v1 相同。
- **两个 0.1 m 要区分**：τ_sat = 0.10 m 是深度分歧的饱和尺度；δ_clear = 0.1 m 是 EXP1 冻结的候选间隙。
- **如实说明**：候选可行性模块构造候选时使用了场景几何（对所有规划器共享，EXP1 冻结）；v1.1 用的是这个动作空间规定，不是 GT 效用、覆盖或 Oracle 分数，不得写成"完全只用传感器的证据"。
- **核对结果**：原规则和 T3 用的都是相机 z 深度，探测光线也相同。差异只有一处：原渲染器有 0.01 m 的近裁剪面，T3 没有。T3 标记的 157 个"世界 × 候选"中有 9 个的最近命中 < 0.01 m。冻结前需要你在 A（复现近裁剪面）和 B（T3 现状）之间选择。
- 【你的决定，2026-10-03】预注册确认并冻结；近裁剪面冻结为 **A**：z_near = 0.01 m 按相机 z 深度定义（t_near = 0.01 / z_factor），从 max(t_entry, t_near) 开始找首次命中，z_min < 0.1 m 时 q_k(c) = 0。B 只在冒烟测试中作为 T3 复现诊断记录。T3 仍为 PASS，但 157/40448、30 → 15 等数字是 B 语义下的结果，不能提前归到 A 版本。冒烟测试只检查实现和结构，只有 A 参与选择，完成后停下，不自动进入 dev100。
- 【冒烟测试结果，2026-10-03】全部检查通过：18 项单元测试；VHD-v1 校验和不变；第 1 步与 S2、VHD-v1、T3（B 版）逐位相同；新进程重跑逐位相同；GT 毒化后规划字段完全相同。全 0 退回 0 次；q_A 触发 0.38%，A 与 B 的 argmax 0 步不同。0.3 m 诊断判为病态的选择有 19/158 个，它们的近场世界的探测深度全部在 0.100–0.243 m，即预注册写明不处理的近场情况。没有发现新的方法级退化。每步 81.4 s。详见 `experiments/exp3_geometry_vhd/vhd_v1_1/results.md`。

### T5 冻结版本的 dev100 评测（2026-10-03 你的定义；done 2026-10-04）
- **前提**【你的决定】：VHD-v1.1 冒烟测试 PASS；进入 dev100 前不允许再改 v1.1 的定义。
- **运行**：VHD-v1（冻结的纯未来观测分歧，`nbv_run --planners vhd`）和 VHD-v1.1（冻结的可行性一致 VHD，`vhd_v1_1/run_v11.py`）。
  - 与 S1/S2 相同的设置：dev100、episode 0、N=64、T=10，同样的模拟器、候选生成、种子（与 S2 相同的 world seeding）、评测和退回规则；
  - 运行期间不修改 VHD-v1、VHD-v1.1、Seen2Scene、候选生成、模拟器或评测；
  - 100 个 episode 全部从头跑；前 16 个必须与各自的冒烟测试逐位相同，作为确定性检查。
  - 4 张卡，每张卡先跑 v1.1 的 25 个 episode，再跑 v1 的 25 个，约 12 小时。输出在 `work_dirs/exp3_geometry_vhd/{vhd_v1,vhd_v1_1}/dev100/`。
- **主比较（预注册，唯一）**：Δ_primary = AUC@10(VHD-v1.1) − AUC@10(S2)。
  - 沿用已冻结的 bootstrap：按 episode 配对，1 万次，种子 0，95% 百分位置信区间；
  - 只有置信区间完全大于 0，才算 VHD-v1.1 优于 S2。
- **次要比较**：
  - VHD-v1 − S2 和 VHD-v1.1 − VHD-v1，作为一个比较族，做 Holm 校正；
  - 规划器主表：按两版预注册，各自对 S1、Frontier、Random 做比较，每个版本一个比较族（3 项），做 Holm 校正。
  - p 值沿用双侧 bootstrap p。
- **报告**：
  - AUC@10、AUC@5、第 10 步表面覆盖；自由空间覆盖仍作辅助指标；辅助指标只给置信区间；
  - 每步运行时间、退回次数（world_invalid 和 vhd_zero 分开）；
  - 已冻结的诊断量：v1.1 的 q 触发率、A 与 B 的差异、0.3 m 诊断频率、与同样世界下 S2 选择的一致率；
  - 健全性：没有规划器高于 Oracle。
- **约束**：
  - 0.3 m 近场诊断只作诊断，不根据它在 dev100 上的频率改规划器；
  - 看到 dev100 结果后，不再引入新的门控、阈值、权重项、与 G_cov 的混合或新的 VHD 变体；
  - 如果 VHD-v1.1 表现不如预期，如实记录，之后作为新版本或新实验诊断，不事后修改 v1.1。
- **顺序**：两个运行都完成后，先出冻结的定量表和配对差异，然后才看轨迹层面的失败案例。
- 【结果，2026-10-04】**主比较 VHD-v1.1 − S2 = −0.183 [−0.202, −0.163]**，100 个 episode 中 98 个更差，VHD-v1.1 不优于 S2。AUC@10：VHD-v1.1 0.432，VHD-v1 0.417，Random 0.435，S2 0.615。v1.1 − v1 = +0.015（显著）；两个版本与 Random 都没有显著差别。失败案例：VHD 的分数与实际覆盖增量弱负相关（S2 为 +0.66）；第 1 步状态和世界都与 S2 相同时，VHD 选择只带来 5.0 pp 的新表面（S2 12.7 pp）。按预注册如实记录，不事后修改。详见 `experiments/exp3_geometry_vhd/vhd_v1_1/results.md`。
- 【你的决定，2026-10-08】接受 T5 的结论，exp3 结束。3D-FRONT 上 S2 > S1 ≫ VHD；VHD-v1/v1.1 保留为分析项（在 EXP3b 中作为跨数据域的失败分析，只跑小子集），不再为 VHD 投入新版本；主线转向多世界期望覆盖 S2，以及更好的世界模型（EXP4 一致性、EXP5 Active Geometry）。

### 合并前原 exp3/exp4/exp5 的设计（2026-09-26，保留作背景，文中编号是原编号）

#### 原 exp3_geometry_hypotheses：官方模型能否给出有意义的多个世界（W2，约 60 GPU 时）

##### task1 多样本的多样性（原 task7，并入原 task6 剩余部分）
- 固定输入：p ∈ {0.05, 0.1}；K ∈ {2, 4, 8, 16}。
- 指标：
  - 未观测区域的多样性：TMD、两两差异、逐体素"是否为表面"的熵；
  - 已观测区域的一致性，以及观测违反率；
  - 逐体素校准：K=16 时的 ECE 和 NLL；逐体素熵预测误差的 AUROC。
- 分组分析：物体框内 / 框外，因为官方模型把框内未观测的表面当成"空"来学；按离已观测区域的距离分组。
- 从原 task6 并入的两项对照：
  - 主干用原始权重 vs EMA 权重，检查训练和推理不一致的影响；
  - 部分扫描编码用采样 vs 后验均值。
- **止损 1**：如果未观测区域的多样性接近 0，也就是 K 个样本几乎相同，VHD 主线暂停，需要讨论。

##### task2 可视化（原 task8）
- 同一个部分扫描：K=8 个世界的网格，以及逐体素分歧热图。挑 10 个 dev patch，一部分用于论文的 teaser。

---

#### 原 exp4_geometry_vhd：分歧能否预测真实收益（W2，约 40 GPU 时）

这是**核心止损实验**，只用官方模型，完全不训练。

- **设置**：dev50；初始观测为官方 p_0.05 和 p_0.1；每个 patch 64 个候选视角；K=8，S 取 exp2-task4 定下的值。
- **基本的一致性处理**：只统计首次命中点位于未观测区域、并且在观测边界 2 个体素之外的光线，避免已观测区域的违反制造假分歧。完整的一致性研究放在 exp6。

##### task1 视角渲染（原 task9）
- 用 exp1-task4 对 K 个世界 × 64 个候选视角做光线投射，结果缓存下来，供 task2–4 共用。

##### task2 VHD 各变体和基线评分（原 task10）
- **VHD 变体**：
  - 命中熵；
  - 深度方差；
  - 深度区间熵（区间宽 5 cm）；
  - 命中熵 + 深度区间熵，这是 VHD-v1。
- **对照评分**：
  - B2 Frontier；
  - B3 体积信息增益：未知体素计数和平均熵两种；
  - B4 占据熵：在世界空间里对可见体素的逐体素熵求和，不做视角条件，用来说明视角条件化的作用；
  - B5 单假设覆盖，即 exp2 的 S1；
  - B6 期望覆盖，即 exp2 的 S2。

##### task3 真实收益（原 task11）
- 用 GT 网格从每个候选视角渲染，得到这个视角实际能新看到的内容：
  - 新增 GT 表面体素，这是主指标；
  - 新增的空间体素；
  - 几何覆盖增益。

##### task4 相关性和止损判断（原 task12）
- 指标：每个 patch 内，各评分与真实收益的 Spearman ρ、NDCG@5、top-1 regret，再对 50 个 patch 做 bootstrap。
- 附加分析：K ∈ {2, 4, 8, 16} 和 S 对 ρ 的影响，用来定下 NBV 的 (K, S)。
- **止损 2**：VHD-v1 的 ρ 必须显著高于 B6（S2）、B5（S1）和 B4，即配对置信区间不跨 0。
  - 如果只打不过 B6，主线改为"生成式期望覆盖"，VHD 降为分析项；
  - 如果连 B3 都打不过，暂停，需要讨论。

---

#### 原 exp5_closed_loop_nbv：闭环主动建图（W3，约 120 GPU 时）

exp4-task4 的相关性只是单步指标，论文需要闭环的结果。Random、Frontier、S1、S2、Oracle 的闭环结果已经在 exp2 中得到，这里直接复用。

##### task1 补充的闭环基线（原 task14）
- 在 dev100 上补跑 exp2 中没有的 B3（体积信息增益）和 B4（占据熵）。
- 主指标：几何覆盖率 AUC（T 由 exp1-task7 定下）。
- 其他指标：最终覆盖率、达到 80% 覆盖所需步数、分歧下降量、决策延迟（分为采样、渲染、评分三部分）。

##### task2 闭环 VHD 和权重选择（原 task15）
- 在 dev50 上网格搜索 Score = β·VHD + γ·覆盖，β ∈ {0.5, 1}，γ ∈ {0, 0.25, 0.5}。
- 再在 dev100 上与 exp2 的表 B 以及 task1 比较，结果填入表 B 的 VHD 一行。
- 通过标准：VHD 的覆盖率 AUC 显著高于 S1 和 S2。
- 预算估计：单次决策约 40–76 s（K=4、S=10），一个 episode 约 7–13 分钟。

---

## exp3b_macaronspp_transfer：EXP3b — MACARONS++ Transfer Baseline（preregistration v1.1 冻结；v1.2 T2-repair 修正案草稿待你确认）

- **完整预注册**：[preregistration.md](exp3b_macaronspp_transfer/preregistration.md)（每个 task 的步骤、参数、阈值、判定和 stop rule 都在这里，本节只写概要）
- **MAGICIAN 协议核对**：[magician_protocol_facts.md](exp3b_macaronspp_transfer/magician_protocol_facts.md)
- **数据**：[MACARONS++ dataset](../../MAGICIAN/data/Macarons++/)，即 `/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/data/Macarons++`（15 个场景，58 GB）。本仓库 `data/` 下只有 3D-FRONT，没有 `data/macaronspp/`；目录不移动、不复制、不重命名。
- **路径配置**【v1】：EXP3b 的代码只通过 [exp3b_paths.py](exp3b_macaronspp_transfer/exp3b_paths.py) 读取数据路径，环境变量 `MACARONS_PLUSPLUS_ROOT` 优先（另有 `MAGICIAN_ROOT`、`MAGICIAN_PYTHON`）；其他脚本不再硬编码绝对路径。

**执行规则**【你的要求，2026-10-08】：每个 task 单独审批。流程为：预注册 → 你确认 → 执行这一个 task → 保存结果 → 更新 results.md 和 README → 完整汇报 → **STOP**。只有你明确回复后才进入下一个 task。【v1】子任务同样单独审批：T4a、T4b、T4c 各自汇报并 STOP，T7a、T7b 也分开审批。不允许事后修改阈值、尺度、窗口、指标或协议；未经你确认不跑 full benchmark。

**研究问题**：
- **RQ1 World transfer**：Seen2Scene 的世界假设能否从 3D-FRONT 零样本迁移到 MACARONS++？
- **RQ2 Planner transfer**：在与 3D-FRONT 相同的 R 协议下，S2 是否仍有规划信号，并保持相对 S1 的多世界优势？
- **RQ3 MAGICIAN benchmark**：能否把我们的生成式规划器接入 MAGICIAN 原生的 M 协议，并与 Random、Frontier、MACARONS、MAGICIAN-greedy、完整 MAGICIAN 在同一个设了种子的评测循环中公平比较？

主线为 S2 → Consistency-S2 → ActiveGeometry-S2；VHD-v1/v1.1 只作分析和迁移诊断。

**两个协议**：
- **R protocol（Regional / transfer）= our pipeline + MACARONS++ dataset**。使用 EXP1/EXP2 冻结的主动建图流程，换到 MACARONS++ 上运行【协议澄清 2026-10-09：模型权重、规划器逻辑、候选/动作协议、评测和方法超参数冻结；MACARONS++ adapter 使用经过独立验证的目标渲染器 sample-0 射线语义；不声称与旧 3D-FRONT 传感器在每个光栅化细节上逐位相同】：
  `MACARONS++ scene → our patch extraction/simulator → partial TSDF → Seen2Scene → W1…WK → our 64 candidate views → S1/S2/VHD → execute → fuse → repeat 10 steps`
  它的作用是跨数据域迁移，不与 MAGICIAN 直接比较。
- **M protocol（MAGICIAN native benchmark）= MAGICIAN benchmark pipeline + the method being evaluated**。使用 MAGICIAN 的全场景探索、相邻动作空间、100 步、传感器和覆盖率。
  - **M 不等于 MAGICIAN 的规划器**：S1-M/S2-M = MAGICIAN 环境 + 我们的 Seen2Scene 世界模型 + 我们的 S1/S2 评分；MAGICIAN 基线 = MAGICIAN 环境 + MAGICIAN 规划器。
  - 它的作用是最终的公平基准比较。我们的世界通过 E-M2 嵌入：每个场景一个尺度，单层窗口铺满场景框，每 10 步重新生成一次。
  - 【v1.1 冻结，2026-10-08 你的决定】
    - **传感器信息 = P1 分层对等**：表面/BAND 只来自 MAGICIAN 这一帧的同一组 10% 样本；同一帧的完整深度只写 EMPTY，最多融合到“深度 − 截断”。论文中要披露：MAGICIAN 在约 80 万个 proxy 点上查询，我们在 TSDF 体素和光线上处理，原始观测来源相同。P2 不采用，P3 永久排除。
    - **共享观测采样**：GT 点、每帧 10% 样本、每格上限对所有方法、所有种子相同，不允许各自独立抽样。
    - **代码状态 = MAGICIAN `9521d69` 的干净 worktree**，不带未提交的 Beta-head 修改；upstream 源码不改。加载失败或复现差 > ±0.03 → STOP / NEED-DISCUSSION，不切回带修改的工作区。

**数据范围**：

| 分组 | 场景 | 用在哪里 |
|---|---|---|
| R gate（MACARONS++ 的 dev） | sestino_museum、barts、pantheon、eiffel、colosseum | T2–T4，以及 T5 的开环状态、T7a |
| R held-out | 其余 9 个测试场景 | EXP3b 中不用，只在 exp7 跑一次 |
| M 协议（MAGICIAN 测试场景） | 14 个（不含 sofia_church） | T6、T7 |
| train-overlap-8 | dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | MACARONS/MAGICIAN 权重的训练场景，补充报告 |
| **Unseen-6** | alhambra、neuschwanstein（MACARONS 验证集）、bridge、eiffel（MACARONS 测试集）、barts、sestino_museum（不在 MACARONS 数据中） | M 协议主报告组之一（另一个是 All-14） |
| 不用 | sofia_church | MAGICIAN 不测，loader 有 bug |

- **协议卡**（T1 产出）：[protocol_card.md](exp3b_macaronspp_transfer/protocol_card.md)；结果：[results.md](exp3b_macaronspp_transfer/results.md)

**Tasks**（逐个审批；阈值都在运行前固定）：

| Task | 内容 | 主要判定 | STOP 条件 | 状态 |
|---|---|---|---|---|
| T1 Protocol audit | MAGICIAN / R / M 三列的完整协议表；冻结 M 协议的报告方式和 E-M2 窗口表；记录 git 状态和 settings 的 md5；**blocking item：核清 M 协议的深度信息流**（交给规划器的观测 / 地图更新用的观测 / 只用于评测的点） | M 协议依赖的每一项都有代码依据；深度信息流能按预注册 §6 的规则得出结论 | 有一项无法确定，或深度信息流不属于规则中的任何一种情况 → NEED-DISCUSSION（传感器信息假设不冻结） | **done，PASS（2026-10-08）**：最初为 NEED-DISCUSSION；你决定 P1 分层对等 + `9521d69` 干净 worktree + 共享观测采样后改判 PASS。见 `results.md` |
| T2 Adapter validation | 相机（对照 MAGICIAN 渲染器）、深度、坐标、反投影、尺度、TSDF、确定性、3D-FRONT 回归；冻结 40 个 gate patch | C1–C6 全部通过 | C1–C5 有一项不过 → FAIL | **✅ PASS after preregistered repair（2026-10-09）**：as-run = FAIL（保留，未改写）；T2-repair-v1.2 = PASS（R3：C1-geometry、C2a/C2b、C4 五个场景全过，C5、C6 PASS）。as-run 记录：C1、C2、C4 不过（pyrender 的 4×MSAA 深度不在像素中心；C4 MC 检查有实现 bug）；C3、C5、C6 通过；40 个 patch 已冻结。T2-repair 修正案 **v1.2 已冻结**（全链路统一 MSAA sample-0 射线：传感器、融合、参考、Oracle、区域外遮挡、S1/S2、VHD、Frontier；I1 作用域内替换；C1 拆分；C4 observed-cell 过滤；阈值不变），见 preregistration §7；R0 PASS、R1 pilot PASS → R2 冻结 → **R3 PASS**；R4/R4-b 待批准 |
| T3 World transfer sanity | 5 个 patch 生成 W1–W4：推理成功、观测复现、自由空间违反、世界多样性、初始覆盖、定性图 | 观测复现中位数 ≥ 0.786；违反中位数 ≤ 0.0786；结构占比 ≥ 0.2；初始覆盖中位数 ≤ 0.508（都来自冻结的 3D-FRONT 参考） | 推理失败，或 ≥ 2 个 patch 不达标 → FAIL | **done，PASS（2026-10-09）**：5/5 推理成功；(b) 中位 0.868–1.000，(c) 0.0000–0.0147，(d) 0.42–0.68，(e) 40 个 episode 初始覆盖中位 0.347；只描述：3D-FRONT 地面先验伪影明显；见 `results.md` |
| T4a MACARONS++-R 框架 | 40 个 episode：Random / Frontier / Oracle | 排序、余量、不饱和、确定性 | 完成后 STOP；任何一项不过 → FAIL | **done，❌ FAIL（2026-10-09）**：gate 3 饱和（Oracle Cov@3 0.925 > 0.90）；gate 1、2、4 PASS；gate 5 = 6；完整性全部通过。STOP，T4b 未运行。见 `results.md` |
| T4b MACARONS++-R 规划器 | 40 个 episode：S1 / S2 | S2 − Random 下界 > 0（排除灾难性的迁移失败）；不要求 S2 − S1 显著 | 完成后 STOP；S2 − Random 下界 ≤ 0 → FAIL；S2 − S1 上界 < 0 → NEED-DISCUSSION | 待确认 |
| T4c VHD 迁移诊断 | 20 个 episode：VHD-v1 / v1.1，只作描述 | 预先写定的分类规则 | 完成后 STOP | ✅ 完成（2026-10-10；VALID；分类规则不满足） |
| T5 M-protocol feasibility | 设了种子的测试循环和规划服务；E-M2 后端只按冻结的复现标准选择（只有一个达标就用它；都达标用官方滑窗；都不达标就 STOP）；75 个开环状态上的排序和遗憾值；陈旧程度（只描述） | 测试循环检查通过；世界在 ≥ 4/5 个场景可用；S2-M 的遗憾值优于 Random | 不满足 → STOP，不事后搜索尺度或窗口；两个后端都不达标 → STOP / NEED-DISCUSSION | 待确认 |
| T6 M-protocol native baselines | Random walk、Frontier-M、MACARONS（修复后）、MAGICIAN-greedy、MAGICIAN（各 3 个种子）、Oracle-greedy | 全部跑完；MAGICIAN 复现差 ≤ 0.03 | 复现差 > 0.03 → NEED-DISCUSSION | 待确认 |
| T7 S1-M / S2-M | T7a：gate 5 个场景 × 5 个起点；T7b：其余 9 个场景（full benchmark，需你单独批准）。主比较：S2-M − S1-M、S2-M − MAGICIAN-greedy、S2-M − full MAGICIAN，分 All-14 和 Unseen-6 报告 | 按预注册比较 | T7a 跑不完或不能复现 → STOP；T7b 没有你的批准 → 不运行 | 待确认 |

**成本**（估计）：T1 不用 GPU；T2 约 0.5；T3 约 0.2；T4 约 20；T5 约 19；T6 约 85；T7a 约 47；T7b 约 143 GPU 时。

---

## exp4_consistency（原 exp6）：观测一致性（W3，约 30 GPU 时，不需要训练）

在官方模型上做，结论之后也用于 exp5 训练出的模型。

### task1 TSDF 写回（原 task16）
- 解码之后按权重融合，已观测部分的权重占主导。

### task2 沿光线 carving（原 task17）
- 沿真实相机光线，把第一个表面之前的区域强制设为空，并去掉小的离散碎片。

### task3 观测边界排除宽度（原 task18）
- 计算分歧时排除观测边界 0 / 2 / 3 个体素。

### task4 RePaint 式替换（原 task19，可选）
- σ < 0.5 时，对完全观测的格子写回编码后的观测潜变量。

**各 task 共同记录**：观测违反率、未观测区域质量、多样性、exp3（原 exp4-task4） 的 ρ、exp3（原 exp5-task2） 的闭环 AUC。据此选出默认的一致性配置。

---

## exp5_active_geometry（原 exp7）：面向主动建图的几何条件分支 GeoControlNet（W3–W5，约 500 GPU 时）

**前提**：exp3（原 exp4） 通过了止损。这一阶段仍然不加语义。

### task1 训练数据（原 task20）
- 数据来自 train_ours，观测来源按比例混合：
  - 70% 取官方 p ∈ {0.01, 0.05, 0.1, 0.2}；
  - 30% 取模拟器（exp1-task6）生成的 NBV 式序列：1–10 个视角，一半随机选、一半按覆盖贪心选，每个场景 8 条，事先生成并缓存。
- 每个格子计算 M_obs = 已观测体素的比例，取值 0–1。
- 输出：缓存好的训练 patch，并统计 M_obs 和观测比例的分布。

### task2 GeoControlNet 训练（原 task21）
- **结构**：从官方分支初始化；新增零初始化的 `Linear(9 → 768)`，输入 `[z_geo_obs, M_obs]`，走分支的 `embed` 加法路径；原来的 `input_zero_proj` 保留。
- **训练设置**：
  - 主干冻结，使用 EMA 权重：`copy_to` 放在 `controlnet.py:108` 的深拷贝之前，以及 `maybe_reload_base` 之后；`gen_ckpt` 固定不变，并记录 sha；
  - 分支维护自己的 EMA（0.999）；
  - `cond_drop_prob=1.0`，不看布局；
  - `progressive_end=0`，否则会丢掉墙和地面的 token。
- **损失**：几何 FM，未观测格的权重为 0.7。
- **训练规模**：30–60k 步，全局 batch 8–16；新层学习率 1e-4，分支主体 2e-5；每 2k 步在 dev50 上评测。节点 B 约 3–4 天，先在 W3 实测吞吐量再定步数。
- **对照**：只加多观测比例数据、不加 M_obs 的版本，节点 A 同时训练，用来区分"数据的作用"和"M_obs 的作用"。

### task3 评测 GeoControlNet vs 官方（原 task22）
- 补全质量：exp2-task3 的指标，按 p 和模拟序列分组。
- 校准：exp3（原 exp3-task1） 的指标。
- VHD 相关性：exp3（原 exp4-task4） 的指标。
- 闭环：exp3（原 exp5-task2） 的指标。
- **止损 3**：以上都没有提升时，保留官方模型，GeoControlNet 只作为消融。

---

## exp6_semantics（原 exp8）：语义扩展（W4–W5，约 120 GPU 时）

**task1 在 W2 就用 CPU 开始**，不阻塞前面的实验。

### task1 包围盒语义标签（原 task23）
- meta 中的类别 id 是 old_id，经 `front3d_mapping.csv` 转换，再合并为约 12 个室内类。类别表放在配置里，不写死在代码中。
- 标注规则：
  - 落在多个框内的体素，取体积最小的框；
  - 框外的体素按法向和高度划为 floor、wall、ceiling；
  - 同时输出每个面的标签、每个格子的类别直方图，以及"干净掩码"（只落在一个框内，或明确属于结构类）。
- 人工核查 20 个 patch。通过标准：物体体素错误率 < 10%，歧义率 < 5%。

### task2 探针（原 task24）
- 冻结官方主干和分支，取它们的特征，训练 MLP 探针预测未观测格的类别，在 σ ∈ {0.3, 0.6, 0.9} 下分别测。
- 同时计算格级标签映射回体素级的 mIoU 上限。

### task3 事后语义预测器（原 task25，语义 MVP）
- 在生成的几何潜变量上跑一个小型稀疏 Transformer（6 层、384 维），输出逐格类别直方图。
- 训练数据：train_ours 的 GT 潜变量，加噪声增强，用来模拟生成的潜变量。
- 对照：解码器已有的 `semantic_head`（用 `feed_gt_structure` 监督）。
- 指标：mIoU（全部 / 未观测区域 / 干净掩码）、稀有类召回、语义 ECE；`others` 不计入 mIoU。

### task4 语义规划器（原 task26）
- 在 dev 上加入 G_sem（语义富度）和语义分歧：先用 BALD 形式，Score = α·G_sem + β·VHD + γ·覆盖。
- 两种传感器设定：只有深度，以及深度加带结构化噪声的语义（ε=0.1）。
- 新增基线：
  - B8 当前地图的语义熵（ActiveSGM 式）；
  - B9 只用语义富度；
  - B10 几何 VHD + 语义富度；
  - B11 几何分歧 + 语义分歧。
- 主指标：实例发现 AUC（一个实例的表面被观测到 ≥ 50% 才算发现）、按类别平衡的语义覆盖 AUC。
- **止损 4**：如果语义不能提升实例发现 AUC，论文就以几何为主，语义放到讨论部分，不做 exp8。

---

## exp7_final_eval（原 exp9）：消融、test 评测和 Replica（W6，约 400 GPU 时）

### task1 消融（原 task27）
在 dev50 上做：
- K ∈ {1, 2, 4, 8, 16}；S ∈ {5, 10, 20}；
- VHD 信号组合；
- 条件分支：官方 / 加多观测比例 / 加 M_obs；
- 一致性处理开或关；
- 语义：不用 / 事后预测；
- 规划器：贪心 vs 2 步 beam；
- 效率：K 个样本批量采样、热启动。

### task2 test 最终评测（原 task28，只跑一次）
- 数据：test200。
- 方法：exp2 的全部基线（S0-F、Random、Frontier、S1、S2、Oracle）、B3、B4，以及最终方法。
- 需要语义时，两种传感器设定都跑。
- 报告主表、配对检验和失败案例。

### task3 Replica 零样本（原 task29）
- 18 个场景，每个场景约 3 个 patch。
- **在任何实验之前**固定类别映射，并对齐到 z 轴向上。
- 内容：补全质量，以及 NBV 主方法的对比。

---

## exp8_joint_semvoxflow（原 exp10）（可选，大概率在投稿之后）

### task1 联合几何-语义流（原 task30）
- 只有 exp6 证明语义对规划有明显收益、而且事后预测器是明显瓶颈时才做。
- 世界状态为 `[z_geo | h_sem]`，条件为 `[z_geo_obs | h_sem_obs | M_obs]`。
- 在 GeoControlNet 上增加语义速度头。
- 损失为 L_geo + λ·L_sem。
- 与 exp6-task3 的事后预测器做对比。

---

## 进度（每次状态变化立即更新）

| 实验 | tasks | 状态 | 当前 task | 最近更新 | 结果文件 |
|---|---|---|---|---|---|
| exp1_infra | 1–7 | **完成 (done)**：patch 已对齐并冻结，工作点 N=64、T=10 已冻结 | – | 2026-09-27 | `exp1_infra/exp1.md`，`exp1_infra/results.md` |
| exp2_baseline | 1–7 | **完成 (done，2026-10-01，PASS)**：task1–7 全部通过；默认配置 S0-F + S=10 + 确定性 world seeding 冻结；规划器主表已定 | – | 2026-10-01 | `exp2_baseline/exp2.md`，`exp2_baseline/results.md` |
| exp_vis（EXP_VIS） | task1 vis_xworld（A、B、B2）、task2 vis_step、task3 vis_stepcov | task1-A、task1-B、task1-B2（含 eiffel k=2、3）、task2、task3 完成；**等你决定是否做单视角的低观测档**（S2 已恢复并完成） | task1-B2 | 2026-09-30 | `experiments/exp_vis/results.md` |
| exp3_geometry_vhd | T1–T5 | **完成 (done)**：T1–T4 PASS；T5 中 VHD-v1.1 显著差于 S2（−0.183），与 Random 相当；2026-10-08 你的决定：VHD 降为分析项，主线转向 S2；`exp3.md` 待写 | – | 2026-10-08 | `exp3_geometry_vhd/vhd_v1_1/results.md` |
| exp3b_macaronspp_transfer | T1–T7 | preregistration v1.1 全部冻结；**T1 PASS**（P1 分层对等、干净 worktree `9521d69`、共享观测采样）；**T2 ✅ PASS after preregistered repair**（as-run FAIL 保留；T2-repair-v1.2 R3 PASS）；**R4 limitation gate = FAIL（永久保留；max \|Δcov_10\| 0.0032 > 0.0011）**；R4-b 光线翻转 0.13%；**R5 固定轨迹配对敏感性：PASS**；**历史射线敏感性问题：CLOSED**（2026-10-09）；**T3 ✅ PASS**（2026-10-09；解释已固定：最低 transfer viability + 永久保留的 domain-gap limitations）；**T4a ❌ FAIL**（early saturation：Oracle Cov@3 0.925 > 0.90；完整性全部通过，确定性 6/6；已归档）；**T4b confirmatory ⛔ BLOCKED**；**T4x-E1 完成**（post-hoc exploratory，执行 VALID；5 个 p0：AUC@10 S1 0.838、S2 0.834、Frontier 0.827、Random 0.660；S2 − S1 −0.003；地面先验引发异常选点）；**T4x 完成**（40 个 episode，post-hoc exploratory，执行 VALID）：AUC@10 S1 0.850、S2 0.852、Frontier 0.850、Random 0.633、Oracle 0.908；S2 − S1 +0.002 [−0.006, +0.009]（没有多世界增益）；S2 − Random +0.219；S2 − Frontier +0.002（持平）；T4x 已归档；Table A（Full-40）已整理；**T4c VHD-20 完成**（pre-specified diagnostic, executed post-hoc as exploratory；执行 VALID；20 个 episode 上 AUC@10：VHD-v1.1 0.853、VHD-v1 0.817、S2 0.864、Frontier 0.856、Random 0.680、Oracle 0.913；VHD-v1.1 − Random +0.173 [+0.102, +0.246]，VHD-v1.1 − S2 −0.011 [−0.037, +0.011]；Spearman v1.1 +0.44；冻结分类规则**不满足**，即 3D-FRONT 的失败模式未重现，但 R 协议早期饱和，只作描述；Table B 已归档，T4C 归档 b1bd7c55…）；**EXP3b-R 阶段暂时关闭**（你 2026-10-10 确认；Table A / Table B 冻结；关闭清单 `R_STAGE_CLOSED_SHA256SUMS` 079be972…）；**T5 feasibility branch：v1.3 已确认；V0–V3 实现验证全部通过**（2026-10-10，冻结 4479789b…，归档 b13e48fb…；没有计算任何 gate 指标）；原 T5 按冻结规则 PASS 不可达（eiffel/2/t0、colosseum/2/t0 世界不可用，原因分别是样本全部在场景框外和传感器量程），未正式运行；**T5-Preflight PASS after separately authorized execution-check repair**（原始 B4 FAIL 保留，你已正式接受）；T5-Diagnostic **已完成（探索性）**：完整性全部 PASS；A1 73 / A2 2；S2 − Random 遗憾 −0.47 [−0.78, −0.20]，S2 ≈ S1，S2 − Frontier 的 CI 包含 0，sestino 无收益；Q3 F@1 0.30；归档 cdcf0eb4；Baseline v2：**T6 已完成（探索性）**——完整性全部 PASS；AUC@10：MACARONS 0.235 > MAGICIAN 0.195 > S2 0.170 > S1 0.158 ≈ Frontier 0.157 > VHD-v1.1 0.142 > VHD-v1 0.134 > Random 0.120（Oracle 0.298）；你已接受 T6 = PASS (execution valid)。T7a（方案 B：6 个核心方法加 Oracle 参考；barts/pantheon/colosseum × 5 起点，100 步，R=10）已批准，pretest 通过，冻结 c685164c；第 1 轮于 10-11 22:00 STOP（colosseum 刷新显存超限，as-run 保留）；你批准方案 1，**第 2 轮** pretest H1–H5 通过、冻结 6af1a691，2026-10-12 17:29 启动；10-13 作业中断后按你的决定用 sbatch 续跑（保留 10 个、重跑 4 个）；**T7a 已完成（探索性）**——完整性 12 项全部 PASS；AUC@100：MAGICIAN 0.697 > Frontier 0.631 ≈ MACARONS 0.628 > S1 0.530 ≈ S2 0.520 > Random 0.366（Oracle 0.768）；归档 c637a6d3；**T7a 定案 = PASS（Execution Valid，Exploratory；你 2026-10-14 的决定）**，结论只针对这 3 个场景；只读补充：同场景前 10 步与 T6 逐位相同，排序变化主要来自第 11–100 步；checkpoint 关联已补记；T7b 不运行（不是 EXP4 的前置条件）；EXP3b-M 关闭 | T7a（关闭） | 2026-10-14 | `exp3b_macaronspp_transfer/results.md`，`protocol_card.md` |
| exp4_multiworld_belief（**新 EXP4**，2026-10-14 你建立：EXP4 — Learning Reliable Generative Multi-World Beliefs for Active 3D Mapping；原 exp4_consistency 的空目录保留） | Task 1 生成器 / Task 2 C0·C1·C2 / Task 3 规划器 | **EXP4-P0 设计审查与 HPC 可行性审查：已完成（只读，不训练）**。6 份文件已提交，结论：有条件可开展；需要你决定数据划分（O3 + 修正案）、P1 批准粒度、是否释放交互作业的 GPU；第一个 GPU 项建议 S0 smoke ；**你新增 Task 0（数据集构建与验证）**：T0-0 设计 + CPU 只读审计完成（`exp4_multiworld_belief/task0/`），GPU pilot 待批准 ；T0-0 = PASS（Design and CPU Audit Completed，不是 GPU 正确性 Gate）；T0-1A 完成（协议 v1.0 + 清单 v1 冻结，修正案 A-EXP4-1 追加生效，Π* 精确规则，脚本 + CPU 检查）；T0-1B smoke 待批准 ；T0-1A = PASS（你确认）；**T0-1B smoke：S1–S4 通过（E1–E3 逐位等价，liberty 目标与 3 条轨迹完成），S5 验证脚本的 3D-FRONT 参照部分出 bug → STOP**（0.82 GPU 时） | Task 0：T0-1B STOP 于 S5 | 2026-10-14 | `exp4_multiworld_belief/exp4_*.md` |
| exp5_active_geometry | 1–3 | 未开始；新路线中即 ActiveGeometry-S2，评测改为 S2 闭环 + 两个数据域，开始前复述时重写；**2026-10-22 空目录已按你的要求删除，EXP5 改为 exp5_future_view**（下一行） | – | 2026-10-22 | – |
| exp5_future_view（**EXP5 / V2**，2026-10-22 建立，任务书 `EXP4_NextStage_Task0_Task1_V2_Research_Plan.md`） | Task 0 低覆盖诊断；Task 1 V2；Task 2 V2-Mix + 最终对照；消融 A1 / A2；Task 3 全深度 P1（E1、E1.2、E2 Pilot） | 全部 **DONE**（2026-10-23 → 10-29）：V2-Mix MP3D AUC@150 0.322 > MAG-B 0.292，Macarons++ Test-3 0.723 > 0.656；全深度 P1 直接替换 No-Go，B2 无退化；E2 全深度微调室内 +0.043 AUC、室外无效（G3 Conditional Go，探索性）。**2026-10-29 归档到 git（`[V2][类型]` 提交）** | 已归档 | 2026-10-29 | `exp5_future_view/README.md`（归档索引 + 进度表） |
| exp6_semantics | 1–4 | 未开始；不在 2026-10-08 的新路线中，等你决定是否保留 | – | 2026-10-08 | – |
| exp7_final_eval | 1–3 | 未开始；新路线中扩展为两个数据域的重新评测（3D-FRONT test200、MACARONS++ R/M 协议）和 MAGICIAN 对比 | – | 2026-10-08 | – |
| exp8_joint_semvoxflow | 1 | 可选 | – | – | – |

已完成的准备工作（不属于任何 task）：
- TensorBoard 日志（wandb 可选）；
- `work_dirs/` 目录结构；
- 官方权重已移到 `work_dirs/weights/official/`。
