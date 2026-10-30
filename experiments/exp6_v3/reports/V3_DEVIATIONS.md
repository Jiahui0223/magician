# EXP6 V3 — 偏差与工程记录（持续追加，不改已写条目）

| # | 日期 | 内容 | 影响 |
|---|---|---|---|
| 1 | 2026-10-29 | **git 范围变更（用户授权）**：2026-10-12 的规则"git 不含 `experiments/`"被本次计划（`EXP6_V3_PLAN.md` §0.1）覆盖。`.gitignore` 改为 `experiments/*` 忽略、放行 `experiments/README.md`、`exp5_future_view/`、`exp6_v3/`；8 个 EXP5 结果文件（> 1 MB）仍留本地。V2 归档以 `[V2][chore/archive/test/docs]` 4 个提交推送（`efd7e14`） | 仓库约定更新，记入 `github.md` 同义规则的记忆 |
| 2 | 2026-10-29 | **计划引用的文件在仓库中不存在**：根目录没有 `idea.md`（`.git/info/exclude` 中列为本地笔记，但文件本身不存在）；没有 `EXP6_V3_EXPERIMENT_DESIGN.md`、`ablationstudy.md`。处理：V3 的研究概述写在 `experiments/exp6_v3/README.md` §1；不新建根目录 `idea.md`；"旧草案路线"按 `EXP6_V3_FROM_SCRATCH_PLAN.md` 的描述视为已废弃 | 无 |
| 3 | 2026-10-29 | **历史 `plan.md` / `exp6_replica` 处理**：`plan.md`（本地笔记）中 exp6 原为 `exp6_semantics`，不存在 `exp6_replica`；本实验独立命名 `exp6_v3`，不改写任何旧目录或旧文字 | 无 |
| 4 | 2026-10-29 | **脚本中的绝对路径**：EXP5 脚本含本集群环境路径（`/bsuscratch/jiahuizhang/envs/...`）。归档时按计划"不借归档改动代码"未修改，已在 `exp5_future_view/README.md` 标注。V3 新代码通过 `seen2scene.PROJECT_ROOT` 和配置文件定位路径 | 无 |
| 5 | 2026-10-30 | **缓存构建器显存不足**：第一版 `partial_tsdf.aggregate` 和分片核对用的 `fu.structure()`（冻结代码，内部用 float64 体积）在 MP3D（2.9 亿体素）/ dunnottar（4.5 亿体素）上峰值 20+ GB，每卡 3 个进程时 OOM。改为逐量池化、半精度指示量、`structure_lean`（由 cnt / empty 直接得到三态，与 `fu.structure()` 等价，单元测试核对），重启构建；已完成的 2 + 28 条轨迹保留，失效的 claim 清除 | 无结果影响 |
| 6 | 2026-10-30 | **配置格式**：计划建议 `configs/common.yaml`；为与 V2 工具链和哈希记录一致，V3 配置使用 JSON（`configs/model_v3.json`、`train_v3.json`），两组共用同一文件、只改 `xattn.enabled` | 无 |
| 7 | 2026-10-30 | **G0 冒烟与缓存构建共用 gpu117**：tiny overfit / DDP profile 与 13 个缓存构建进程同时运行，GPU 0 上 1 个 MP3D 构建进程 OOM 退出（其轨迹由构建结束后的补漏一遍重建，输出原子写入无残缺）；第 2 次单元测试运行因 CPU 争用超时（EXIT 124），改在干净 profile 之后重跑；P3.6 的 profile 计时为争用下数值，训练预算以 P3.8 的干净 profile 为准 | 只影响时间，不影响结果 |
| 8 | 2026-10-30 | **resume 规则**：第一次 DDP resume 测试发现恢复时 `torch.set_rng_state` 收到被 `map_location` 搬到 GPU 的状态而报错（B2 段退出码 1）。修正为 V2 训练器的规则：恢复模型 / EMA / optimizer，随机数发生器按 (seed, rank, step0) 重新确定性播种（rank 0 保存的状态只作记录；逐 rank 精确恢复需要所有 rank 保存状态）。因此中断后的样本流可复现，但与不中断的运行不同；resume 测试比较的是"能否正确继续、损失量级一致"，不是逐位相同。正式训练前修复，不影响任何结果 | 训练前修复 |
| 9 | 2026-10-30 | **G3 的执行顺序**：`eval_fixed.py`（G1 / G2）在同一状态、同一噪声下一次评 T0 与 T1 两个 checkpoint，因此 G1 / G2 只能在两组都结束后运行（T1 预计 10-31 03:14）；为不浪费 gpu117 的分配时间（11-01 14:50 到期），每组训练一结束就按冻结选择规则发布 `final.pt`（`shared/select_final.py`，只用 VALSEL / S-c）并立即开始该组的 G3 闭环（`post_arm.sh`）。协议 §4 规定 G3 无论 G2 结果都要运行（只影响 formal / exploratory 标签），闭环输出与 G2 无关，因此这是调度顺序偏差，不改变任何结果；标签在 G2 结束后补。队列前先跑 2 步闭环冒烟（室内 Z6MF S6、室外 barts s0，输出 `eval/smoke/`，不计入结果） | 调度 |
