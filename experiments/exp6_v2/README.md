# exp6_v2 —— V2 家族 B2 / E2 的统一整理（2026-11-02）

本目录只做文件整理、目录规范化和运行验证：
- 没有改模型架构，没有改 Flow、DistP1 或训练参数；
- 没有训练，也没有新实验；
- 没有移动、复制或删除任何权重。

B2 和 E2 共用同一个 V2 架构：3D U-Net + 32 点射线编码器 + Conv1D + 2D Rectified Flow + DistP1。两者的训练历史、数据配置和评估设置分开保存。
- 详细说明：`B2/README.md`、`E2/README.md`；
- 机器可读配置：`B2/configs/b2.json`、`E2/configs/e2.json`；
- 其他版本（V3–V5.2-M）的架构和 checkpoint 总览：`experiments/MODEL_VERSIONS.md`、`experiments/CHECKPOINTS.md`。

## 1. 目录树
```text
experiments/exp6_v2/
├── README.md                      本文件
├── shared/                        B2 / E2 共用的小工具（不含模型代码）
│   ├── env.sh                     运行环境（seen2scene env、PYTHONPATH）
│   ├── v2_arm.py                  加载臂配置，核对 checkpoint（链接 / SHA / step / model_cfg / 严格加载 / 参数），生成 Core
│   ├── smoke_v2.py                最小推理 smoke + 整理前后逐位对比
│   ├── g12_fixed.sh               历史 G1 / G2 命令（e2_fixed.py + e2_analyze.py），只补写缺失文件
│   └── g3_queue.sh                G3 闭环 worker（参数化的历史 run_queue2.sh / run_queue_g3.sh），默认输出到新目录
├── B2/
│   ├── README.md
│   ├── configs/  b2.json · g3_jobs_in.txt（MP3D 10 ep）· g3_jobs_out.txt（Macarons++ Dev-3 + Test-3，6 场景 × 5 起点 = 30 ep）
│   ├── scripts/  verify.sh · smoke.sh · g12_fixed.sh · g3.sh · train_lineage.sh
│   └── results/  verify_config_checkpoint.json · smoke_dev3.json · smoke_valsel.json · smoke_g3_2moves.json
│                 historical/ → TASK2_FINAL_REPORT.md · T3_E1_B2_CLOSED_LOOP.md · T3_E1_DECISION.md · tables.md ·
│                               t34_closed_loop.json · E2_pilot_g12_s5000_with_B2.md   （指向原报告的相对软链接）
└── E2/
    ├── README.md
    ├── configs/  e2.json · g3_jobs_{in,out}.txt（E2-D100）· control_e2s10_g3_jobs_{in,out}.txt（E2-S10 对照）
    ├── scripts/  verify.sh · smoke.sh · g12_fixed.sh · g3.sh · train.sh
    └── results/  verify_config_checkpoint.json · smoke_dev3.json · smoke_valsel.json · smoke_g3_2moves.json
                  historical/ → E2_AUDIT.md · E2_G12_REPORT.md · E2_G3_REPORT.md · g12_s5000.{md,json} · g3.{md,json} ·
                                protocol.md · DEVIATIONS.md   （软链接）

work_dirs/checkpoints/exp6_v2/     （work_dirs 不进 git；全部是指向原文件的相对软链接）
├── README.md · MANIFEST.json      每个文件的原路径、SHA256、大小、step、model_cfg 哈希、参数、严格加载结果
├── B2/
│   ├── v2mix_ema17500.pt          ★ B2 模型（V2-Mix EMA@17.5k）
│   ├── v2mix_ema20000.pt          V2-Mix 末 EMA
│   ├── v2mix_raw20000.pt          V2-Mix 原始训练状态（含优化器）
│   ├── intermediate/v2mix_ema{000000,002500,…,015000}.pt
│   └── lineage/v2orig_ema60000.pt · v2orig_raw60000.pt     V2 Original（V2-Mix 的初始化）
└── E2/
    ├── e2d100_ema5000.pt          ★ E2 模型（E2-D100 EMA@5k）
    ├── e2d100_raw5000.pt          原始训练状态
    ├── intermediate/e2d100_ema{000000,001000,…,004000}.pt
    └── control_e2s10/e2s10_ema5000.pt · e2s10_raw5000.pt · intermediate/e2s10_ema{001000…004000}.pt
```

## 2. B2 与 E2 的区别、训练来源和 checkpoint 关系
```
V2 Original（EXP5 Task 1；随机初始化，Outdoor 稀疏 P1，60k）── EMA@60k  a379c0fb…
        │  初始化（张量级核对：V2-Mix step 0 权重 = V2 Original EMA@60k）
        ▼
V2-Mix（EXP5 Task 2；MP3D + Train-6 稀疏，20k，冻结规则选 17.5k）── EMA@17.5k  343f41ab…  ══ B2 的权重（同时也是 B0）
        │  初始化（张量级核对：E2 step 0 权重 = V2-Mix EMA@17.5k）
        ├──▶ E2-D100（EXP5 Task 3 E2 pilot；D100 分片，5k）── EMA@5k  61439612…   ══ E2
        └──▶ E2-S10（同配方，S10 稀疏分片，对照）── EMA@5k  578855bb…
```
| | B2 | E2（E2-D100） |
|---|---|---|
| 权重 | V2-Mix EMA@17.5k，不再训练 | V2-Mix EMA@17.5k 再微调 5k 步 |
| 训练数据 | （V2-Mix：MP3D + Train-6，稀疏 P1） | MP3D + Train-6，**D100** 重建分片 |
| 规划 | D100 输入 / S10 DistP1，K 4，n_win 5，R10 | 相同 |
| 冻结协议 | Task 2 `protocol.md`、Task 3 `protocol.md` | `e2_pilot/protocol.md` |
| G3 历史已报告结果（AUC / Final；MP3D 10 ep × 150；Macarons++ Dev-3 + Test-3（6 scenes，30 episodes × 100 steps）） | MP3D 0.334 / 0.530；Macarons++ 0.730 / 0.906 | MP3D 0.364 / 0.553；Macarons++ 0.711 / 0.880 |

## 3. 重要 checkpoint：原路径 → 新路径
| 新路径（`work_dirs/checkpoints/exp6_v2/…`） | 原路径（保持不变，仍有效） | step | sha256（前 16） |
|---|---|---|---|
| `B2/v2mix_ema17500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt`（= `work_dirs/exp5_future_view/task2/training/mix_v1/ema_017500.pt` 硬链接） | 17500 | 343f41ab8d566c52 |
| `B2/v2mix_ema20000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-2_V2Mix_last/ema_020000.pt` | 20000 | 5c68720ea6c64f1a |
| `B2/v2mix_raw20000.pt` | `work_dirs/exp5_future_view/task2/training/mix_v1/ckpt_last.pt` | 20000 | 9805f081da3a… |
| `B2/intermediate/v2mix_ema0xxxxx.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_0xxxxx.pt` | 0–15000 | MANIFEST |
| `B2/lineage/v2orig_ema60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt`（= `task1/training/main/ema_060000.pt` 硬链接） | 60000 | a379c0fb9d7d8026 |
| `B2/lineage/v2orig_raw60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ckpt_last.pt` | 60000 | 7e827ecc2ce5… |
| `E2/e2d100_ema5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_005000.pt` | 5000 | 61439612cb6132af |
| `E2/e2d100_raw5000.pt` | `…/e2/training/d100/ckpt_last.pt` | 5000 | f647eb5f7a7c… |
| `E2/intermediate/e2d100_ema0xxxxx.pt` | `…/e2/training/d100/ema_0xxxxx.pt` | 0–4000 | MANIFEST |
| `E2/control_e2s10/e2s10_ema5000.pt` | `…/e2/training/s10/ema_005000.pt` | 5000 | 578855bb14995626 |
| `E2/control_e2s10/e2s10_raw5000.pt` | `…/e2/training/s10/ckpt_last.pt` | 5000 | 2cb3f1357f02… |
| `E2/control_e2s10/intermediate/e2s10_ema00x000.pt` | `…/e2/training/s10/ema_00x000.pt` | 1000–4000 | MANIFEST |

- 合计 25 个链接。没有失效链接。
- E2-S10 的 step 0 与 E2-D100 的 step 0 字节相同，只登记一次。
- SHA 与历史 manifest / 报告逐一核对一致：343f41ab、a379c0fb、61439612。

**为什么用软链接而不是移动**：旧路径仍被大量冻结配置和脚本引用，例如：
- `task2/configs/eval_{indoor,outdoor}_v1.json` 的组 V2M；
- `e2_d100_v1.json` 的 init_checkpoint；
- `e2_fixed.py`、`ab_common.py`；
- EXP6 V3–V5.2-M 各评估器中的 B2 / E2 行。

移动这些文件会使旧实验无法复现，所以按要求原路径保留，新目录用软链接。

## 4. 保留在原位置的共享代码和文件（没有复制）
- **模型 / 推理**：`seen2scene/futureview/{model.py, predictor.py, rays.py, geometry.py, scoring.py}`。
- **规划核心**：`experiments/exp5_future_view/task1/scripts/v2_plan_core.py`（另有 `v2_common.py`、`v2_data.py` 等）。
- **训练**：
  - Task 1 `t12_train.py`；
  - Task 2 `t22_train_mix.py`、`t22_select.py`；
  - E2 `e2_pilot/scripts/{e2_paths,e2_build,e2_train}.py`。
- **规划器 / 模拟器**：
  - Task 2 `t23_in_sim.py`、`t24_out_sim.py`、`t23_v2_in_planner.py`、`t24_v2_out_planner.py`；
  - Task 3 `p1_full.py`（b2 模式）、`t32_{in,out}_planner.py`、`t33_{in,out}_sim.py`、`t35_out_cov.py`、`t34_analyze.py`。
- **G1 / G2**：`e2_pilot/scripts/{e2_fixed,e2_analyze,e2_valsel_oracle,e2_g3_analyze}.py`、`density_ablation/scripts/e12_fixed.py`。
- **配置**：
  - `task1/configs/{model_v1,data_v1,scoring_v1,train_main}.json`；
  - `task2/configs/{train_mix_v1,eval_indoor_v1,eval_outdoor_v1}.json`；
  - `e2_pilot/configs/e2_{d100,s10}_v1.json`。
- **权重注册表**：`work_dirs/weights/`（W20 / W21，硬链接，未改动）。

## 5. Smoke Test 结果（2026-11-02，gpu115 L40S；全部通过）
| 检查 | B2 | E2 |
|---|---|---|
| 配置加载、引用文件存在（`scripts/verify.sh`） | ✅ | ✅（含 E2-S10） |
| 链接 = 原文件；SHA256、step 与记录一致 | ✅ 343f41ab… / 17500 | ✅ 61439612… / 5000；S10 578855bb… / 5000 |
| checkpoint 内嵌 model_cfg = `model_v1.json` | ✅ | ✅ |
| 严格加载 V2Model，参数 24,822,114 | ✅ | ✅ |
| **固定状态推理**（`scripts/smoke.sh`）：Dev-3 barts_s0（10 态 / 73 候选）+ MP3D 8194nk5LbLH L0（10 态 / 79 候选）；分别从旧路径和新路径加载 | | |
| 未来深度（每个候选 K 4 × 射线的首中张量 SHA256）旧 = 新 | ✅ 逐位一致 | ✅ 逐位一致 |
| DistP1（每候选 × K）旧 = 新 | ✅ | ✅ |
| 与历史结果比较（E2 pilot 当时写下的 `e2/fixed/<set>_s5000` 行）：DistP1 分数、相机排序、预测误差统计 | ✅ 全部完全相等（B2 与 B0） | ✅ 全部完全相等（D100 与 S10） |
| 张量级谱系 | V2-Mix step 0 = V2 Original EMA@60k ✅ | E2 step 0 = V2-Mix EMA@17.5k ✅；EMA@5k ≠ 初始化 ✅ |
| **闭环入口**（`shared/g3_queue.sh`，MOVES=2）：Macarons++ alhambra s0（Test-3 场景）与历史 100 步 episode 的前两步比较 | 候选集、DistP1 分数、选择、动作、覆盖率完全一致 ✅ | 完全一致 ✅ |
| MP3D 8194nk5LbLH L0 S10 两步闭环 | exit 0（该 episode 不在历史 G3 列表中，只验证能运行） | exit 0 |

推理耗时（单个 episode，旧 / 新）：
- Dev-3：47 / 41 s；
- MP3D：约 160 s。

结果文件：`B2/results/`、`E2/results/` 下的 `verify_config_checkpoint.json`、`smoke_{dev3,valsel}.json`、`smoke_g3_2moves.json`。没有运行完整 G3，也没有重新训练。

## 6. 缺失文件、失效路径和对旧实验的潜在影响
- **没有移动或删除任何文件**，所有旧路径继续有效，因此对旧实验没有影响。新入口默认输出到 `work_dirs/exp6_v2/`，不会写入历史结果目录。唯一例外是 `g12_fixed.sh`：它沿用 E2 pilot 的写入目录，但只补写缺失文件。
- **原来只在 work_dirs 中的启动脚本**（不进 git）：
  - `work_dirs/exp5_future_view/task3_full_depth_p1/{run_queue2.sh, chain_b2.sh, jobs_b2_{in,out}.txt}`；
  - `…/e2/{train_arm.sh, build_out.sh, build_mp3d_node.sh, eval_node.sh, chain_g12.sh, run_queue_g3.sh, chain_g3.sh, jobs_g3_*.txt}`。

  现在有了版本化的等价入口：`shared/g3_queue.sh`、`shared/g12_fixed.sh`、`*/scripts/*.sh`、`*/configs/g3_jobs_*.txt`。job 列表是原样复制的。
- **命名冲突**：`task1/configs/closed_loop_v1.json` 中的 "B2" 基线行和 "E2" 臂是旧命名，与这里的 B2 / E2 无关。
- **B2 的 G1 / G2 结果没有单独文件**：它们存在 E2 pilot 的固定状态行里（模型键 `B2`）。EXP6 V3–V5.2-M 的分析脚本都从这些行读取 B2 / E2 的基线。
- **`train_lineage.sh` / `train.sh` 只打印命令**：`RUN=1` 时才执行。其中 V2-Mix 阶段按 `train_mix_v1.json` 从历史 V2 Original EMA@60k 初始化，而不是从重训的第一阶段初始化——这是配置文件本身的设定。
- **未纳入本目录的项**：MAGICIAN（官方权重 W00-2）；B0 / B1 的其他 Task 3 组（与 B2 共用权重，只差 P1 模式，见 `T3_E1_*`）；V2 Original 的中间 EMA（`work_dirs/weights/W20_exp5_v2/W20-4_*`）。

## 7. Git 修改摘要（本次整理）
- **新增 `experiments/exp6_v2/`**：README ×3、`configs` ×2（JSON）、job 列表 ×6、shared ×5、scripts ×10、results 中的验证 / smoke JSON ×8，以及 15 个指向历史报告的相对软链接。
- **新增 `experiments/checkpoint_registry.py`**：生成 `work_dirs/checkpoints/exp6_*` 的软链接、MANIFEST 和 README。
- **新增 `experiments/CHECKPOINTS.md`**（自动生成的全版本 checkpoint 索引）和 `experiments/MODEL_VERSIONS.md`（架构与继承总览、审查发现）。
- **修改 `.gitignore`**：放行上述新文件和目录。
- **修改 `experiments/README.md`**：新增变更记录和索引行。
- **未修改**：任何模型代码、训练 / 评估脚本、冻结配置、权重文件或历史结果。
- `work_dirs/checkpoints/` 不进 git（`work_dirs/` 被忽略），内容由 `checkpoint_registry.py` 可重建。

## 8. 第二轮修正与验收（2026-11-02 19:40）

### 8.1 本轮修改的文件和内容
| 文件 | 修改 |
|---|---|
| `experiments/MODEL_VERSIONS.md` | ① 完整 G3 的 Outdoor 范围改为 **Macarons++ Dev-3 + Test-3（6 scenes，30 episodes × 100 steps）**：Dev-3 = barts / pantheon / colosseum，Test-3 = alhambra / bridge / neuschwanstein；表头与说明同步修改，数值不变。<br>② 加入命名备注 `V5.3 = V5.2-M Scratch70k（同一模型，讨论名称）`：标题、总览表、审查发现第 7 条。<br>③ “已发表结果”统一改为“历史已报告结果”。<br>④ V5.2-M 的 smoke 数字注明只含 Test-3 的 4 个 episode，不可与 30 episode 的结果直接比较。 |
| `experiments/exp6_v2/README.md`、`B2/README.md`、`E2/README.md` | G3 表头注明评估范围（MP3D VALSEL 10 ep × 150；Macarons++ Dev-3 + Test-3，30 ep × 100）；“历史结果”改为“历史已报告结果”；alhambra 注明为 Test-3 场景 |
| `B2/configs/b2.json`、`E2/configs/e2.json` | `evaluation.G3_closed_loop.episodes` 的说明同步修正（只改描述字段，不改任何参数） |
| `B2/scripts/g3.sh` | 注释中的评估范围修正 |
| `B2/scripts/smoke.sh`、`E2/scripts/smoke.sh` | 增加可选第 3 个参数 LABEL，结果写到 `results/smoke_<set>_<LABEL>.json`，不覆盖第一轮的 smoke 记录 |
| `experiments/checkpoint_registry.py` | 新增只读核验模式 `--verify OUT.json`；V5.2-M 的标题加上 “= V5.3” |
| `experiments/CHECKPOINTS.md`、`work_dirs/checkpoints/{README.md, exp6_v5_2_scratch70k/README.md, MANIFEST.json}` | 只改 V5.2-M 的标题（加 “= V5.3”），没有重新生成，SHA / 路径行未改动 |
| `experiments/CHECKPOINTS_VERIFY.json`（新增） | 全量核验报告 |
| `experiments/exp6_v5_2_scratch70k/FROZEN_PROTOCOL.md` | §5 中 G3 Full 的 Outdoor 范围原文误写为 “Test-3 30 episode”，就地勘误为 Dev-3 + Test-3（30 episodes），并在 §9 记录。G3 Full 未运行，没有结果受影响 |
| `experiments/exp6_v5_2_scratch70k/scripts/final_report_v52m.py` → `FINAL_REPORT.md` | 用户基准表表头注明 30 episode 的范围，并说明它与只含 Test-3 的 smoke 不可直接比较。重新生成后只改了这两行和时间戳，`STATUS.md` / `TRAINING_LOG_SUMMARY.md` 只改了时间戳 |

### 8.2 Checkpoint 检查结果（`checkpoint_registry.py --verify`，19:23，只读）
- 范围：7 个实验目录，共 108 个登记的软链接，以及 9 个 alias（字节相同、未重复链接的文件）。
- **全部通过，问题数 0**。

逐项检查内容：
- 软链接存在，链接目标未变，目标文件存在；
- 链接与原文件是同一文件（samefile）；
- 重新计算的 SHA256 与 MANIFEST 一致，大小一致；
- checkpoint 内的 step 一致；
- EMA 权重按对应架构严格加载（strict=True），参数量与 MANIFEST 一致；
- 原始训练状态中 model 与 EMA 的键和形状一致；
- 没有孤立的链接（不在 MANIFEST 中的链接）；
- `experiments/CHECKPOINTS.md` 中每一行的新路径、原路径、step、参数、SHA 前缀都与真实文件一致。

重点三个 checkpoint 还与仓库中独立的历史记录做了交叉核对：

| checkpoint | 注册表 SHA256 | 独立记录 | step / 参数 |
|---|---|---|---|
| `exp6_v2/B2/v2mix_ema17500.pt` | 343f41ab8d566c52… | 与 `work_dirs/weights/W21…/MANIFEST.json`、Task 2 `eval_{indoor,outdoor}_v1.json` 的 V2M 组、E2 `init_checkpoint_sha256`、`exp6_v4/CHECKPOINT_LINEAGE.md` 都一致 ✅ | 17500 / 24,822,114 ✅ |
| `exp6_v2/E2/e2d100_ema5000.pt` | 61439612cb6132af… | 与 `exp6_v4/CHECKPOINT_LINEAGE.md` 一致 ✅ | 5000 / 24,822,114 ✅ |
| `exp6_v5_2_scratch70k/V5_2M/v52m_best_ema50000.pt`（= V5.3） | a540950738cc7191… | 与 `exp6_v5_2_scratch70k/RUN_STATUS.json` 的 `final.sha256` 一致 ✅；step 与 `best_train_holdout.json` 一致 ✅ | 50000 / 26,999,143（= PARAMETER_AUDIT）✅ |

### 8.3 B2 / E2 Smoke Test（从新目录入口：`scripts/verify.sh` + `scripts/smoke.sh dev3 <GPU> acceptance`；19:23，L40S）
| 检查 | B2 | E2 |
|---|---|---|
| `verify.sh`：配置加载、链接 = 原文件、SHA、step、model_cfg = model_v1.json、严格加载、参数 | ok ✅ | ok ✅（含 E2-S10 对照） |
| 推理：Dev-3 barts_s0，10 态 / 73 候选，`score_state` 调用 20 次 | 运行正常 | 运行正常 |
| 未来深度（首中张量 SHA256）：旧路径 = 新路径 | ✅ 逐位一致 | ✅ 逐位一致 |
| DistP1：旧 = 新 | ✅ | ✅ |
| 与历史行比较（DistP1 分数、相机排序、预测误差统计） | ✅ 全部相等 | ✅ 全部相等 |
| 张量级谱系 | ✅ | ✅ |
| 耗时（旧 / 新，单个 episode） | 45 / 39 s | 46 / 41 s |

结果：`B2/results/smoke_dev3_acceptance.json`、`E2/results/smoke_dev3_acceptance.json`。第一轮的 `smoke_{dev3,valsel}.json`、`smoke_g3_2moves.json` 保留未动。

### 8.4 尚未解决的问题
- 本轮没有未解决的整理问题。
- 第一轮列出的事项仍然成立，但都不影响复现：旧命名冲突（`closed_loop_v1.json` 中的 B2 / E2）、`exp6_v3.1` 与 `exp6_v3_1` 目录名不一致、`final.pt` 与最后一个 EMA 字节相同（按 alias 处理）。
- 这些都没有改动，因为旧脚本依赖原路径。

### 8.5 最终 Git 修改摘要（本轮）
- **修改**：`.gitignore`（放行 `experiments/CHECKPOINTS_VERIFY.json`）、`experiments/MODEL_VERSIONS.md`、`experiments/CHECKPOINTS.md`（只改标题）、`experiments/checkpoint_registry.py`。
- **修改 exp6_v2**：README ×3、`configs` ×2、`scripts/g3.sh`、`scripts/smoke.sh` ×2。
- **修改 exp6_v5_2_scratch70k**：`FROZEN_PROTOCOL.md`（勘误）、`scripts/final_report_v52m.py`、`FINAL_REPORT.md`、`STATUS.md`、`TRAINING_LOG_SUMMARY.md`（重新生成）。
- **新增**：`experiments/CHECKPOINTS_VERIFY.json`、`B2/results/smoke_dev3_acceptance.json`、`E2/results/smoke_dev3_acceptance.json`。
- **未删除、覆盖或移动任何 checkpoint**；没有训练，没有运行 G3。
