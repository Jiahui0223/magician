# OVERNIGHT_AUDIT — CVPR overnight run 2026-11-10 的预检查与资源审计

写于 2026-11-10 05:10（节点 gpu114）。依据 `cvpr_overnight_experiment_plan.md` §1。本文件只记录**已核对的事实**；逐条 episode 的机器可读清单在 `manifest/episodes.csv`（由 `scripts/build_manifest.py` 生成，可重跑）。

## 1. 仓库与主方法

| 项 | 核对结果 |
|---|---|
| git | 分支 `seen2scene`，HEAD `b8d65febc72b`；工作区有未提交的日志改动（XB logs），本轮新增文件全部在 `experiments/cvpr_results/{scripts,fig,manifest,report}`、`table1/MODEL_REGISTRY.yaml`、`table1/scripts/{t1_queue.sh,import_episodes.py}` |
| 主方法 checkpoint | `work_dirs/exp6_v3/task0_vae/checkpoints/final.pt`，sha256 `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6` = 计划书预期值 ✅（`manifest/main_method_checkpoint.json`） |
| 无交叉注意力 | checkpoint 内 `model_cfg.xattn.enabled = False`，EMA 权重中 `unet.xattn.*` 张量 0 个，参数 23,539,622，step 40000 ✅ |
| 推理设置 | K = 4、Euler 10（`model_cfg.flow`）；D1 = 冻结 D-v1 策略 `d1`（depth 2、beam null、lambda_risk 0.0）；p1_mode b2 = D100 编码输入 + S10 观测点做新颖性；R10 刷新 —— 均取自 70 条 episode 的规划器握手记录（`planner_hello`），与 `V3_T0_D1_MODEL_ARCHITECTURE.md` 一致 |
| D1 / H 策略文件 | `d_policy.py` sha `c8418279…`、`h_policy.py` sha `ccfdf4af…`、`d_config.yaml` sha `e53c230d…`；70 条 T0+D1 记录的握手里三个 sha 全部相同 |
| 官方覆盖重放 | `experiments/exp5_future_view/task2/scripts/t25_out_cov.py`（冻结）：按执行位姿重放，官方覆盖必须与存档曲线逐位相同，否则 STOP；Table 1 流水线 `t1_queue.sh` 对每条 episode 都执行 |

## 2. 现有 episode 记录（`work_dirs/cvpr_results/table1/episodes/`，规范记录）

逐条校验项：曲线长度 101、全部有限且在 [0, 1]、执行位姿 101 个、scene / start 在冻结网格内、split 标签与注册表一致、episode 文件曲线 = 规范记录曲线、官方重放 = 存档曲线（`official_equal_stored`）、规划器握手里的 K / depth / checkpoint sha 与期望一致。

| 方法（Table 1 id） | 记录数 / 期望 | 校验 | 用途 |
|---|---|---|---|
| V3-T0-D1（Ours = V3-T0 + D1，K4 / depth 2） | 70 / 70 | 全部 COMPLETE | Table 1 / 2 / 3、Fig 3 / 5 |
| MAG-B（MAGICIAN beam 10×10，官方代码 e697009） | 70 / 70 | 全部 COMPLETE | Table 1 / 2、Fig 3 / 5 |
| MAG-G（MAGICIAN greedy） | 70 / 70 | 全部 COMPLETE | 参考 |
| V3-D1-H3-NORISK（V3-T1 + D1，交叉注意力对照） | 70 / 70 | 全部 COMPLETE | 参考（不是主方法） |
| V3-H1（V3-T1 + 一步 value_cov） | 70 / 70 | — | 仅用于回归（深度 1 路径的参照） |
| 其他 23 个历史模型目录（B0 / B1 / B2 / E2 / V2O / V3 / V4 / V5 / H / D / Y / F1 …） | 各 25–70 | 未改动 | 不进入本轮图表 |
| RANDOM、MACARONS | 0 / 70（本轮新跑） | — | Table 1 |
| SCONE、FisherRF | — | BLOCKED | Table 1 留空 |
| V3-T0-K4D1、V3-T0-K1D2、V3-T0-K1D1 | 0 / 70（本轮新跑） | — | Table 3 |

复算核对（场景等权，All-14）：T0 + D1 = 0.420 / 0.646 / 0.633 / 0.763 / 0.911（AUC@20 / Cov@20 / AUC@50 / AUC@100 / Final），MAGICIAN Beam = 0.307 / 0.537 / 0.544 / 0.711 / 0.906，与计划书里的截图数值三位小数一致 ✅。

## 3. 场景划分与训练数据来源（`table1/SCENE_REGISTRY.yaml`、`table1/DATASET_AUDIT.md`、`protocol_audit/`）

| 划分 | 场景 id |
|---|---|
| Train-6 | bannerman, dunnottar, fushimi, liberty, pisa, redeemer |
| Dev-5 | barts, colosseum, eiffel, pantheon, sestino_museum |
| Test-3 | alhambra, bridge, neuschwanstein |
| 未纳入 | sofia_church（冻结模拟器的加载器对它报错，协议 v1.0 下所有模型都不可运行；始终写 All-14，不写 All-15） |

- **T0 的训练**：室外 = Train-6 去掉 S-c 留出轨迹（138 条轨迹，`seen2scene/v3/data.py:43-58`）+ MP3D 训练房屋；Dev-5 / Test-3 没有任何训练数据。Train-6 每个场景有 3–4 个官方起点同时是某条训练轨迹的起点（`split_manifest.csv` 列出）。
- **T0 的选模**：MP3D VALSEL + S-c（Train-6 场景的 42 条留出轨迹），规则 = ES 比值和最小；实际选中最后一个 checkpoint（40k）。Dev-5 / Test-3 没有参与。
- **规划器 D1**：value_cov 的参数（eps_cov = 官方覆盖容差 1.0、cell = 1.0、depth 2）在 H-v1 / D-v1 协议里先验冻结；D1 / D2 / D3 之间按 Dev-5 的 AUC@100 预注册选择。但 H1–H3、D1、T1、T0 各行此前都在 All-14（含 Test-3）上评过并被查看，所以 **Test-3 对 T0 权重是留出场景，对系统设计不是盲测集**。
- **MAGICIAN 的权重**：没有自己的权重，用 MACARONS 作者的 `trained_macarons.pth`（sha `34c5247f…`）。按 MACARONS 训练配置，我们 14 个评测场景中 8 个是它的训练场景（Train-6 全部 + colosseum、pantheon），alhambra / neuschwanstein 列为它的 validation，eiffel / bridge 列为 test，barts / sestino_museum 不在它的数据里。因此 Table 2 只能写 **"performance on T0-held-out scenes"**，不能写"两种方法都没见过的场景"。

## 4. 基线可执行性（逐项核对代码 / 权重 / 缓存）

| 方法 | 代码 | 权重 | 同协议缓存 | 判定 |
|---|---|---|---|---|
| MAGICIAN Beam | 官方 core e697009 + `t24_mag_out_planner.py` | trained_macarons.pth ✅ | 70 / 70（已验证） | **GO（复用）** |
| Ours V3-T0 + D1 | 冻结 | T0 ✅ | 70 / 70（已验证） | **GO（复用）** |
| Random Walk | 上游 `macarons/testers/random_planning.py` **不在本地仓库**（`test_random_planning.py` 引用但模块缺失）。EXP3b 冻结定义 "Uniform Random"（`m_t6_common.uniform_choice`，对模拟器合法动作列表带种子均匀选择）可由 `m_t7_sim.py --method random` 在同一 EpisodeSim 上运行 | 不需要 | T7a 只有 barts / pantheon / colosseum × 5 起点 × 3 种子 | **GO（adapted）**：冒烟 barts_s0 与 T7a 归档 seed 0 记录逐位一致，官方重放通过 |
| MACARONS | EXP3b `m_t7_macarons.py`（适配修复 R1–R8，同一 EpisodeSim、合法动作列表、官方覆盖） | trained_macarons.pth ✅ | T7a 同上 3 个场景 | **GO（adapted）**：冒烟 barts_s0 与 T7a 归档 seed 0 记录逐位一致，官方重放通过；单条 6.5–13 min，显存占用 23 GB（nvidia-smi） |
| SCONE | 只有 ShapeNet 物体级 demo（`MACARONS/demos/scone_demo.py`、`configs/scone`）；没有 Macarons++ 场景级规划器 | `weights/scone/{occupancy,coverage_gain}/pretrained_*.pth` 存在 | 无 | **BLOCKED**：需要新写场景级适配器，不在本轮范围 |
| FisherRF | 仓库内没有任何代码 | 无 | 无 | **BLOCKED** |

MAGICIAN 论文报告值（附录表 6 / 7，14 场景均值：Random Walk 0.238 / 0.324、SCONE 0.535 / 0.673、MACARONS 0.649 / 0.820、FisherRF 0.542 / 0.766、MAGICIAN 0.721 / 0.921，AUC@100 / Final）只记在 `fig/table1/provenance.csv` 的参考列里，**不进入同协议主表**；论文没有 AUC@20。

## 5. 计算资源（2026-11-10 04:30–05:10 实测）

| 项 | 事实 |
|---|---|
| 可用分配 | Slurm job **3489889**（vscode-dev，gpu114，4 × L40S 46 GB，60 CPU，500 GB，到 11-14 20:35）；job 3491984（gpu107，1 × P100，另一会话在用，P100 跑不了 fvdb 规划器，本轮不用） |
| QOS | `gpu` QoS 每用户 8 GPU 上限；已用 5（4 L40 + 1 P100），另有 2 个 1-GPU 作业在排队 |
| 新提交能否启动 | `sbatch --test-only -p gpu --gres=gpu:L40:1 -t 08:00:00`：预计 **2026-11-11 02:58** 才能启动，且需要抢占 gpu117 上其他用户的 6 个作业；`-p shortgpu-l40` 显示可立即启动但同样要**抢占他人作业**（3488143、3492950–3492954，用户 osg01）。全部 L40 节点的 4 张卡都已分配。按"不抢占他人 GPU"的约束，**本轮不提交新的 sbatch 作业**，所有任务在分配 3489889 内以 `setsid nohup` 进程运行 |
| 线程上限 | `ulimit -u` 4096；一条闭环 episode 约 200–250 个线程 → 同时最多约 13 个 worker |
| 历史耗时 | V3-T0-D1 在 12 个 worker（3 / GPU）下 70 条用时 110 min（单条 16.8 min，含官方重放）；dunnottar 规划器峰值 30.5 GB，需独占一张卡 |
| 冒烟 / 回归（04:33–04:43） | 见 `OVERNIGHT_RUN_PLAN.md` §2：K4 depth 2 逐位复现 V3-T0-D1；K4 depth 1 + T1 权重逐位复现 V3-H1 |

## 6. GO / BLOCKED 汇总

| 产物 | 判定 | 说明 |
|---|---|---|
| Table 1：Ours、MAGICIAN | GO | 复用 140 条已验证记录 |
| Table 1：Random Walk、MACARONS | GO（adapted） | 各 70 条新跑，All-14 全部场景可运行 |
| Table 1：SCONE、FisherRF | BLOCKED | 留空"—"，原因见 §4 |
| Table 2 | GO | 同一批 episode 的划分统计 |
| Table 3：K4 / depth 2 | GO | 复用 V3-T0-D1 |
| Table 3：K4 / depth 1、K1 / depth 2、K1 / depth 1 | GO | 各 70 条新跑；K = 1 噪声复核（噪声图 1–3）作为 P2 |
| Fig 3、Fig 5 | GO | 与 Table 1 同一批曲线 |
| Fig 4 | GO | 单状态前向（不跑整条闭环），候选池与选例规则预先写在脚本里 |
