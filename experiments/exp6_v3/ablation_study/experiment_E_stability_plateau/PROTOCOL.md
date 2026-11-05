# Experiment E — 稳定性比较与 colosseum 低覆盖平台诊断（协议 E-v1）

**冻结时间**：2026-11-04 15:10（MST），在任何正式诊断运行之前。仓库基线 `e878e3b`。

**本实验不做的事**：
- 不训练；不扩大搜索；
- 不修改 H3 / D1 / YBEST-H3 的代码、配置或历史结果；
- 不写 Table 1。唯一例外是已注册行 V3-YD-COMBINED-Y50K-D1 的补齐，走 Table 1 自己的 `t1_queue.sh`。

## 冻结的对象（sha256 前 16 位）
| 对象 | sha256 |
|---|---|
| `experiments/exp6_v3/H/scripts/h_policy.py`（H-v1） | ccfdf4af2783be3f |
| `experiments/exp6_v3/H/FROZEN_CONFIG.yaml` | ab8aff7077dd6038 |
| `.../experiment_D_h3_lookahead/scripts/d_policy.py`（D-v1） | c8418279771e787e |
| `.../experiment_D_h3_lookahead/configs/d_config.yaml` | e53c230db866e821 |
| V3-T1 40K `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt` | 639c9d3ae3f3c069 |
| Y50K `work_dirs/exp6_v3/Y/runs/main/ema_050000.pt` | 64cea1d62e7b2a30 |

**推理设置**（与 Table 1 的 V3 系列逐项相同，由冻结的模拟器与规划器子进程保证，本实验不传入任何覆盖参数）：
- b2：D100 输入 + S10 band_dil 评分；
- R10，K = 4，Euler 10；
- MAGICIAN settings 网格动作空间与官方合法列表；
- 固定起点 0–4，100 步；
- 官方 MAGICIAN S10 覆盖。

## Task 1：D1 vs YBEST-H3 稳定性（只用已有 episode）
- **数据**：Table 1 规范记录（官方覆盖），模型 D1、YBEST-H3、H3、YD、H0（V3）、Y50K+H0，参照 B0、B2、MAG-B。
- **指标**：AUC@20 / AUC@50 / AUC@100（trapz / k，与 Table 1 相同）、Cov@20、Cov@50、Final。
- **分组**：Train-6 / Dev-5 / Test-3 / All-14，另加派生分组 Unseen-8 = Dev + Test。
- **统计**：
  - 场景级配对差，bootstrap 95% CI（10k，seed 0），场景胜负，同起点 episode 胜负；
  - 集中度：逐一去掉场景、去掉 |d| 最大的两个场景、中位数；
  - Final 退化事件：同起点比 H0 低 0.05 以上；
  - 离散度：Final P10、最小值、场景内起点间标准差。
- **2 × 2**：{40K, 50K} × {λ 0.25, λ 0} = {H3, YBEST-H3} × {D1, YD}。报告微调效应、去风险效应与交互，联合场景 bootstrap。
  - YD 只有 Dev-5，Train / Test 正在用 Table 1 流程补齐（45 条），补齐前 2 × 2 只在 Dev-5 上报告并标为 partial。

## Task 2：colosseum 诊断
**分类规则**（事先固定，用于计数，原始数值同时报告）：
- 低覆盖平台 episode：Final < 0.70。现有 colosseum 数据中，两种状态之间的空档是 0.665–0.749。
- 分支续跑后“脱困”：续跑的 Final ≥ 0.75。

### E-A：已有数据分析（CPU，只读）
- 曲线、执行轨迹（settings 网格索引 → H 的 pose table 中心与朝向）；
- 与 H0 / H3 的首次分叉步、平台开始步；
- 重复访问、往返、位置范围、零增益步；
- 规划器日志中每个候选的预测值：h_mean、h_std、q、best_c2；
- 已执行动作的实现值：用 svc 档案中下一帧的 S10 点，按 value_cov 的同一定义计算。

### E-B：GT 反事实（评估侧，MAGICIAN 环境，`scripts/e_oracle.py`）
- 对 colosseum 5 个起点 × {H0, H3, D1, YBEST-H3} 共 20 条 episode，逐步重放（断言覆盖与合法列表一致）。
- 在每一步对每个合法候选，用冻结的 EXP3b snapshot / restore 计算：
  - 真实的官方覆盖增益 dcov；
  - 真实视图在 value_cov 定义下的值，并拆成落在 GT 评估点附近（相关）与不在附近（无关）的部分；
- H3 系列另外检查规划器想象的第二步 c2*：在真实执行 c1 之后它是否合法，以及真实的两步增益。
- 断言：执行动作的反事实增益必须等于真实增益。
- 每 10 步保存逐 GT 点的覆盖掩码，用于定位平台期缺失的区域。
- **这些 GT 量只用于离线诊断，从不进入任何规划器。**

### E-C：分支续跑（state swap，`scripts/e_branch.py` + `e_sim.py`）
- **方法**：复制源 episode 前 T 步的模拟器检查点行、svc 记录与规划器日志到新目录。冻结模拟器重放这 T 步（逐位校验 svc），规划器在 T 处恢复，从第 T 步起换成另一策略。
- **回归**：同策略分支必须逐位复现源 episode；rep 0 的 h0 必须复现 Table 1 V3。
- **正式分支**（colosseum）：
  - **B1**：H3 低平台起点 s0、s1、s2、s4，T ∈ {10, 20, 50}，续跑策略 h0 与 d1；
  - **B2**：YBEST-H3 低平台起点 s0、s1、s4，T = 20，续跑 Y50K 权重的 h0（权重不变，只换决策规则）；
  - **B3**：D1 的 s1，T ∈ {20, 50}，续跑 h0；
  - **B4**（反向）：H0 的 5 个起点，T ∈ {10, 20}，续跑 h3。
- **判读**：
  - 如果从低平台状态换成 h0 或 d1 后普遍脱困，平台由策略的选择维持，而不是状态本身不可逃逸；
  - 反之则支持可达性或状态陷阱。

### E-D：噪声重复（`e_sim.py --rep`）
- colosseum 5 个起点 × {H0 = h0, H3 = h3, D1 = d1, YBEST-H3 = Y50K + h3} × rep ∈ {1, 2}，共 40 条。
- rep 只改规划器噪声键，GT 观测种子不变。
- **目的**：判断低平台是起点决定的，还是对微小分数差异敏感的随机事件，并估计各策略落入低平台的频率。

### 执行顺序与资源
- 顺序：E-A 与 E-B 先行，然后 E-C，最后 E-D。
- 资源：gpu114 的 4 张 L40S，每张最多 3 个进程，dunnottar 独占。
- **不补跑 Test 集以挑选结果；不根据结果修改任何阈值。**

## 偏差记录
（运行中如有偏差，在此追加，注明时间与原因。）

- **15:40 资源**：gpu114 显存压力导致 B1_H3s2_T50_d1 的规划器在 t = 84 OOM。检查点保留，用同一机制从 t = 84 逐位续跑（B1r_H3s2_T50_d1）；不改变任何科学设置。
- **15:45 新增参照（E-B 扩展）**：为判断 colosseum 的现象是否特有，对 pisa s1（H0 / H3 / D1）、pisa s4（H0 / H3）、alhambra s0（H0 / D1）做同样的 GT 反事实与预测重建。只做诊断，不进 Table 1。
- **15:55 放弃的想法（未运行）**：曾准备“只计场景盒内新增”的变体 E-BOX-v1，动机是 GT 视图中无关新增 99% 在盒外。
  - 运行前检查发现，规划器的预测值本来就有 97% 落在盒内（规划域 ≈ 场景盒），该变体几乎等于不改。
  - 盒外占比高，只是 e_oracle 的 GT value 没有施加规划域限制所致。
  - 因此删除该变体代码，不运行。之后与预测对比时，GT 一侧统一用“相关部分”（GT 评估点 1.0 以内）。
- **16:10–16:50 新增离线分析**（只读，CPU，不改变任何运行）：在看到预测审计结果后加入，属于事后分析，在报告中按事后分析表述。
  - `occlusion_check.py`：执行视图的预测 cell 与下一帧真实观测的视线遮挡检验；
  - `refute_analysis.py`：反证记忆的离线一步重打分；
  - `occlusion_rescore.py`：用过去观测作遮挡物的离线一步重打分。
- **16:30–16:50 Table 1 YD dunnottar**：
  - 5 条在 gpu114 上因与 E 进程同卡而 OOM，3 次重试用尽后被放弃。
  - 处理：删除锁文件，failed 目录全部保留，每卡一条重新排队。
  - 其中 s2 再次 OOM，原因是我把 E 的分支续跑留在了同卡（判断错误）。
  - 已停掉该 E 进程，s2 用冻结机制从 t = 35 逐位续跑。
  - 详见 Table 1 的 `logs/requeue.log` 与本目录 `EVENTS.jsonl`。
