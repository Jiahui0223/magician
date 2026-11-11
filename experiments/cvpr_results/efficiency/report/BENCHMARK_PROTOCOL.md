# EXP8 测量协议（2026-11-11 01:45，在任何计时运行之前确定）

## 1. 被测对象
冻结的规划器子进程，与闭环中使用的完全相同，未修改任何代码或配置：

| 方法 | 子进程 | 关键配置 |
|---|---|---|
| RAYSCOUT（R4 = K4 + CDGR/D1） | table3 `kd_out_planner.py`，KD_K = 4，D-v1 策略 d1 | V3-T0@40K（sha256 7ed0ddd7…），bf16 autocast，Euler 10 步，K = 4，R10（GeoAnchor 每 10 步刷新一次），D1 两步前瞻 |
| R1 / R2 / R3 | 同一子进程，KD_K ∈ {1, 4} × 深度 ∈ {1, 2} | 深度 1 = 一步 value_cov（KD-v1） |
| RAYSCOUT 带模块计时 | `scripts/eff_prof_planner.py`（与上面同一条链，只在函数外层包同步计时器） | 只用于模块拆分；决策不变（逐步与记录比对） |
| MAGICIAN beam | task2 `t24_mag_out_planner.py --method MAG-B` | 官方 e697009 核心，beam 宽 10、深 10，官方 settings，默认精度（fp32） |

## 2. 输入状态（真实在线状态，严格配对）
- **来源：** CVPR Table 1 的真实闭环轨迹（Macarons++ 官方协议，100 步），使用每步的服务记录：深度、mask、10% 采样点、合法候选、执行历史。
- **主测轨迹：** 7 条，即按论文场景顺序隔一个取一个的场景，起点均为 0：dunnottar、bannerman、redeemer、pisa、alhambra、eiffel、barts。轨迹来自 V3-T0-D1（RAYSCOUT 自己的轨迹）。
- **配对方式：**
  - 两种方法在同一条轨迹的同一步收到完全相同的观测；各自维护自己的内部地图（P1 融合 / MAGICIAN 的 proxy 与 surface 场景）。
  - RAYSCOUT 在自己的轨迹上重放，必须逐步选出与记录相同的动作，作为确定性核对。
  - MAGICIAN 在 RAYSCOUT 的轨迹上做出的决策会与记录不同，这是预期的；我们测的只是它在这些状态下的计算时间。
- **补充：** MAGICIAN 在自己的 2 条轨迹上重放（eiffel、pisa，起点 0，来自 Table 1 MAG-B），用于核对确定性，并得到它在自身状态分布下的耗时。
- **缩放实验（Task 5）：** R1–R4 在 redeemer、eiffel 两条轨迹上测（R4 取主测数据）。
- **模块拆分：** 带计时器的 RAYSCOUT 在 redeemer、eiffel 上测。

## 3. 硬件与隔离
- 每个作业独占 1 张 NVIDIA L40S（46 GB），作业内各次运行顺序执行，同一时刻只有一个规划器进程。
- EXP7（gpu114）和其他用户的 GPU 不参与。
- 硬件、主机、GPU 编号逐次运行记录在 `summary.json` 中。

## 4. 计时口径
| 口径 | RAYSCOUT | MAGICIAN | 说明 |
|---|---|---|---|
| **决策延迟（主指标）** | `decision_s − p1_s` | `occ_s + novelty_s + beam_s` | 从"地图状态可供规划器访问"到"输出合法的下一动作"。包含：RAYSCOUT 的规划几何、GeoAnchor 刷新（发生在哪一步就计入哪一步）、两层生成、CDGR、选择；MAGICIAN 的占用场（imagined Gaussians）、novelty 渲染、beam search |
| 计算周期 | `decision_s` | `decision_s` | 决策延迟加在线地图更新（RAYSCOUT：读记录并融合进 P1；MAGICIAN：observe） |
| 往返墙钟 | 驱动端 `perf_counter` | 同左 | 计算周期加进程间通信。规划器在 GPU 计算结束后才写出应答，因此是同步的 |

- **同步：** RAYSCOUT 规划器在每个阶段边界调用 `torch.cuda.synchronize()`。MAGICIAN 用 `time.time()`，但阶段末尾的布尔索引和 `.item()` 会强制同步；它的计算周期以往返墙钟交叉核对。
- **排除项：** 模型初始化与 checkpoint 加载（`boot_s`，在 ready 握手之前）、仿真渲染、GT 覆盖率评价、物理移动都不计入。
- **主窗口与预热：** t = 10..99，每条轨迹 90 个决策，恰好包含 9 次 GeoAnchor 刷新，所以编码开销按真实频率摊销。t < 10 视为进程预热和地图初始化阶段。另外每个作业先单独跑一次不计入统计的预热运行。
- **分阶段统计：** early t = 1..20、middle 21..50、late 51..99。
- **统计量：** 均值、中位数、P95、标准差、n；每条轨迹的中位数；峰值显存（RAYSCOUT 为规划器进程的 `max_memory_allocated`，MAGICIAN 为进程结束时报告的值）。
- **配对比较：** 在同一（轨迹、步）上计算 RAYSCOUT 与 MAGICIAN 的比值，报告中位数之比、均值之比、每条轨迹的比值，以及 RAYSCOUT 更快的状态数。

## 5. 不做的事
- 不修改任何方法的搜索深度、宽度、候选数或精度。
- 不使用论文报告的时间。
- 不把生产日志（共享 GPU）的时间与本测量混用。
- 不以合成输入替代真实状态做完整决策测量；模块拆分来自同一批真实状态。
