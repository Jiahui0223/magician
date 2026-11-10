# EXP7 — MP3D-150 + UAVScenes cross-dataset parallel evaluation

> **立即执行的 Coding AI / HPC 任务书**。主方法冻结为 V3-T0@40K + D1（无 Cross-Attention）。用户已授权：无需再次征询，先审核已有代码和资源，再实际启动能安全执行的作业。**注意：此 Markdown 是执行指令，不代表作业已由 ChatGPT 提交。** 运行者必须在最终回报中附真实作业号、日志、覆盖曲线和数据。禁止捏造数值、同协议性或完成状态。

## 0. 科学目标与 P0/P1

P0：在 **MP3D Drone-150** 上用冻结的 V3-T0+D1、MAGICIAN Beam、Random Walk 以及可无歧义复用的现有 MP3D baseline 进行闭环、同场景同起点比较。输出 AUC@20 / @50 / @100 / @150、Cov@20 / @50 / Final@150，及逐步 Coverage curves（优先步 0..150）；记录完整轨迹、官方表面覆盖及每 scene 的配对差值。**150 steps 是本研究的固定短预算协议，不等同于 MAGICIAN 论文的 MP3D 5,000-step 设置。** MP3D 场景数、起点数和 GT coverage 标准由仓库冻结脚本与配置验证，不可套用 Outdoor 的 14×5。

P1：选 **UAVScenes** 作外部室外数据集，先做 scene ingestion + camera calibration + mesh/depth render + 传感器/覆盖协议 feasibility，满足 GO 才运行冻结 T0+D1 的 closed-loop **zero-shot** pilot，再运行 **相同协议** Random / MAGICIAN baseline（若能兼容）作为对照。未通过完整闭环与 baseline 配对验证的结果只能作为 *feasibility / qualitative transfer*，不能宣称是官方 active mapping benchmark 或 SOTA。

P2：保留现有 Macarons++ 14×5×100 results，不重跑已核验 T0+D1 / MAGICIAN；最终 4.2 主表以 Outdoor 与 Indoor 两个 panel 展示；UAVScenes 如 protocol 足够可靠可另作为 External Transfer 小表/附录。**先拿真实结果，再排版。**

## 1. 为什么选 UAVScenes

- 官方 GitHub https://github.com/sijieaaa/UAVScenes ；dataset https://huggingface.co/datasets/sijieaaa/UAVScenes ，包含 **4 个大场景 AMtown、AMvalley、HKairport、HKisland**，图像、LiDAR、6-DoF poses、calibration、DJI Terra 重建点云与 mesh；提供 full interval=1 和 1/5 interval=5（Hugging Face 可获取精简版）。
- NavBenchGS https://huggingface.co/datasets/UCLA-VAIL/NavBenchGS 有 mesh、3DGS、waypoints，约 17.6GB，但 HF 数据访问需先接受条款，S2E 仿真建议 Isaac Sim 5.x / IsaacLab；不适合作为本轮主路径。**不可假装机器已有授权，不能绕过 gated 下载。**
- UAVScenes 的 mesh 是多模态重建产物，可能存在孔洞、非流形、坐标偏移/尺度与相机高度错误；它并非现成的可交互 NBV/GT surface benchmark。先做 **read-only preflight**，切勿让全量下载或预处理阻塞 MP3D。
- 遵守 CC BY-NC-SA 4.0（用于学术研究、署名），落盘时保留许可和来源。

## 2. 第一阶段（立即）：仓库与 GPU 资源预审计

按下面顺序执行并写 `experiments/cvpr_results/report/EXP7_AUDIT.md`，每一步给真实终端证据：

```
set -euo pipefail
pwd
git status --short
git rev-parse HEAD
sinfo -o '%P %D %t %G %C %m'
squeue -u "$USER" -o '%i %j %P %t %M %l %D %R'
scontrol show partition gpu
find experiments -type f \( -iname '*mp3d*' -o -iname '*indoor*' \) | head -80
find work_dirs -maxdepth 6 -type f \( -iname '*.json' -o -iname '*.npz' \) | head -80
sha256sum work_dirs/exp6_v3/task0_vae/checkpoints/final.pt
```

1. 先确认 frozen checkpoint SHA256 = `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6`。`xattn.enabled=false`、R10、K4、Euler10、D1 λ0，P1 D100/S10 语义和地图尺度同训练一致；若有差异停止修改参数并审计。
2. 审计 **现有 MP3D 的 B2/D1/H0/G3** 闭环脚本、scene lists、starting poses、METRIC（surface coverage vs completeness）、坐标/TSDF、5000/150 steps 的实现；优先重用 frozen simulator 与 scoring，必要时新建 wrapper，**不改旧协议代码**。
3. 找出 MAGICIAN MP3D Baseline 在仓库中的权重/入口/适配，动作候选、安全碰撞用同一 simulator protocol。如果缺失，先以 Random 和 Ours 产出配对 pilot，记录 MAGICIAN blocked 的具体原因；不要用论文 MP3D-5000 指标填 MP3D-150。
4. 统计现有 MP3D jobs 实测每步时间与峰值显存；先 1 scene×1 start×2–5 steps smoke test，查看真实硬件情况后才大规模排队。每 GPU job 的 CPU 建议先 8–12 core，不做“1 GPU 占 64 CPU”调度。
5. 检查是否已有 4 张已分配 L40、是否获准再申请 4 张；不能凭 `sinfo` 的总空闲数保证立刻获得第二组 GPU。**不要在登录节点执行 GPU-heavy tasks。**

## 3. 双轨任务划分与资源优先级

**Pool A（已有 4 GPU，P0）：MP3D-150。** 一次只启一个资源受控的实验启动器，优先 Ours T0+D1 与 MAGICIAN（均同场景同起点），Random 可以放 CPU 队列如果模拟器 render 不需 GPU，否则在 GPU pool 分片运行。按实际峰值显存拆成 4×1 GPU、2×2 GPU、或1×4 GPU job；不要默认“一条轨迹一定只需一张卡”。将各 scene×start 作为可恢复独立 unit，Slurm array 限制最大并发，`time` 按实测+裕度。

**Pool B（另申请 4 GPU，P1）：UAVScenes 外部验证。** 在 GPU 作业排队期间，**使用 CPU/download 作业**完成轻量数据下载及 mesh/pose/calibration 检查；在 GPU pool 就绪后做单场景渲染/模型前向的 smoke test。无 GO 不允许启动 100/150 step 大规模 rollout。若 Pool B 申请不到，优先让 Pool A 完成 MP3D，同时通过 CPU 下载/校验 UAVScenes；如 UAVScenes 早期被 BLOCKED，可把 Pool B 改作 MP3D 剩余场景、MAGICIAN 或可视化，不让卡空闲。

**调度模板（仅在真实 `sinfo` / 集群 policy 合法的前提下使用，按实际入口替换 `<...>`）：**

```
# 在仓库生成带完整路径的任务清单：每行 scene,start,method,protocol
# 分片 array 的工作由一个 seed-stable wrapper 执行；不允许同一 episode 双重写入
sbatch --job-name=cvpr-mp3d-150 --partition=gpu --nodes=1 \
  --ntasks=1 --cpus-per-task=12 --mem=96G --gres=gpu:L40:1 \
  --time=08:00:00 --array=0-<N_MINUS_1>%4 \
  --output=work_dirs/cvpr_results/exp7/logs/mp3d_%A_%a.out \
  --wrap='bash experiments/cvpr_results/exp7/scripts/run_mp3d_array.sh "$SLURM_ARRAY_TASK_ID"'
```

**模板不是立即可运行脚本**，须创建 wrapper，且**先检查**节点分片、GPU L40/L40S 型号、Slurm array limits 和单任务显存/CPU 需求。`--array` 任务数量及并发上限根据 manifests 才能展开。为申请第二池单独提交 UAV job，但用户没明确已获额度之前不声称已经分配。若 `--gres=gpu:L40:1` 不适用，以集群实际 GRES 为准。不要用多个嵌套 `salloc` 占位。

## 4. MP3D-150 实验规范

- 建立一个**冻结** `mp3d150_manifest.csv`，字段 `scene_id,start_id,method,checkpoint_sha,protocol_id,model_config_hash,sim_config_hash,random_seed,output_path,status`。同场景同起点/同模拟器动作空间，尽可能同 camera model、合法候选、碰撞策略、观测预算、真实深度和评价器。
- 场景要么采用 MAGICIAN 官方五个 MP3D mesh（与 paper code 同），要么采用仓库已有 indoor eval 子集；**不可混用这两套又不说明**。先读取 `data/mp3d_magician/` 和现有 project scenes、起点缓存；若选择五个官方场景，记录 exact scene IDs 和起点构造。
- 所有指标使用**相同的固定真实表面评价域**；严禁把 Ours 的局部 P1 window 当作评价覆盖分母，严禁拿 observed voxel volume 代替 official surface coverage。若 MAGICIAN 官方 MP3D 只输出 completeness 与 accuracy，必须建立并审核可对全部方法同样计算的 `C[t]`。没有可靠 evaluator 就报告结果不可比，不能生成虚假 AUC。
- 固定 `t=0..150` 151个覆盖值；`AUC@T = trapz(C[0:T+1])/T`，`Cov@T=C[T]`；每 scene 先聚合 starts、再对 scene 等权；需要原始序列，不能只导出最后一个数。
- Round 1 P0 做 1 scene × 1 start × 三法，同环境、同初始化，核查 Ours/MAGICIAN/Random 的 action合法率、coverage 曲线、地图和耗时；GO 后才扩展全 scenes/starts。
- Round 2 真实并发：优先 5 scene × 5 starts × (Ours + MAGICIAN)（**仅当此前已存在且可核对对应起点**）；Random 并行 CPU/剩余 GPU。若五场景/五起点尚未定义，则选择由存档协议允许的统一列表并冻结。
- Round 3 产出 per-scene 指标表和 paired plot，检查 Test/OOD 的 scene provenance；MP3D 训练轨迹存在并不意味着评估场景未见。所有 MP3D 结果须写 clearly seen/heldout 状态。

## 5. UAVScenes 同步验证规范（严格从 feasibility 逐步 GO）

### Step 5.1 Data acquisition（优先最小可用包）

1. 首先检查已有数据目录/共享盘缓存、许可证、磁盘配额和下载工具。GitHub README 提供 OneDrive/Google Drive、Hugging Face interval=5 的下载入口。只下载**一个场景**的 `Mesh.ply`（或 `terra_ply/` 必要 mesh blocks）、校准/位姿索引、少量标定图像；不要一开始下载全部 LiDAR 与语义标签，更不要下载四个场景全部体积。
2. 在官方下载页验证每个资产的实际路径/大小和可无交互下载情况；记录实际 URL、hash、授权。HF 可在条件允许时使用 `hf download sijieaaa/UAVScenes --repo-type dataset --include '<verified file pattern>' --local-dir <destination>`；**pattern 必须从真实 repo 文件清单读取，不可凭 README 里的路径猜测**。
3. 建 `uavscenes_inventory.csv`，包含 scene, mesh filename, vertices, faces, bounding box, camera frames, intrinsics, extrinsics, align errors, units, source/license。

### Step 5.2 Geometry/sensor validation（No GPU heavy job before GO）

- Mesh 可载入、三角面非空、bounds 有限且尺度可信；测非流形、孔洞、法线与射线穿透的比例；相机外参和 mesh 坐标对齐，渲染 silhouette 与提供的标定图/点云大致对应。
- 生成候选射线/几何投影：使用同一套投影约定，检查 depth near/far 与单位尺度，随机抽样对已知 Mesh 射线渲染图进行 debug，不能把原 LiDAR sparse depth 当成官方完整真实深度。
- 建立场景可通行的**相机候选子集/动作图**：设定起点、候选移动、边界与障碍判定，禁止让无人机从建筑物内部穿越；若只有已录制轨迹而无法生成新位置视角，则**不能直接进行 active NBV 规划**，只能做 fixed-pose future depth predictive transfer。
- Coverage evaluator 固定 GT reconstructed mesh surface 的采样/区域/点密度以及 tolerance，保存每 step `C`。要区分“mesh proxy coverage”与 Macarons++ 官方 coverage，不做跨 dataset 数值等价比较。

### GO / NO-GO 决策

- **GO-DEPTH**：有靠谱的重建 mesh、相机标定且渲染深度与有效区域一致 → 可做冻结 T0 future-view prediction / visualization（1 scene 多候选），并计算可验证的预测误差。
- **GO-NBV**：在 GO-DEPTH 基础上，有合法行动图、GT mesh-based observation simulator、无泄漏的 online planner、可重复的 coverage evaluator、可运行的公平 baseline → 可开始 1 scene×至少2 start×100-step pilot，再看 GPU 剩余与扩展。
- **NO-GO**：网格缺失、坐标不一致、模型尺寸 OOM、无合法候选、无法公平比较 → 停止在线 closed-loop 声明，保留具体日志/截图和可复用数据资产，切回 MP3D/P0。
- 新数据集严格 **zero-shot**：固定训练好的 T0 和规划配置，严禁为追求高 coverage 用 UAVScenes 数据微调或选择最优场景；可以调整为适配数据集单位/相机的确定性配置，但必须公开并对 baseline 一致。

## 6. 结果目录与真实交付

```
experiments/cvpr_results/
  exp7/
    configs/  scripts/  manifests/  report/
  fig/
    table1/   # indoor/outdoor main results; includes prior Macarons++ audited results
    fig3/     # indoor/outdoor curves 150/100 steps, optionally AUC@T
    fig4/     # MP3D and UAVScenes visualization if valid
    external_transfer/ # UAVScenes feasibility/pilot, not official benchmark
work_dirs/cvpr_results/exp7/
    logs/  jobs/  episodes/mp3d150/  episodes/uavscenes/  downloads/  visuals/
```

要求：1) 每条闭环 episode 保存 `C[0..T]`, camera actions, starting pose, checkpoint/config/protocol hashes, seed, walltime/device, status, actual filenames, ground-truth access boundaries。2) 分场景聚合和全局 scene-equal mean，包括 CI；不能用未完成案例冒充全部平均。3) 输出 `MP3D_150_RESULTS.md`、`UAVSCENES_FEASIBILITY.md`、`JOBS.md`、`VALIDATION.md`、`MISSING.md`。4) 对最终 PDF/PNG/CSV/LaTeX 由统一 summary 生成；优先保存可审计结果，排版后做。

## 7. 执行优先级与异常处理

1. **立即审计、启动 MP3D 1-scene smoke + baseline pairing；同时启动 UAVScenes minimal download on CPU/data node。**
2. **MP3D smoke PASS 后立刻提交分片 full 150-step MP3D**。GPU Pool A 维持饱和，但不超过节点显存与用户配额。多卡不一定多进程越快，依据前 10–20 步的真实 walltime 决定并发。
3. **Pool B** 申请4张卡，若尚未分配，先做 CPU 的数据校准；获得资源后 GO-DEPTH/GO-NBV 检查与外部 pilot。若 NO-GO 或新数据集准备进度不足，转为 MP3D 次级 baseline/缺失 start 或稳定可视化。
4. **对任何输入缺失、OOM、超时、Slurm 队列未调度**，写客观诊断、运行状态和最小修复（并保留已完成 episode），不要悄悄替换数据/模型/评分方法。
5. 关键检查点：第一个 smoke 完成、MP3D full jobs 真正排队、首次 episode 完成、UAVScenes feasibility GO/NO-GO、最终 output validation。每个阶段需要完整 AI 回复、绝对路径、作业号和样本数。**禁止编造运行完成时间或8小时内必然完成的保证。**

## 8. 给执行 AI 的直接任务

请从真实终端开始，按照 §2–§7 立刻执行，不要停留在拟定方案。先用 `sinfo/squeue` 核对两池 4 GPU 的资源与实际可运行任务，审计已有 MP3D 150-step 环境、脚本及 T0 frozen checkpoint；创建有实义的 MP3D launcher，运行1个配对 smoke test，校验完成后启动能够实际跑的分片闭环作业。同一时间在 CPU/data 节点下载 UAVScenes 1 场景最小资产，校准并尝试深度渲染；通过 GO-DEPTH/GO-NBV 后才进行 UAV 迁移实验。每一步都记录真实命令、Slurm Job ID、进度、错误和输出；禁止宣称无法确认的实验已经运行。优先保存可用结果并复用 Macarons++ 已核验数据，最后报告已完成/运行中/阻塞的确切数量。
