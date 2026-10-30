# 给代码 AI 的执行指令：V2 归档推送 → EXP6 V3 Task0 / Task1

请以仓库实际的 `plan.md` 为基本项目规范，并将本次明确授权的任务视为最新阶段要求。**先完成 V2 归档并 GitHub Push，然后提供简短回复，再自动开始 `experiments/exp6_v3/` 两个主任务的实现、测试与训练；除硬阻塞外，无需每一阶段等待我的人工审核。**

我同时提供三份完整文档，请完整阅读、核实仓库代码后执行：

- `EXP6_V3_FROM_SCRATCH_PLAN.md`：科学目标、数据与 VAE/Flow/X-Attn 架构。
- `EXP6_V3_REPORT_TEMPLATE.md`：统一报告结构。
- **`EXP6_V3_PLAN.md`：本次新增的工程目录、Git、自动测试、GPU/Slurm 与执行规范。**

请不要直接执行旧草案里的 Frozen Seen2Scene VAE、双 Encoder、Adapter/Gate、V2/E2 checkpoint 微调等方案。当前确定的是：**Simple 3D VAE + Flow，两组独立 Scratch Training，T1 额外加入一处 Cross-Attention**。

## Task A：先整理 V2，必须真正 Push 完成

1. **先检查仓库**：`git status -sb`、`git branch -vv`、`git remote -v`、`git log -5 --oneline`；阅读当前仓库 `plan.md`、`github.md`（如果有）、V2/EXP5/E2 代码与报告。不要假设分支名，不重写 Git 历史。
2. 梳理 V2-Mix、EXP5 Task3 B0/B1/B2、BAND Density、E2 G1/G2/G3 的已存在代码、配置和真实结果；索引 README 和复现入口；不能为归档而改模型、评估数字或覆盖已有结果。
3. 仅选择需要提交的代码、轻量实验文档和报告。忽略/排除数据、缓存、临时 checkpoint、大型日志、私有配置和任何 token/secret。运行 V2 的现有回归与最小 smoke，检查 `git diff --check`。
4. **Git Commit Subject 一律为 `[V2][xxxx] xxxxx` 格式**，例如：`[V2][archive] Freeze V2-Mix and E2 G3 results`；后续 V3 使用 `[V3][task0] ...`、`[V3][task1] ...`。不要加入 Claude co-author footer。
5. 将已核实的当前分支推送至已核实的 remote；**禁止 `--force`、随意 rebase、hard reset、删除无关文件**。推送后验证远端 SHA 与本地 HEAD 对应。如果没有权限或冲突，准确报告，不声称成功，也不要悄然开始有风险的 V3 大改。
6. **V2 Push 一完成，先给我一个简短阶段回复**：branch、remote、commit SHA、commit title、测试结果与归档文件概要、剩余风险。此后立即继续 V3，不必等我回复。

## Task B：创建独立 EXP6 V3，保持 V2 式结构化代码

请创建 `experiments/exp6_v3/`，其中 `task0_vae/` 和 `task1_xattn/` 为两个明确独立任务。**两者内部必须各有 `report/` 文件夹**，每个任务分别存 `AUDIT`、`G0_TESTS`、`TRAINING`、`G12`、`G3` 的详细 Markdown 报告与图表路径。根目录的 `reports/V3_COMBINED_REPORT.md` 用于 T0/T1 的最终统一比较。配置、代码入口、自动化脚本、单测、输出/日志分开；公共模块通过 `shared/` 或项目原生 package 复用，严禁复制两份大训练器、添加无意义辅助 head 和过多嵌套 helper。大数据与模型文件放在被 Git 忽略的 `work_dirs/exp6_v3/`。

同步更新 `experiments/README.md` 的 EXP6 索引，并按 `plan.md` 的理念在 `idea.md` 补充简洁研究目标（如有必要，不覆盖旧研究）。旧 `plan.md` 的 HDSAM/`exp6_replica` 属于历史阶段，不能覆盖新 `exp6_v3`；本次要求明确 supersede 旧“暂不提交、不 Push”的规则，其余通用规范继续遵守。

## Task C：具体模型与数据要求（两组不可混淆）

- **数据**：仅使用 MP3D + Macarons++ 的合法 scene-level 训练数据，复用已验证的 split、候选轨迹和评估协议。训练时输入为截至 `t` 的真实历史 Depth 融合构建的 100% BAND **Partial TSDF**，保留 `EMPTY/BAND/UNKNOWN` 和物理 voxel scale；候选未来 GT 只用于训练标签。DistP1 Scoring 始终读取原始 10% Sparse P1。预测结果不写回历史地图。
- **Task0 V3-VAE**：单路 Simple 3D VAE Encoder 替代 3D U-Net，训练期 Decoder 重建历史观测几何。损失 `FM + λrec · MaskedRecon + β · KL`，`z_rec=μ+σϵ`，Flow condition 使用 `z_cond=μ`。维持 V2 的 Ray Traversal、Ray Encoder、Future Depth/HIT Decoder、K=4 Flow 与 NBV Scoring 的物理语义。**3D VAE、Ray Encoder 和 Flow 全部 Scratch 初始化**。
- **Task1 V3-VAE+XAttn**：使用与 Task0 相同的结构和损失，只额外在 Flow U-Net 的单个低分辨率层放 Camera-Ray Query → 3D VAE Spatial Memory Key/Value 的 Cross-Attention；不另加 Adapter/Gate、第二条 3D Encoder 或新评分器。**Task1 也必须从头训练，不读取 Task0 训练后的权重**。
- **公平性**：T0/T1 共同层尽量匹配初始参数，保持相同训练样本索引、随机种子、FM 噪声控制、optimizer steps、effective batch、LR schedule、KL/recon 权重、EMA、Dev checkpoint 选择规则。保存 hash 与验证日志。
- **训练轨迹**：一个轨迹是一条在独立 Scene 中按时间顺序真实执行/渲染的观测序列；一个监督记录来自 `(scene, trajectory, t, candidate)`，不是一条轨迹只产生一个样本。核对训练/验证/测试是否场景隔离，记录 MP3D 和 Macarons++ 的独立场景、轨迹、候选记录数。

## Task D：实现后小测试，允许 AI 据实动态优化

不必在每个微小超参上找我确认；请先做必要的简短 G0/smoke/profile，再用结果固定正式训练配置：

1. **TSDF 真实性**：未来 GT 泄漏、UNKNOWN≠EMPTY、SIGNED SDF、BAND/EMPTY precedence、相机/坐标系与不同 voxel size、tile stitch 及 Ray Feature 对齐。
2. **VAE 正确性**：Encoder μ/logvar、reparameterization、Masked Reconstruction、KL 归一化、posterior collapse、FM/Recon/KL 对 Encoder 的梯度；小样本重建可视化与 tiny overfit。
3. **XAttn 正确性**：Q 随 pose/ray 改变，K/V 只读取当前历史 Scene Memory，invalid mask 生效，局部 tokens/坐标正确，训练时不会因错误 cache/detach 截断梯度。
4. **训练运行**：import/syntax/tests、单卡和 4-GPU DDP、BF16/AMP、checkpoint restore、数据 loader、训练吞吐、每卡显存与磁盘占用；100–300 steps profile 决定 patch size、tokens、gradient accumulation、合理 LR、KL warmup 和初始总训练预算。

允许 AI 根据实际证据修复数值问题、合理选择 tile/downsample、patch、token budget、loss 权重的初始校准及 training schedule，但必须**两组统一，在正式训练前冻结并写入 `protocol.md` 与 task 报告**。开始正式训练后，不能暗中调整模型、loss、数据域比例、测试集或只对某一组优化；确需改动应按协议记录并公平重跑。

## Task E：使用当前 4 + 申请另 4 张 L40/L40S，自动训练

现在已有 4 张 GPU 分配，但请实际检查当前用户的 Slurm allocation、job ID、node、型号和剩余 walltime，不能假设已永久占用；如可用，请按集群要求通过 `sbatch` 再申请 4 张。**总并发预算最多 8 张 GPU**。不同实验分别各使用 4 张卡、独立 DDP job ID/log/checkpoint/seed；两者 G0 和公平配置全部锁定后，**允许 Task0 4 卡 + Task1 4 卡同时完整训练**，提高资源利用率。若排队或环境限制，就按顺序运行，不改变实验定义。不要直接 SSH 禁止登录的 GPU 计算节点，应使用合法 `srun/sbatch`。

两组都是从头训练，因此不要使用 E2 fine-tuning 的 5000 steps 直接作为 scratch 预算。以项目先前提出的 ~60k steps 为规划上限候选，经 profile 和 Dev 曲线确定预冻结预算、周期 checkpoint 和 Early Stop；保存 optimizer/EMA/RNG/scheduler 支持自动 resume。完整运行使用真实 stdout/sacct 证据，不能把 `sbatch submitted` 当作训练已成功。

## Task F：评估与报告，无需再次询问

- Task0/Task1 的 `report/` **边跑边记录**：输入/输出 tensor shapes、数据真实数量、KL/recon/FM 曲线、Loss/grad、G0 单测、G1 known/unknown/truly-unseen、G2 regret/top1、G3 AUC/final/early Cov、逐 Scene 差异、bootstrap CI、真实训练耗时/显存与所有失败修复。
- 两模型在同一协议和同一评价场景上完成 G1/G2/G3。G2 失败但仍跑 G3，必须注明 exploratory，不得事后宣布 G2 PASS。
- 最终更新 `reports/V3_COMBINED_REPORT.md`，同时对比 **B2、E2-D100、V3-VAE、V3-VAE+XAttn**，分别报告 MP3D 和 Macarons++；保留负结果、训练预算差异和真实配置。旧 V2 不重训，当前也不做额外 Indoor-only/Outdoor-only/AE-only 消融。
- 完成后给我清晰简短的结论：V2 Push 信息，T0/T1 训练完成与否、每组最优开发集 checkpoint、两域的 AUC/Final、模型效率、异常与下一步建议。

## 执行边界

**请主动完成常规工程调优，不必逐步等我；但遇到以下问题必须停下并报实情**：敏感文件将被 Push、远端提交冲突、训练/测试场景泄漏、TSDF 世界坐标严重不一致、梯度或模型无法正常训练且短程修复无效、Slurm 资源无权限、重大不可恢复 OOM、训练预算超出已确定上限。不得删除历史 EXP5/V2 结果或无关文件，不得用测试 AUC 做参数搜索，不得虚构运行指标。

**立即从 Task A（V2 整理、GitHub Commit & Push）开始。完成该任务后必须先向我做一次简短阶段反馈，再按上述要求自主继续 EXP6 V3。**
