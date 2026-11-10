# CVPR overnight experiment execution instruction — V3-T0 + D1

> **交给仓库 Coding AI 的完整执行指令。** 目标：使用约 8 小时可用计算窗口，尽最大可能生成可以审计、可复现、可直接插入 CVPR 的 Table 1/2/3 和 Figure 3/4/5。严格禁止伪造任何数据、假定未确认的实验已完成、未经核实地混用不同评估协议。**不需要再问我确认，立即完成预检后自行执行可行部分，直到作业提交并产生实际结果**；不得承诺后台运行而不真正提交作业。

## 0. 核心约束（最高优先级）

- **论文主方法固定**：V3-T0@40K + D1，`xattn.enabled=false`，K=4、Euler=10、D1 λ=0、D100 用作编码输入、S10 观测表面作 Coverage 新颖性判定、R10 latent 刷新。**不是 T1**，不使用 Cross-Attention 或 Scene Memory。
- **checkpoint**：`work_dirs/exp6_v3/task0_vae/checkpoints/final.pt`，预期 SHA256 `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6`；运行前核验。不能替换为 T1 权重。
- **现有主方法验证结果（待文件核验）**：T0+D1 All-14 14 scenes × 5 starts: AUC@20=0.420、Cov@20=0.646、AUC@50=0.633、AUC@100=0.763、Final=0.911（用户最新正式截图）。MAGICIAN Beam 同协议：0.307、0.537、0.544、0.711、0.906。截图仅是待核验索引，不是原始 episode 曲线；**只能在核验70条完整记录、protocol和重放一致后作为已验证结果**。
- **评估**：Macarons++ 100 steps、固定 14 scenes × 5 starts；`C[0..100]` 的 MAGICIAN 官方 coverage；`AUC@T = np.trapezoid(C[:T+1], dx=1) / T`；`Cov@T=C[T]`；每 scene 先平均 5 starts，再 scene 等权（对其他子集同理）。官方 Coverage 可能下降：绝不能 `cummax` 或人为平滑原始数据。
- **引用和公平性**：MAGICIAN 论文报告值不得直接混入统一协议主表。所有 baseline 保留来源（复现/公开原始曲线/论文数字）及每项是否协议一致。随机、SCONE、MACARONS、FisherRF、MAGICIAN 的命名、权重、算法、合法动作与每一步评价方式须逐项核查。
- **数据泄漏**：GT 仅允许模拟器做合法动作碰撞判定、标签生成（离线训练）和官方 Coverage 评估，在线 Planner 不得读取 GT 或未来实际深度。严格遵循原 LeakGuard。
- **资源**：不抢占他人 GPU；检查 Slurm 配额、节点、已有 job 和 8h 额度。重用验证的实验输出；不重复训练 T0。运行中的任务要可断点恢复、避免重复提交。
- **产物要求**：每阶段写独立完整 AI 记录（audit、plan、execution、final），报告真实 Slurm ID、状态、stdout/stderr、真实路径、完成计数与失败原因。不能用“已完成”替代检查。

## 1. 预检查与资源审计（立即）

1. 检查 `git status`, commit SHA；只读检查 `V3_T0_D1_MODEL_ARCHITECTURE.md`、`experiments/cvpr_results/table1/`、`work_dirs/cvpr_results/table1/episodes/`、T0/T1 冻结配置、D1 策略、官方重放脚本 `t25_out_cov.py`、任务脚本入口及日志。
2. 枚举现有 `method × scene × start` 的真实完整 episode；列出每行 70/70 或缺失数量。校验曲线长度101、数值有限、编号、scene、start、官方重放一致性。保留 `manifest.csv`（不可仅凭汇总截图推定完整）。
3. 读取训练数据清单：Macarons++ Train 6 / Dev 5 / Test 3 的**精确场景 ID**、T0 checkpoint 训练与选模场景；审计 MAGICIAN 使用的预训练/微调权重与其 scene provenance。若无法验证 MAG 未见 Test，只能写 **“performance on T0-held-out scenes”**，不能宣称两者从未见过相同场景。
4. 检查可用 GPU 节点和队列、Slurm job 时间限制、可分配 GPU 数、每条 episode 历史实际时长；**先做 1 条 smoke test**（或已审计的近期成功记录），给出排期估算。T0+D1 9–13 min/episode，70 条串行至少 10.5–15.2h；8h 全部完成依赖并行/现有缓存。
5. 检查 Random、SCONE、MACARONS、FisherRF、MAGICIAN 每个 baseline 的代码/权重/episode 覆盖缓存是否存在；缺失须单独列明，不允许假设可运行。复现动作集合需与 frozen protocol 一致；适配者明确标注 `adapted`。
6. 写 `cvpr_results/OVERNIGHT_AUDIT.md`、`cvpr_results/OVERNIGHT_RUN_PLAN.md`，按每方法每场景可执行性做 GO / BLOCKED 判断，不改变冻结协议。

## 2. 目录与数据契约

沿用已有仓库实验根目录 `experiments/cvpr_results/`，全部最终可提交图表在：

```
experiments/cvpr_results/
  fig/
    table1/  table1_all14_full.csv  table1_paper.csv  table1.tex  table1.pdf  table1.png  provenance.csv
    table2/  table2_splits.csv  table2.tex  table2.pdf  table2.png  split_manifest.csv
    table3/  table3_ablation.csv  table3.tex  table3.pdf  table3.png  ablation_manifest.csv
    fig3/    coverage_curve.pdf  auc_curve.pdf  fig3_combined.pdf  fig3_combined.png
    fig4/    hypothesis_depth.pdf  hypothesis_depth.png  panels/  visualization_manifest.csv
    fig5/    per_scene_radar.pdf  per_scene_radar.png  per_scene_auc20.csv
  manifest/  episodes.csv  protocol.json  provenance.json  sha256.txt
  report/    AUDIT.md  RUN_LOG.md  RESULTS.md  VALIDATION.md  MISSING.md
  scripts/   [reproducible extraction/aggregation/plotting runners]
work_dirs/cvpr_results/
  overnight_20261110/  logs/  jobs/  episodes/  intermediates/
```

用户口头说 `cvpr_results/fig`：以上选择沿用仓库已存在的 `experiments/cvpr_results` 根；若实际 repo 已有统一 `cvpr_results` 根，先检查再统一使用一个，不要创建两个互不关联的结果树。Fig1 为 teaser、Fig2 已定稿，**Fig3 = curves，Fig4 = multi-hypothesis，Fig5 = radar**。所有导出的 PDF 使用矢量线条和可嵌入的系统字体，PNG 至少300 dpi，保持 source script 与数据。

每条 raw record: `method_id, method_display, scene_id, start_id, split, protocol_id, checkpoint_sha, seed, K, planning_depth, score_kind, noise_key, c0..c100, trajectory_path, source_path, official_replay_hash, run_status`。不能丢弃失败 episode；分开 `COMPLETE/INCOMPLETE/BLOCKED/FAILED`。无效数据不参与聚合；结果必须标 `n_completed/n_expected`。禁止对缺失 scene 或 start 进行补值。

## 3. Table 1: Main benchmark, all-14

- 行：Random Walk、SCONE、MACARONS、FisherRF、MAGICIAN Beam、Ours (V3-T0+D1)。先检查公开实现/冻结复现和缓存；不可把论文里仅有 AUC@100 的统计值当作 AUC@20。
- 内部**完整计算**：对每条 0..100 覆盖曲线，计算 AUC@T / Cov@T，`T={20,34,45,50,60,78,80,90,100}`（包含用户指定节点以及具有明确科学意义的50、80）。另存完整每一步 `T=1..100` 的 AUC@T 与 Cov@T，自动生成可供后续选择的宽表和长表。0步不定义 AUC。
- `table1_all14_full.csv`: 每方法所有T的均值和实际完成数；`table1_paper.csv / .tex / .pdf`: `Method | AUC@20 | AUC@50 | AUC@100 | Cov@20 | Final`。论文指标必须在检查所有曲线之后统一预注册，不得按 Ours 优势选择对比点。
- `n=14 scenes` scene-equal；同 scene 内5 starts。提供 paired per-scene bootstrap CI 或 paired per-scene difference 的置信区间（14 scenes，小样本，说明重抽样单位），禁止把 70 episodes 假装为70个独立场景。
- 验证：对 T0/MAG 已验证汇总，重算结果应在四舍五入3位后一致；否则 STOP 核查。
- 若基线无法复现，输出明确的 `BLOCKED` 并保留空白 `—`，不要插入论文非同协议数值或伪造。

## 4. Table 2: Held-out scene comparison

- 在 same episodes 基础上按**真实 scene manifest**计算 Train(6), Dev(5), Test(3), Dev+Test(8)（只在实际成员数吻合时采用）。正文优先 Test(3) 与 Dev+Test(8)；Dev单独可入补充。
- 对比 MAGICIAN Beam 与 V3-T0+D1；AUC@20, AUC@50, AUC@100, Final；同场景同start配对，若基线缺失则报告缺口。
- 重点检查**T0 的训练数据、验证/选模是否涉及 Dev/Test**。对 Test Only 的泛化结论须核实没有参与训练、调参和模型选择。另审计 MAGICIAN 的训练数据 provenance，避免错误声称它在 Macarons++ 完全零训练。
- 禁止让训练集的 All-14 平均替代 Test-only 的真实统计。

## 5. Table 3: K × planning, 核心因果消融

- 2×2 条件：`K in {1,4}`, `plan_depth in {1,2}`；全部使用**同一个冻结的 T0 checkpoint**、same 100-step environment、相同 D100/S10/R10、same 10-step Euler、相同 seeded action tie-break/fallback、固定candidate合法性、same planner coverage novelty `eps_cov=1m`, `cell_size=1m`, distance weight/union max, λ=0。
- depth=1 分数必须是 D1 中 `value_cov` 的 per-sample coverage 平均，**不是 H0 DistP1**。depth=2 在完全同一评分和候选条件下做两步联合覆盖；当没有合法第二步时显式fallback一步。
- `K=1` 优先用与K4的第0个 frozen噪声种子配对；`K=4` 用固定四个噪声。记录噪声组、随机调用、K1 的差异波动；如果有多次噪声seed复核预算，优先补 K1 的多个指定 noise sets，别偷偷挑最优seed。
- 复用当前 `K4/depth2` 70条真实结果；**只需补其他三个配置**（如果尚未存在）。在预算紧张时优先 `K4/depth1` 和 `K1/depth2`，最后 `K1/depth1`，所有中途结果必须标完成率，不得将部分场景结果误称 All-14。
- 分析主效应与交互作用：比较K1→K4在两种depth下差值、depth1→depth2在两种K下差值。别提前宣称K4优于K1。标准误/CI以scene为单位。

## 6. Fig 3: Coverage and cumulative AUC

- 使用 Table1 的同一组原始 101点曲线；(a) mean official coverage vs step 0..100 with 0..20 inset；(b) cumulative AUC@T vs T=1..100。避免示意/平滑造数，不对官方coverage取cummax。
- 6方法如果拥挤，主图 Ours、MAGICIAN、MACARONS、FisherRF，剩余方法以较细线或Supplementary; **不得因为结果弱而隐藏某方法**。建议透明度与图例一致。报告可用真实曲线的方法数。
- 每方法 bootstrap shaded CI 可选，按scene聚合；论文主图可不添加复杂统计阴影，补充图呈现。

## 7. Fig 4:真实多假设 depth visualization

- **额外提取原生模型中间结果**：同一 `scene, start, t, candidate pose` 固定 `P_t` 和 `C_c`，用 K=4 frozen noises, 保存四个 sample `b_hat, y_hat`, decode first-hit depth、UNKNOWN mask、已有表面、GT next-view depth（仅用于可视化或离线诊断，不参与规划）。
- 图片布局：row1 Partial TSDF visible / candidate rays / GT depth；row2 sample k=1,2,3,4 decoded depth；可增加每个样本 novel coverage overlay, per sample quantitative label；颜色尺完全一致、相机视角一致。**k=1..4是样本索引；不要把它们错误命名为K=1..4四个模型配置**。
- 在 train/dev/test 各挑代表例子，按事先客观criteria（有效 unknown rays, diversity, prediction/GT可读性），记录所有候选和选择理由以防 cherry-picking；至少一个成功案例和一个困难案例。如果没有可视化缓存，跑低开销的单状态模型前向而非整条100步。
- 严格区别 decoded depth、generated SDF：本方法并不生成四个完整一致的3D SDF世界。**两步不同视角用同k噪声只是Monte Carlo配对，不保证共享同一个真实世界。**

## 8. Fig 5: Radar scene-by-scene

- 用同一14场景所有5起点的 AUC@20 scene-mean；Ours、MAGICIAN Beam，第三条MACARONS仅在可审计数据齐备时加入。所有14轴统一0..1标尺，不做per-scene分别归一化；按冻结scene清单顺序。将Train/Dev/Test用标签颜色或背景分区，避免错误解释。
- 输出 `per_scene_auc20.csv`, radar .pdf/.png；同时输出 paired per-scene ΔAUC@20 排序条形图 `per_scene_delta.pdf` 供可读性选择；必要时把雷达放补充材料，比较清晰的差值图放正文。

## 9. 8小时资源调度（按真实时间窗口动态调整，非保证完成）

- **P0**：预检已有完整episode；生成Table1/2和Fig3/5（若T0+MAG缓存已经完整，这些几乎只需汇总计算与绘图）；图表每张都对应真实数据。
- **P0**：Table3 三组缺失的闭环配置，尽可能多个独立 Slurm jobs（按场景/起点分片，单卡单episode；并行上限由资源与内存决定，避免争抢节点）；每个job明确 time/memory/cpus/gpu，预留失败重跑窗口。
- **P0**：Fig4单状态可视化，尽量与Table3闭环并行但不占用会阻塞主作业的GPU资源。
- **P1**：补跑 Table1 缺失 baseline。先有缓存的算法、能无GPU执行的Random；SCONE/MACARONS/FisherRF如代码和权重缺失就只做发现与复现可行性，不要因下载/重训未知工程阻塞主实验。
- **P1**：抽取Test-only，并对结果校验与导出论文级PDF/TeX，完成全部实际能得到的图表。
- **P2**：多噪声seed、Ray Encoder ablation、runtime对照，仅在前三表和三图完成后再安排。
- 预计成本：70条 × 9–13min ≈ 10.5–15.2 node-episode-hour（串行）；三个新的70条配置≈31.5–45.5 episode-hour，加上六baseline更高。实际8h可完成的范围受GPU配额、Slurm排队与脚本开销限制。**先做可执行性计算，不能承诺8h内全部跑完**。
- 每个job写可恢复进度；正常checkpoint/episode落盘即刻聚合，避免直到所有任务完成才产出数据。

## 10. 自动质量关卡（必须 PASS 才能叫 CVPR-ready）

1. 主方法权重 SHA、T0的无xattn、D1 frozen config 校验。
2. Exact same scene/start evaluation pairing; 101点coverage、official replay bitwise identical；统计三位小数与历史核验值一致。
3. 每张表检查source manifest、n scene、n start、evaluation split、公式、CI方法；不混合paper数字。
4. Table3 四行每行相同完整评估场景和起点；如不同，标 `preliminary subset` 并另算交集上的配对表。
5. Fig4不同k来自同一状态与候选、同色尺、真实生成的深度、GT不可进入在线planner。
6. 雷达图14轴同一尺度、真实每scene数据；不隐藏失败场景。
7. pdf, png, tex, csv 所有产物由同一不可变中间records生成；脚本重新运行再生成应一致。生成 checksums。

## 11. 最终交付和自检文档

无论8小时内完成多少，必须产生：

- `experiments/cvpr_results/report/RESULTS.md`：逐个Table/Figure总结发现（没有数据就写缺失）、结果路径、完成度与核心对比。
- `.../report/RUN_LOG.md`：命令、Slurm job id、GPU/CPU/RAM、walltime、退出码、resume状态、失败原因。
- `.../report/VALIDATION.md`：QA每项PASS/FAIL + 可复现验证命令；是否跟历史T0/MAG汇总一致。
- `.../report/MISSING.md`：尚缺每个baseline或配置的episode数，下一步可以直接执行的任务，不允许泛泛写“待补充”。
- 最终 ChatGPT/Coding AI 回复要给出 **完整 AI 回复**：执行的改动、真实运行命令与job ID、已经完成的table/figure链接或路径、指标、未完成和原因。不可把提交成功当成数据已完成；不可在没有运行时声称已经跑了实验。

**立即开始：先完成0预检查，直接运行能执行的优先项，不用等再次确认。**
