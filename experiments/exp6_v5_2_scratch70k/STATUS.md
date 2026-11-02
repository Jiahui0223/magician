# EXP6 V5.2-M Scratch70k —— 状态板（驱动脚本自动更新）

**最近更新：2026-11-02 15:27:03**　标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 跳过

## GPU / 作业

| 作业 | 节点 | 到期 | 用途 |
|---|---|---|---|
| 3443137 `vscode-dev`（交互分配，头节点） | gpu115 4 × L40S | 11-04 15:11 | 驱动、训练 rank 0–3、里程碑 G2、最终评估 |
| 3449325 `v52m-nodeB`（sbatch） | gpu112 4 × L40S | 11-04 03:00 | 训练 rank 4–7、最终评估 / G3（结束后释放） |

## 进度

| 编号 | 内容 | 状态 |
|---|---|---|
| P1 | Scratch70k 训练 | 70,000 / 70,000 step；状态 completed；阶段 done |
| P2.1 | 里程碑 G2 20k | ✅ 2026-11-02 06:43:32 → 2026-11-02 09:21:57 |
| P2.2 | 里程碑 G2 40k | ✅ 2026-11-02 10:05:07 → 2026-11-02 12:43:14 |
| P3 | 最终 G1/G2（last70k + best） | best step 50000；主权重 final_best |
| P4 | 门槛 / 决策 | FAIL → G3 smoke only (G2 gate not met) (smoke invalid) |
| P5.smoke | G3 smoke | ✅ 2026-11-02 15:07:47 → 2026-11-02 15:26:48  |
| P5.full | G3 full | ⬜  →  G3 smoke only (G2 gate not met) (smoke invalid) |

## 事件（最近 15 条）

- 2026-11-02 03:39:40 driver start pid 19378 phase all on gpu115 (job 3443137)
- 2026-11-02 03:39:40 round r000_1793615980: 2-node 8-rank from step 0, deadline Wed Nov  4 02:50:56 2026
- 2026-11-02 06:43:32 milestone G2 20k started (read-only, head node, concurrent with training)
- 2026-11-02 09:21:57 milestone G2 20k done
- 2026-11-02 10:05:07 milestone G2 40k started (read-only, head node, concurrent with training)
- 2026-11-02 12:43:14 milestone G2 40k done
- 2026-11-02 14:55:43 round r000_1793615980 exit 0, step 0 -> 70000
- 2026-11-02 14:55:45 final G2: last70k + best_train_holdout (step 50000)
- 2026-11-02 14:55:45 final_g2: node B gpu112 task final_g2_1793656545
- 2026-11-02 15:07:47 gate: FAIL -> G3 smoke only
- 2026-11-02 15:07:47 G3 smoke started (2 workers / GPU, head + node B)
- 2026-11-02 15:07:47 g3_smoke: node B gpu112 task g3_smoke_1793657267

