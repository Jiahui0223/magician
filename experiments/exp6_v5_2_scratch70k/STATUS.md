# EXP6 V5.2-M Scratch70k —— 状态板（驱动脚本自动更新）

**最近更新：2026-11-02 09:21:58**　标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 跳过

## GPU / 作业

| 作业 | 节点 | 到期 | 用途 |
|---|---|---|---|
| 3443137 `vscode-dev`（交互分配，头节点） | gpu115 4 × L40S | 11-04 15:11 | 驱动、训练 rank 0–3、里程碑 G2、最终评估 |
| 3449325 `v52m-nodeB`（sbatch） | gpu112 4 × L40S | 11-04 03:00 | 训练 rank 4–7、最终评估 / G3（结束后释放） |

## 进度

| 编号 | 内容 | 状态 |
|---|---|---|
| P1 | Scratch70k 训练 | 35,000 / 70,000 step；状态 running；阶段 train |
| P2.1 | 里程碑 G2 20k | ✅ 2026-11-02 06:43:32 → 2026-11-02 09:21:57 |
| P2.2 | 里程碑 G2 40k | ⏳  →  |

## 事件（最近 15 条）

- 2026-11-02 03:39:40 driver start pid 19378 phase all on gpu115 (job 3443137)
- 2026-11-02 03:39:40 round r000_1793615980: 2-node 8-rank from step 0, deadline Wed Nov  4 02:50:56 2026
- 2026-11-02 06:43:32 milestone G2 20k started (read-only, head node, concurrent with training)
- 2026-11-02 09:21:57 milestone G2 20k done

