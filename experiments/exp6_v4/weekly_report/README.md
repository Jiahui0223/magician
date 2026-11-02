# weekly_report —— EXP6（V2 / V3 / V3.1 / V4）组会汇报材料

- `WEEKLY_REPORT.md`：正文（中文），含全部图表引用与结论。
- `figures/`：fig1–fig11（PNG，英文标注，可直接放进幻灯片）；**fig10 室内 / fig11 室外 = 各模型的平均 AUC（左）与平均 Final coverage（右）柱状图**。
- `tables/`：T3–T6（Markdown + CSV）。
- `scripts/make_weekly_report.py`：从实测结果文件重新生成图表（`python scripts/make_weekly_report.py`）。
- `figures/viz_*.png`：一次 NBV 决策的可视化（当前视角、候选相机、V4 的 K = 4 预测深度 vs GT、预测新表面 vs 真实新表面、预测分数 vs 真实增益）+ 全部状态汇总；由 `scripts/viz_extract.py`（GPU）和 `scripts/viz_candidates_plot.py` 生成，说明见 `WEEKLY_REPORT.md` 5.1 节。
