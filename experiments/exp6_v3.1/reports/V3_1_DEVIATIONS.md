# EXP6 V3.1 — 偏差记录（与 V4 共用一份：正本 `../../exp6_v4/reports/V4_DEVIATIONS.md`）

| 编号 | 时间 | 内容 | 影响 / 处理 |
|---|---|---|---|
| D-0 | 2026-10-31 20:35 | G0 吞吐冒烟（profile / 1k short）按训练器的冒烟规则只读前 6 条训练轨迹（bannerman 等小场景），不含 dunnottar 等大网格 → 冒烟 s/步 低于正式运行；正式吞吐以 `runs/main/log.jsonl` 前 500 步实测为准 | 报告两者，不以冒烟值作 ETA |
