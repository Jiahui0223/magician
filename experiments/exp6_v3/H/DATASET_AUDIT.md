# DATASET_AUDIT — H 的场景与起点（沿用 CVPR Table 1）

- **场景注册**：`experiments/cvpr_results/table1/SCENE_REGISTRY.yaml`，冻结的划分见 `dataset_manifest_v1.json`。

| 组 | 场景 | episode / 策略 |
|---|---|---|
| Train-6（训练场景，已见过） | bannerman、dunnottar、fushimi、liberty、pisa、redeemer | 30 |
| Dev-5 | barts、colosseum、eiffel、pantheon、sestino_museum | 25 |
| Test-3（历史上用过多次，不是盲测） | alhambra、bridge、neuschwanstein | 15 |
| **All-14** | 以上 14 个场景 | **70** |
| Other | sofia_church：冻结的加载代码不支持该场景，为 PENDING | 0（没有 All-15） |

- **起点**：每个场景 settings.json 的 `camera.start_positions[0..4]`，与 MAGICIAN LMDB 相同。
- **步数**：100 步。
- **模拟器**：冻结的 M 协议 EpisodeSim。
- **位姿表**：14 个场景都已生成（`work_dirs/exp6_v3/H/pose_tables`），共 39,243 行，与 5,600 条历史服务记录逐位一致，见 `pose_tables/verification.json`。
