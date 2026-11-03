# Macarons++ 数据集审计（CVPR Table 1）

- **日期**：2026-11-03。
- **方式**：对已有的代码、配置、数据、LMDB 和论文只读。新文件只写在 `experiments/cvpr_results/table1/`。临时帧目录 `work_dirs/cvpr_results/table1/tmp` 用完已删除。
- **GPU**：只用了 GPU 3（`CUDA_VISIBLE_DEVICES=3`）。
- **标记**：
  - 【事实】= 本次从代码、配置、数据或运行结果直接核对；
  - 【推断】= 由事实推出、未直接验证的判断。
- **机器可读版本**：`SCENE_REGISTRY.yaml`（每个场景一条）。
- **路径缩写**：

| 缩写 | 路径 |
|---|---|
| MAN | `experiments/exp4_multiworld_belief/task0/configs/dataset_manifest_v1.json`（冻结的数据清单） |
| PROT | `experiments/exp4_multiworld_belief/task0/dataset_protocol.md` |
| CM | `MAGICIAN_clean_9521d69/configs/macarons/macarons_default_training_config.json` |
| CT | `MAGICIAN_clean_9521d69/configs/test/test_in_default_scenes_config.json` |
| MU-clean | `MAGICIAN_clean_9521d69/macarons/utility/macarons_utils.py` |
| MU-main | `MAGICIAN/macarons/utility/macarons_utils.py` |
| MU-up | `work_dirs/exp5_future_view/task0/magician_comparison/external/MAGICIAN/macarons/utility/macarons_utils.py`（官方 HEAD e697009） |
| EXP3B | `experiments/exp3b_macaronspp_transfer` |
| T1 | `experiments/exp5_future_view/task1` |
| T2 | `experiments/exp5_future_view/task2` |

---

## 0. 结论摘要

| 项 | 结果 |
|---|---|
| 官方场景数 | **15**（MAGICIAN 论文表 6/7，`MAGICIAN/Magician.pdf` 第 12 页） |
| 本地场景目录 | **15**（`MAGICIAN/data/Macarons++/`）。资产齐全，obj / mtl / settings 的 sha256 与冻结清单一致（`sha256_check.txt`，45 项全部 OK） |
| 冻结 M 协议下可运行 | **14**。`EpisodeSim(scene, 0)` → `observe(0)` → `legal()` 全部成功；**sofia_church 失败** |
| MAGICIAN LMDB | 14 个场景 × 5 个起点，每条 102 个位姿、101 个覆盖值；**没有 sofia_church** |
| 我们规划器的场景表 | 14 个场景齐全。sofia_church 缺 E-M2 窗口表条目（`KeyError`）；Z0 冻结值只有 Dev-3 三个，其余场景在运行时按同一规则算出（派生，非冻结） |

**划分**（【事实】，依据 MAN 和 PROT:30-35）：

| 集合 | 场景 |
|---|---|
| Train | Train-6：bannerman、dunnottar、fushimi、liberty、pisa、redeemer |
| Dev | Dev-5：barts、eiffel、pantheon、colosseum、sestino_museum。其中闭环 Dev-3 = barts、pantheon、colosseum |
| Test | Test-3：alhambra、neuschwanstein、bridge |
| Other | sofia_church（MAN 中的集合为 `reserved` = 不使用） |

**sofia_church 失败的确切原因**（【事实】）：
- 错误：`TypeError: tensor() missing 1 required positional arguments: "data"`。
- 位置：MU-clean:494 `load_scene` 多纹理分支中一行多余的 `torch.tensor()`（MU-main 中是 570 行）。
- 调用链：`m_t6_sim.py:51` → `load_scene`。
- 官方上游在 eec9e1d 中已把这一行注释掉（MU-up:481）。

---

## 1. 官方名称与本地目录的对应

- 【事实】论文表 6/7 的场景名和顺序，以及本地 `.obj` / `.mtl` 中的纹理文件名，见下表。
- 【推断】名称与目录的对应依据纹理名和 obj 名。这个对应是确定的，只有一处需要注意：
  - pisa 的纹理叫 `a-LeaningTowerOfPisa.jpg`，论文写作 "Pisa Cathedral"；
  - 两者是同一处地标建筑群，不影响对应。

| 论文名 | local_id | 类型 | obj | 纹理（map_Kd） |
|---|---|---|---|---|
| Dunnottar Castle | dunnottar | 室外 | dunnottar_rgb.obj | dunnottar.jpeg |
| Colosseum | colosseum | 室外 | colosseum_rgb.obj | a-Colosseum.jpg |
| Bannerman Castle | bannerman | 室外 | bannerman_rgb.obj | ny.jpeg（mtl 来自 bannerman_2.blend） |
| Pantheon | pantheon | 室外 | pantheon_rgb.obj | a-Pantheon.jpg |
| Christ the Redeemer | redeemer | 室外 | redeemer_rgb.obj | a-ChristTheRedeemer.jpg |
| Statue of Liberty | liberty | 室外 | liberty.obj | a-StatueOfLiberty.jpg |
| Pisa Cathedral | pisa | 室外 | pisa_rgb.obj | a-LeaningTowerOfPisa.jpg |
| Fushimi Castle | fushimi | 室外 | fushimi_rgb.obj | Momoyama.jpg |
| Alhambra Palace | alhambra | 室外 | alhambra_rgb.obj | a-Alhambra.jpg |
| Neuschwanstein Castle | neuschwanstein | 室外 | bavaria_rgb.obj | a-NeuschwansteinCastle.jpg |
| Eiffel Tower | eiffel | 室外 | eiffel_rgb.obj | a-EiffelTower.jpg |
| Manhattan Bridge | bridge | 室外 | bridge_rgb.obj | a-ManhattanBridge.jpg |
| St. Sofia Church | sofia_church | 室内 | church.obj | **4 张**：interior_146k_u{1,2}_v{1,2}.jpg |
| Barts | barts | 室内 | stair.obj | StBartsMuseumStairwell03.jpg |
| Sestino Museum | sestino_museum | 室内 | museum.obj | Museum_Interior_Sestino_u1_v1_diffuse.jpeg |

---

## 2. Train / Dev / Test / Other 划分与证据

### 2.1 我们项目的冻结划分（【事实】）

- **来源**：MAN，`"status": "FROZEN 2026-10-14 (user approval)"`，sha256 `e3cf35d7…`，本次 `sha256sum -c` 通过。
  - `sets`：train 在 MAN:8-15，dev 在 MAN:16-22，test 在 MAN:23-27，reserved 在 MAN:28-30。
  - 每个场景的 `"set"` 字段：alhambra :47、bannerman :64、barts :121、bridge :162、colosseum :179、dunnottar :276、eiffel :556、fushimi :597、liberty :654、neuschwanstein :696、pantheon :713、pisa :780、redeemer :867、sestino_museum :914、sofia_church :1020。
- **文字版**：PROT:32（Train-6）、:33（Dev-5）、:34（Test-3）、:35（Reserved）。
- **其他代码中的同一划分**：
  - Train-6：`T1/configs/data_v1.json:5`（`scenes_train6`）；`T1/protocol.md:29`。
  - Dev-3 / Test-3（闭环）：
    - `T2/configs/eval_outdoor_v1.json:9-13`（test：alhambra、bridge、neuschwanstein）、`:14-18`（dev：barts、pantheon、colosseum）；
    - `experiments/exp5_future_view/ablation_study/common/ab_common.py:28`（`SCENES_OUT`，读的是上面这份配置）；
    - `T1/configs/closed_loop_v1.json:4`（Dev-3）；
    - `exp6_v3/ablation_study/shared/ab_data.py:30`；`exp6_v4/tests/test_split_and_leakage.py:23`。
  - Dev-5（EXP3b gate）：`EXP3B/m_common.py:34`（`GATE`）、`EXP3B/t1_protocol_audit.py:31`、`EXP3B/r_protocol.py:51`、`experiments/exp5_future_view/task0/scripts/t05_outdoor.py:27`、`exp4_multiworld_belief/task2/scripts/c0x_windows_audit.py:27`。

### 2.2 eiffel、sestino_museum、sofia_church 的全部归属记录（【事实】）

| 场景 | 归属 | 位置 |
|---|---|---|
| eiffel | dev | MAN:18、MAN:556；`task0/configs/split_draft.json:6`；PROT:33 |
| eiffel | gate（= Dev-5） | `EXP3B/m_common.py:34`、`t1_protocol_audit.py:31`、`r_protocol.py:51`；`work_dirs/exp3b_macaronspp_transfer/task1_protocol/scenes.json`（group Unseen-6，R_role gate） |
| eiffel | **MACARONS 权重的 test** | CM:32-35；`experiments/exp5_future_view/task0/magician_comparison/configs/t09_official/macarons_mp3d_formal.json:34`（MACARONS 配置的副本） |
| eiffel | MAGICIAN 14 个测试场景之一 | CT:7 |
| sestino_museum | dev | MAN:21、MAN:914；`split_draft.json:6`；PROT:33 |
| sestino_museum | gate | `EXP3B/m_common.py:34` 等 |
| sestino_museum | MAGICIAN 测试场景 | CT:18 |
| sestino_museum | 不在 MACARONS 数据中 | CM:17-35 中不存在 |
| sofia_church | **reserved（不使用）** | MAN:29、MAN:1020；`split_draft.json:8`；PROT:35 |
| sofia_church | unused | `scenes.json`；`EXP3B/protocol_card.md` §3 |
| sofia_church | 不在 MAGICIAN 测试列表中 | CT:4-19 |
| sofia_church | 不在 MACARONS 数据中 | CM:17-35 中不存在 |

- **exp7**：`experiments/exp7_final_eval/` 是空目录，没有任何配置。
- **结论**：我们自己的配置之间**没有冲突**。eiffel 唯一不同的标签是 MACARONS 自己的权重划分（test），属于另一个范围，见 §8 冲突 1。

### 2.3 S-c 的含义（【事实】）

- **定义**：S-c 是 **Train-6 场景内部的起点级留出集**，不是场景级划分。
- **规则**：每个 Train-6 场景留出 3 个起点：从 4 个 "A" 起点中取 1 个，从 11 个 "B" 起点中取 2 个。种子为 `sha256("exp5-V2|holdout|<scene>")`。
  - 依据：`T1/protocol.md:152-154`、`T1/scripts/t11_split.py:36-43`；结果写在 `T1/frozen_values.json` 的 `split`（第 3 行）。
- **规模**：共 44 / 180 条轨迹（bannerman 8、dunnottar 6、fushimi 8、liberty 6、pisa 6、redeemer 8）。
- **用途**：所有历史模型都在 `train6_minus_Sc` 上训练（`T1/scripts/v2_data.py:176-191`），S-c 只用于监控和校准，例如 `T1/protocol.md:362-376` 的 S-c 校准状态。
- **E2 D100 分片**：`work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100` 的 `heldout` 标记，与冻结的 S-c 逐条一致（6 个场景本次全部核对）。

### 2.4 Train-6 的官方起点进入过训练轨迹（【事实】）

- **核对方法**：本次把 `data/macaronspp_belief/v1/trajectories/<scene>/*/records/*_t0.npz` 的 `pose_idx`，与 settings 的 `start_positions` 逐一对照。
- **结果**：6 个 Train-6 场景都一样：
  - 官方起点 0、1、2、3 都有训练轨迹（分别为 6、5、4、4 条）；
  - 官方起点 4 从未用作起点。
- **S-c 留出的官方起点**：bannerman 0、dunnottar 3、fushimi 0、liberty 2、pisa 3、redeemer 0。
- **设计依据**：`experiments/exp4_multiworld_belief/task0/trajectory_generation.md:36,76,78`。

### 2.5 MAGICIAN 基线的训练 / 验证重叠（【事实】）

- **MAGICIAN 没有自己的权重**，用的是 MACARONS 的 `trained_macarons.pth`（`EXP3B/magician_protocol_facts.md:91`）。
- **MACARONS 的划分**（CM:18-35）：
  - train：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum；
  - val：alhambra、neuschwanstein；
  - test：bridge、eiffel。
- **结果**：我们的 Dev-3 中 pantheon、colosseum 是 MAGICIAN 的训练场景；我们的 Test-3 中 alhambra、neuschwanstein 被 MACARONS 用来选 checkpoint。PROT:34 要求在报告中披露这一点。

---

## 3. 历史模型用了哪些场景训练（【事实】）

**室外部分全部是 `train6_minus_Sc`**（Train-6 去掉 S-c）。Dev-5、Test-3 和 sofia_church 都没有用来训练任何模型。

| 模型 | 室外训练数据 | 室内 | 依据 |
|---|---|---|---|
| V2 Original | train6_minus_Sc（稀疏分片） | 无 | `T1/configs/train_main.json:5-6` → `t12_train.py:170` → `v2_data.py:176-191` |
| V2-Mix（B0 / B2 权重） | train6_minus_Sc | MP3D train | `T2/configs/train_mix_v1.json:8-9`；`t22_train_mix.py:68` |
| E2-D100 | train6_minus_Sc（D100 分片） | MP3D train | `e2_pilot/configs/e2_d100_v1.json:8-9`；`e2_train.py:4-7`（运行冻结的 t22） |
| V3-T0 / T1 | out_views_d100 的 train（= Train-6 去 S-c） | MP3D train | `seen2scene/v3/trainer.py:117-118`；`seen2scene/v3/data.py:43-58` |
| V3.1 | 同上 | 0 | `exp6_v3.1/configs/outdoor_train.json:6-9`；`trainer_outdoor.py:54` |
| V4 | train6_minus_Sc（D100） | 0 | `exp6_v4/configs/outdoor_train.json:7-11`；`train_v4.py:85` |
| V4C | train6_minus_Sc | 0 | `exp6_v5/configs/train_v4_continue.json:2-5` |
| V5 | train6_minus_Sc | 0 | `exp6_v5/configs/train_v5_outdoor.json:2-5`；`train_v5.py:198` |
| V5.2 pilot / V5.2-M | train6_minus_Sc | 0 | `exp6_v5_2/configs/train_v52.json:5-8`；`exp6_v5_2_scratch70k/configs/train_v52m.json:5-8` |

---

## 4. 资产（【事实】）

- **15 个目录都有**：1 个 `.obj`、1 个 `.mtl`、mtl 引用的全部纹理、`settings.json`、`occupied_pose.pt`。
- **sha256**：obj / mtl / settings 与 `exp4…/task0/configs/SHA256SUMS` 一致（`sha256_check.txt`）。
- **sofia_church**：
  - 是唯一一个 `map_Kd` 多于 1 张（4 张）的场景；
  - obj 只有 12 MB / 108,572 个面，其余场景是 62–236 MB；
  - 没有 `test_memory_*` 目录。
- **其他 14 个场景**：有 MAGICIAN 测试时写入的 `test_memory_0/1` 派生目录；Train-6 和 pantheon、colosseum 还有 `macarons_memory`。这些是派生目录，冻结清单标为 "never read"。
- **`occupied_pose.pt`**：EpisodeSim 会加载（`m_t6_sim.py:57`），但 MAGICIAN 测试时不使用（`EXP3B/magician_protocol_facts.md:62`）。
- **缺失**：除 sofia_church 加载器的问题外，没有任何缺失的资产。

---

## 5. 起点（【事实】）

- **来源**：唯一的起点定义是 `settings.json` 的 `camera.start_positions`（5 维网格下标，每个场景 5 个）。
  - MAGICIAN 测试循环在 `MAGICIAN_clean_9521d69/macarons/testers/magician_planning.py:1457-1462` 依次跑全部 5 个；
  - CT 中没有起点列表。
- **EpisodeSim**：`EpisodeSim(scene, k)` 把 `settings.camera.start_positions[k]` 传给 `setup_test_camera`（`m_t6_sim.py:69-71`）。
- **本次核对，14 个场景全部通过**：
  1. 构造 `EpisodeSim(scene, 0)` 后，当前位姿键等于 `start_positions[0]`；
  2. 5 个起点经模拟器相机的 `pose_space` 得到的 (X, 俯仰, 方位)，与 `magician_lmdb` 中 `X_cam_history[0]` / `V_cam_history[0]` **完全相同**（最大 |ΔX| = 0）。
  - 由此可知，我们的 EpisodeSim 起点 k 就是官方起点 k。
- **sofia_church**：
  - 冻结路径无法构造；
  - 在诊断补丁下（§6.2），起点 0 的键也等于 `start_positions[0]`；
  - 它没有 LMDB，所以无法与 MAGICIAN 的实际轨迹对照。

---

## 6. 加载测试（GPU 3，magician 环境）

### 6.1 冻结 M 协议

- **命令**：`scripts/load_test.py`，每个场景一个新进程，导入方式与 `T2/scripts/t25_out_cov.py` 相同。
- **结果文件**：`load_test_results.json`；日志：`load_test.log`。
- **所有成功场景的共同值**：eps = 1.0；peak 显存 1.15–1.34 GB；`observe(0)` < 0.07 s（第 0 帧在构造时已渲染）。

| 场景 | 成功 | 构造 s | GT 点 | vis | t=0 合法动作 | cov(t=0) |
|---|---|---|---|---|---|---|
| alhambra | ✓ | 32.5 | 45,355 | 0.8889 | 8 | 0.0086 |
| bannerman | ✓ | 23.8 | 43,586 | 0.9996 | 5（另 5 个碰撞） | 0.0020 |
| barts | ✓ | 16.8 | 47,885 | 1.0 | 10 | 0.0790 |
| bridge | ✓ | 29.9 | 40,252 | 0.9939 | 9 | 0.0 |
| colosseum | ✓ | 34.0 | 49,067 | 0.8002 | 9 | 0.0025 |
| dunnottar | ✓ | 18.9 | 55,022 | 1.0 | 9 | 0.0180 |
| eiffel | ✓ | 30.0 | 22,443 | 0.9679 | 9 | 0.0 |
| fushimi | ✓ | 21.6 | 25,081 | 0.9999 | 10 | 0.0778 |
| liberty | ✓ | 9.9 | 23,568 | 0.8632 | 9 | 0.0874 |
| neuschwanstein | ✓ | 33.1 | 68,569 | 0.9944 | 10 | 0.0055 |
| pantheon | ✓ | 32.1 | 54,170 | 0.7269 | 9 | 0.0425 |
| pisa | ✓ | 30.2 | 47,968 | 0.8036 | 10 | 0.1088 |
| redeemer | ✓ | 31.5 | 23,942 | 0.9566 | 8 | 0.0179 |
| sestino_museum | ✓ | 17.2 | 55,514 | 1.0 | 7 | 0.0029 |
| **sofia_church** | **✗** | – | – | – | – | – |

**sofia_church 的错误**（完整 traceback 在 `load_test_results.json` 中）：

```
m_t6_sim.py:51 __init__: self.mesh = RP.load_scene(objs[0], ...)
MAGICIAN_clean_9521d69/macarons/utility/macarons_utils.py:494 load_scene: torch.tensor()
TypeError: tensor() missing 1 required positional arguments: "data"
```

- `load_scene` 先用 `count_map_kd` 数纹理（MU-clean:447-463）。
- 纹理 > 1 时，走 `load_obj(create_texture_atlas=True)` 分支（MU-clean:482-500）。
- 这个分支在 `return` 之前有一行不带参数的 `torch.tensor()`（MU-clean:494；MU-main:570）。
- 只有 sofia_church 有多于 1 张纹理，所以只有它触发。

### 6.2 诊断：去掉这一行能否运行（不是冻结协议）

- **方法**：`scripts/sofia_diag.py` 只在本进程内，把 EpisodeSim 调用的 `load_scene` 换成 MAGICIAN 原函数去掉这一行的副本。没有改任何文件。
- **结果**（`sofia_diag_result.json`）：**成功**。
  - 构造 12.3 s；GT 点 55,639；eps 1.0；vis 1.0；纹理为 TexturesAtlas；
  - cov(t=0) 0.0138；合法动作 9（另 1 个碰撞）；峰值显存 3.8 GB。
- **上游情况**：官方上游在 eec9e1d（"Remove unused occupied_pose data path"）把这一行注释掉了（MU-up:481）。但官方 CT 仍然只列 14 个场景。

---

## 7. 规划器的每场景表与缺口

### 7.1 每个场景需要的表（【事实】）

- **V2 系列规划器子进程**：
  - B2、E2、V3、V4、V5 都经 `t24_v2_out_planner.py` 进入 `T1/scripts/v2_planner.py`。
  - 依据：`task3_full_depth_p1/scripts/t32_out_planner.py:1-10`、`exp6_v4/scripts/v4_out_planner.py:1-13`。
  - 每个场景只需要下面两张表：
    1. **几何**：`t5_diagnostic.scene_geometry(scene)`（`v2_planner.py:112`）→ `frame_of`（`EXP3B/t5_diagnostic.py:110-113`）。它读的 `work_dirs/exp3b_macaronspp_transfer/task1_protocol/em2_windows.json` **只有 14 个场景**，没有 sofia_church。
    2. **Z0**：`v2_planner.py:114` 读 `LC.Z0[s]`。
       - `v2_common.py:32` 冻结的 Z0 只有 barts 15.1142、pantheon 38.2155、colosseum 36.6305；
       - Task 2 的室外子进程在启动时用规则 `abl_common.z0_of(scene)`（场景框三边 ×10 的平均 / 2，`abl_common.py:75-82`）补上（`t24_v2_out_planner.py:46`）；
       - 这个规则在 Dev-3 上与冻结值逐位相等（断言在 `abl_common.py:80-81`）。
- **不需要每场景表的部分**：
  - 合法动作列表由模拟器实时计算（`m_magician_replay.py:84-102`）；
  - 起点来自 settings；
  - MAGICIAN 规划器子进程（`T2/scripts/t24_mag_out_planner.py:38-63`）只读 `settings.json`。
- **EXP4 的位姿表 `_pose_tables`**：只有 barts、colosseum、pantheon，只被 EXP4 abl_1 规划器使用，V2 系列不用。

### 7.2 每个场景的值（`planner_tables.json`；场景几何用 seen2scene 环境实际调用 `scene_geometry` 核对）

| 场景 | Z0 | Z0 状态 | k_M（E-M2） | 网格 dims | scene_geometry |
|---|---|---|---|---|---|
| barts | 15.1142 | 冻结 | 11.561 | 256×256×256 | ✓ |
| pantheon | 38.2155 | 冻结 | 17.013 | 401×568×256 | ✓ |
| colosseum | 36.6305 | 冻结 | 11.919 | 792×628×256 | ✓ |
| alhambra | 39.5103 | 派生（规则；Test-3 闭环已按此值运行） | 16.464 | 505×548×256 | ✓ |
| bridge | 49.8323 | 派生（同上） | 12.470 | 1424×500×256 | ✓ |
| neuschwanstein | 47.4465 | 派生（同上） | 29.641 | 430×256×256 | ✓ |
| eiffel | 30.7886 | 派生，非冻结 | 33.664 | 256×256×256 | ✓ |
| sestino_museum | 15.4910 | 派生，非冻结 | 4.426 | 1286×367×256 | ✓ |
| bannerman | 44.1586 | 派生，非冻结 | 21.676 | 393×462×256 | ✓ |
| dunnottar | 52.3693 | 派生，非冻结 | 9.670 | 1110×1588×256 | ✓ |
| fushimi | 29.7657 | 派生，非冻结 | 15.896 | 345×421×256 | ✓ |
| liberty | 28.3087 | 派生，非冻结 | 28.714 | 256×256×256 | ✓ |
| pisa | 40.4934 | 派生，非冻结 | 13.995 | 959×363×256 | ✓ |
| redeemer | 25.8680 | 派生，非冻结 | 16.824 | 256×386×256 | ✓ |
| **sofia_church** | 15.9840 | 派生，非冻结 | **6.430（派生，非冻结）** | **656×443×256（派生）**，6 个官方窗口 | **✗** `KeyError: 'sofia_church'` |

- sofia_church 的 k_M 按 em2 规则计算：k_M = y 方向高度 ×10 / 2.816（规则文本在 `em2_windows.json` 的 `rule` 字段）。
- 同一规则在 barts、eiffel、sestino_museum 上与冻结的 k_M 逐位相等。

### 7.3 要在某个场景上跑我们的规划器还缺什么

- **13 个可运行场景**（Train-6、eiffel、sestino_museum、Test-3）：不缺任何表。
  - Z0 由子进程运行时按规则算出，几何表完整；
  - 如果希望 Z0 也有冻结值，只需把上表的规则值写进一张新的冻结表（【推断】：不会改变行为，因为数值相同）。
- **Dev-3**：冻结表完整。
- **sofia_church 需要以下 4 项**，任何一项都改变了冻结状态，需要审批：
  1. **模拟器**：冻结的 `MAGICIAN_clean_9521d69` 中的 `load_scene` 会报错（MU-clean:494）。必须选一种做法：
     - 在 EXP3b 一侧做一个"去掉这行"的适配（§6.2 已验证可行）；
     - 或者对 sofia_church 改用官方上游 e697009 的 `load_scene`。
     - 两种都属于对冻结 M 协议的偏差。
  2. **几何**：需要给 sofia_church 补 E-M2 条目。规则派生值为 k_M 6.4304、dims [656, 443, 256]、6 个窗口。em2 是冻结文件，不能改，需要新增一张表，或者在运行时派生。
  3. **Z0**：规则派生值 15.9840（`t24_v2_out_planner` 已自动派生）。
  4. **MAGICIAN 基线**：sofia_church 没有 `magician_lmdb` 轨迹，需要在同一个模拟器里重跑 MAG-B / MAG-G。MAGICIAN 规划器子进程只读 settings，【推断】它本身不受加载 bug 影响。

---

## 8. 冲突与需要披露的事项（【事实】，附出处）

1. **eiffel 的标签**：我们的冻结划分是 dev（MAN:556），MACARONS 权重划分是 test（CM:32-35）。
   - 两者范围不同：前者针对我们的模型，后者针对 MACARONS/MAGICIAN 的权重。
   - 我们的配置内部没有冲突。这里只报告，不做选择。
2. **pantheon、colosseum**：在我们的 Dev-3 中，但它们是 MAGICIAN 所用权重的训练场景（CM:18-27）。
   - Train-6 对我们的模型和 MAGICIAN 也都是"见过的"（PROT:112）。
3. **Test-3 已经不是盲测**：
   - PROT:34 和修正案 A-EXP4-1（PROT:94-113；`EXP3B/preregistration.md:2237`）规定 Test-3 "只在全部冻结后跑一次"。
   - 但 `work_dirs` 中已有至少 30 组 Test-3 的完整 100 步闭环（`closed_loop_inventory.json`）：V2O、V2M、MAG-B、MAG-G、B1、B2、E2D、E2S、V3 T0/T1、V3.1、V4、V4C、V5、V5.2 λ0/λ0.1、density A/B、temporal refresh 等。
   - `experiments/MODEL_VERSIONS.md:8` 也写明 Dev-3 / Test-3 "不是盲测"。
4. **15 个 vs 14 个**：
   - 论文表 2 的 0.721 / 0.919 是 15 个场景的平均，包括 St. Sofia（`Magician.pdf` 第 12 页）。
   - 代码（CT:4-19）和 LMDB 只有 14 个。
   - 按论文表 6/7 去掉 St. Sofia 后，14 个场景的平均为 MAGICIAN 0.7214 / 0.9211、MACARONS 0.6494 / 0.8203、Random Walk 0.2383 / 0.3239。
   - 本地 LMDB 复现的 14 个场景平均为 0.7111 / 0.9060（AUC = trapz(cov[0..100]) / 100）。
5. **论文内部不一致**：表 7 中 FisherRF 的平均写 0.786，但 15 个逐场景值的平均是 0.7721。其余各列的平均都与逐场景值一致（差异 ≤ 0.0006，属于四舍五入）。
6. **EXP3b 预注册的分组已经不适用**：
   - 原分组为 All-14、Unseen-6、held-out-9（`EXP3B/preregistration.md:112,116`）；
   - A-EXP4-1 规定这些分组对 EXP4 起的结果不再适用，只报告 Dev-5 和 Test-3（PROT:109-110）。
   - 一张 All-15 / All-14 的 Table 1 会把 Train-6（训练场景）也纳入，这不在现有报告规则中。
7. **行号**：bug 在 MU-main 中是 570 行，在 M 协议实际使用的 MU-clean 中是 494 行。两处说的是同一个 bug。

---

## 9. 历史闭环（【事实】，`closed_loop_inventory.json`）

- **Dev-3 + Test-3**（barts、pantheon、colosseum、alhambra、bridge、neuschwanstein）：
  - 每个场景都有 30 组以上 "5 个起点 × 100 步" 的完整闭环，主要来源：
    - `work_dirs/exp5_future_view/task2/eval/outdoor/closed_loop/{V2O,V2M,MAG-B,MAG-G}`；
    - `task3_full_depth_p1/closed_loop/outdoor/{B1,B2,E2D,E2S}/V2M`；
    - `exp_density_ab/g3`、`exp_temporal_refresh/g3`；
    - `exp6_v3`（T0、T1、B2_S1c、T1_S1c、C1、C2）、`exp6_v4`（V31、V4、A1）、`exp6_v5`（V4C、V5）、`exp6_v5_2`（V52L0、V52L01）。
  - Dev-3 另有 EXP4 / EXP5 Task 1 的 Dev-3 闭环。
- **eiffel、sestino_museum**：只有 EXP3b T6 的 10 步基线闭环（`work_dirs/exp3b_macaronspp_transfer/task6_baseline_v2/run`，每个场景 90 条），以及 T7a 的 eiffel 冒烟测试。**没有任何 V2 系列或 100 步的闭环。**
- **Train-6 和 sofia_church**：没有任何闭环。
- **exp6_v5_2_scratch70k**：只有 alhambra、bridge 的 s0 和 s1。

---

## 10. MAGICIAN 官方逐场景数字在本地的位置（【事实】）

- **唯一来源**：`/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN/Magician.pdf` 第 12 页。
  - 表 6：15 个场景的 AUC，列为 Rand. Walk / SCONE / MACARONS / FisherRF / MAGICIAN；
  - 表 7：15 个场景的最终覆盖。
- **用法**：`pdftotext -layout -f 12 -l 12` 可以直接提取。本次已逐项转录进 `SCENE_REGISTRY.yaml`（每个场景的 `magician_paper` 字段），并用列平均核对过。
- **其他地方没有逐场景数字**：MAGICIAN 仓库、它的 `jiahui/` 笔记、官方 clone，以及我们 `experiments/` 下的文档中，都没有找到这两张表的文本版，只有平均数：
  - 0.721 / 0.919：`EXP3B/magician_protocol_facts.md:105-116`、`MAGICIAN/jiahui/stage_1/exp1/README.md:17`；
  - 去掉 St. Sofia 的 14 个场景平均：`magician_protocol_facts.md:116`。
- **本地复现的逐场景数字**：`magician_lmdb`（汇总在 `lmdb_summary.json` 和注册表的 `magician_lmdb` 字段）。

---

## 11. 推断（与上面的事实分开）

1. **可直接进入 Table 1 的场景**：冻结 M 协议下 14 个场景可以直接跑。All-15 必须先解决 sofia_church 的加载问题（§7.3 第 1、2 项），并且 sofia_church 上的 MAGICIAN 基线只能自己重跑。论文中 St. Sofia 的数字由作者的代码得到，我们无法复核。
2. **按划分报告**：如果 Table 1 按划分报告，干净的未见场景只有 Test-3，且它已多次评估、并非盲测。eiffel 和 sestino_museum 是我们从未闭环评估过的 Dev 场景：
   - 对我们的模型是未训练过的场景；
   - 对 MAGICIAN 也是未训练过的场景（eiffel 是 MACARONS 的 test，sestino_museum 不在 MACARONS 数据中）。
3. **Train-6 不能作为未见场景**：它们对我们所有模型都是训练场景，而且官方起点 0–3 的轨迹在训练集中（S-c 只留出其中一个），在 Table 1 中应单独标注。

---

## 12. 产物与复现（本次新建）

| 文件 | 内容 | 环境 |
|---|---|---|
| `SCENE_REGISTRY.yaml` | 15 个场景的注册表 | `scripts/build_registry.py`，seen2scene |
| `lmdb_summary.json` | `magician_lmdb` 70 条记录的形状、起点、AUC / final | `scripts/audit_lmdb.py`，magician |
| `load_test_results.json`、`load_test.log` | 冻结 EpisodeSim 加载测试 | `scripts/load_test.py all <tmp> <out>`，magician，GPU 3 |
| `sofia_diag_result.json` | sofia_church 去掉多余行后的诊断 | `scripts/sofia_diag.py`，magician，GPU 3 |
| `planner_tables.json` | Z0、k_M、`scene_geometry` | `scripts/planner_tables.py`。seen2scene 环境需要 `LD_PRELOAD=<env>/lib/libtbb.so`，见 `m_t6_common.seen2scene_env` |
| `closed_loop_inventory.json` | 历史闭环清单 | `scripts/audit_closed_loops.py` |
| `sha256_check.txt` | 资产 sha256 与冻结清单的核对 | `sha256sum -c` |

**magician 环境命令前缀**：`env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python`。

`scripts/` 中的其他文件（`aggregate.py`、`audit_model_episodes.py`、`plot.py`、`validate.py`），以及 `audit/`、`figures/`、`reports/`、`results/` 目录，**不是本次审计创建的**。
