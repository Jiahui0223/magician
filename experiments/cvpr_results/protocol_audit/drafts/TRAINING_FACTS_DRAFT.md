# TRAINING_FACTS_DRAFT —— Table 1 各方法训练协议的事实核对（草稿）

- **日期**：2026-11-04。**性质**：只收集事实，不写结论报告。没有修改任何已有文件，没有训练，没有用 GPU（只在 CPU 上读了 3 个官方 `.pth` 的元数据与张量）。
- **标记**：【已核实】= 本次直接读到原文 / 代码 / 文件；【推断】= 由代码或数据推出，未运行确认；【未知】= 证据缺失，注明缺什么。
- **路径缩写**（R = `/bsuscratch/jiahuizhang/projects/occupancy/seen2scene`，下文相对路径都相对 R）

| 缩写 | 指向 |
|---|---|
| OC | 官方克隆 `work_dirs/exp5_future_view/task0/magician_comparison/external/MAGICIAN`（HEAD e697009） |
| FK | 用户 fork `/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN_clean_9521d69`（冻结 M 协议使用） |
| MW | 工作副本 `/bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN`（HEAD 9521d69） |
| CM | `configs/macarons/macarons_default_training_config.json`（OC 与 FK 两份逐字节相同，本次 `diff` 为空） |
| CT | `configs/test/test_in_default_scenes_config.json`（OC 与 FK 只差 `numGPU`） |
| MAGP | MAGICIAN 论文 arXiv 2603.22650（`MAGICIAN/Magician.pdf`，PDF 页码） |
| MACP | MACARONS 论文 https://arxiv.org/pdf/2303.03315v2（本次下载，PDF 页码） |
| SCP | SCONE 论文 https://arxiv.org/pdf/2208.10449（v2，本次下载，PDF 页码） |
| FRP | FisherRF 论文 https://arxiv.org/pdf/2311.17874（本次下载，PDF 页码） |
| MACGH | https://github.com/Anttwo/MACARONS（main）；原始文件经 `raw.githubusercontent.com/Anttwo/MACARONS/main/<path>` 用 curl 取得 |
| W00-2 | `work_dirs/weights/W00_external/W00-2_magician_official_e697009/MANIFEST.json` |
| FV | `experiments/exp5_future_view/task1/frozen_values.json`（S-c 冻结划分） |
| BEL | `data/macaronspp_belief/v1`（Train-6 的 180 条训练轨迹） |
| REG | `experiments/cvpr_results/table1/MODEL_REGISTRY.yaml` |

---

## 0. 要点

1. MAGICIAN 没有自己的权重：运行时加载 MACARONS 发布的 `trained_macarons.pth`（sha `34c5247f…`，epoch 42）。MAGICIAN 测试代码全程 `no_grad` / `eval`，没有发现 MAGICIAN 在 Macarons++ 上做任何训练。
2. 这份权重由 MACARONS 在 8 个场景上自监督在线训练：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum。这 8 个都在 Table 1 的 14 个评估场景中。训练起点是在整个位姿网格上随机抽的合法位姿，**不是** settings 的 5 个官方起点。训练轨迹没有发布，所以与评估起点是否重叠无法确定。
3. MACARONS 主结果对应 "Pretrained"（ShapeNet 预训练）配置；发布配置 `pretrained: true`。ShapeNet 版本：代码与 SCONE 论文均为 ShapeNetCore **v1**，SCONE 用 8 个训练类别。
4. MAGICIAN 论文表 2 中的 SCONE、FisherRF、Random Walk 三列具体用了哪份权重和哪段代码，**未知**。上游初始提交 e4e08fe 中名为 `fish_planning.py` 的测试器不含任何 Fisher 信息代码，打分用的是 MACARONS 的 SCONE 模块（`trained_macarons.pth`）。
5. 我们的全部模型都没有 ShapeNet 或其他外部预训练。室外数据都只来自 Train-6（去掉 S-c）。混合（MP3D + Macarons++）训练的有：B0/B1/B2、E2D、E2S、V3（T1）、V3-Y*；F1-GQ 只是通过 B0 初始化间接用到 MP3D。纯室外的有：V2O、V3.1、V4、V4C、V5。
6. 起点重叠：每个 Train-6 场景有 3 个官方起点是训练轨迹的起点。原因是 S-c 只从官方起点 0–3 中留出 1 个，起点 4 从未出现在任何轨迹里。Dev-5、Test-3、sofia_church 没有任何训练轨迹。
7. 评估场景参与过选择（不是训练）：
   - V3-Y50K 按 Dev-5 闭环 AUC@100 选出；
   - F1-GQ@5k 按 Dev-3 闭环 AUC@20 选出；
   - V3-D1 规划器按 Dev-5 选出；
   - Dev-3 一直是 V2 系列的开发集，Test-3 已多次评估，不是盲测。
8. 训练数据的轨迹中有 9 条（train 部分）由 MAGICIAN beam 10×10 + 官方 `trained_macarons.pth` 生成，另有 CG-ε 策略用 GT 一步增益选动作。这两类只决定相机走到哪里，标签仍来自 GT 网格。
9. 与已有审计不一致的 3 处（§D）：
   - S-c 条数应为 42 / 180，不是 44；
   - "官方 MACARONS 用完美深度训练"没有证据，证据指向 RGB 自监督深度；
   - "alhambra / neuschwanstein 被 MACARONS 用来选 checkpoint"没有证据。

---

## A. 基线

### A1. MAGICIAN

| 项 | 事实 | 标记 | 证据 |
|---|---|---|---|
| 占据模型来源 | 先在 ShapeNet 预训练，再在场景上微调；结构照搬 MACARONS | 【已核实】 | MAGP p4："The occupancy model is first pre-trained on ShapeNet [9] and then fine-tuned on 3D scenes [19]"；p11："We adopt this model architecture from [19] without modification."；p12："The pretrained occupancy model was trained using four NVIDIA Tesla V100 SXM2 32 GB GPUs" |
| 代码加载的权重 | 先 `load_pretrained_macarons(pretrained_macarons.pth)`，再用 `trained_macarons.pth` 的 `model_state_dict` 覆盖 | 【已核实】 | OC CT:21 `"model_name": "trained_macarons.pth"`；OC `macarons/testers/magician_planning.py:225-230`；我们的适配层 `experiments/exp5_future_view/task0/magician_comparison/adapters/magician_core.py:34-39` 同样做法 |
| sha256 | `trained_macarons.pth` = `34c5247f13bedda0…`；`pretrained_macarons.pth` = `c7d66c9f…` | 【已核实】 | W00-2:13-14、:19-20；CHECKPOINT_AUDIT.md:30 |
| 覆盖是否完整 | trained 与 pretrained 的键集合相同（去掉 `module.` 前缀后：depth 150 个、scone 232 个），所以载入后实际生效的全部是 trained 的参数 | 【推断】 | 本次 CPU 读取（`torch.load(weights_only=True)`，只放行 numpy scalar / dtype） |
| checkpoint 内容 | `epoch` = 42；`train_losses` 等数组各 84 项；**没有**配置、场景列表或 `val_losses` | 【已核实】 | 本次 CPU 读取。各键与上游训练器保存的字典逐项一致（OC `macarons/trainers/train_macarons.py:1596-1607`，其中 `# 'val_losses'` 被注释掉） |
| 权重是谁训练的 | MACARONS 作者在 2023-05 发布；MAGICIAN 直接复用 | 【推断】 | Drive 文件夹（OC README.md:84 给出的链接）顶层只有 macarons / resnet / scone 三个目录，修改日期都是 2023-05-16（本次 WebFetch：https://drive.google.com/drive/folders/1wyc9_QFmcxOz4oerE8kCQ3I8LO5zioZL）；MACGH README 的下载说明里有 `trained_macarons`（本次 WebFetch，工具摘要） |
| MAGICIAN 自己是否训练 / 微调 | 测试路径全程 `torch.no_grad()`、`macarons.eval()`；仓库里只有 MACARONS / SCONE 的训练器，没有 MAGICIAN 专用训练脚本 | 【推断】没有训练 | OC `test_magician_planning.py:25`；`magician_planning.py:412`；OC `macarons/trainers/` 只有 `train_macarons.py`、`pretrain_scone_occ.py`、`pretrain_scone_vis.py` |
| ShapeNet 版本 | MAGICIAN 论文只写 "ShapeNet [9]"。代码配置是 `./data/ShapeNetCore.v1`；SCONE 论文是 ShapeNetCore v1 | 【推断】v1 | OC `configs/scone/occupancy/occupancy_pretraining_config.json:14`；`macarons/utility/scone_utils.py:123,135`（`train_categories` / `test_categories`）；SCP p17 |
| 表 5（MP3D 微调消融） | Pretrained AUC 0.652 / Cov 0.888；Fine-tuned 0.646 / 0.893。评估场景是 Sestino Museum、St. Sofia、Neuschwanstein | 【已核实】 | MAGP p8："we fine-tuned the occupancy model on the MP3D dataset and evaluated on these three scenes. Tab. 5 shows that fine-tuning on indoor scenes does not provide any clear improvement."；p7："three unseen and challenging scenes: Sestino Museum, St. Sofia Church, and Neuschwanstein Castle" |
| 表 5 的微调细节 | 用了哪些 MP3D 房屋、多少步、从哪份权重开始，论文都没有写；微调后的权重在本地 Drive 副本中不存在（W00-2 只有 9 个文件） | 【未知】（细节）；【推断】未发布 | MAGP p8；W00-2 全部条目 |
| 表 2 / 4 用的是哪份权重 | MP3D 表 4 明确未微调；Macarons++ 表 2 没有点名权重文件 | 表 4【已核实】；表 2【未知】（【推断】为 `trained_macarons.pth`） | MAGP p7："even without further fine-tuning on MP3D dataset" |

### A2. MACARONS（Guédon et al., CVPR 2023）

| 项 | 事实 | 标记 | 证据 |
|---|---|---|---|
| 训练场景 | 8 个：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | 【已核实】 | CM:18-27（FK 与 OC 相同）；MACGH 原始 `configs/macarons/macarons_default_training_config.json`:18-26 也相同；MACP p8 表 1："The 8 scenes above the bar were seen by MACARONS during self-supervised training (but with different, random starting camera poses and trajectories), and the 4 scenes below the bar were not." |
| val / test 标签 | val = alhambra、neuschwanstein；test = bridge、eiffel | 【已核实】 | CM:28-35；MACGH 同名文件 :28-34 |
| val 是否用于选 checkpoint | 发布的训练器**丢弃** val loader；checkpoint 按训练 depth loss 命名为 `unvalidated_` / `best_unval_`。论文说下方 4 个场景训练时"were not"见过。作者在发布代码之外是否用 alhambra / neuschwanstein 选过 epoch 42，没有证据 | 【已核实】（代码）；【未知】（实际做法） | OC `train_macarons.py:1394` `train_dataloader, _, _ = get_dataloader(`（上游 MACGH 同一行在 :1399）；:1503 `# print("Validation data:"…`；:1590 `current_loss = depth_loss_per_epoch[-1]`；:1594 `"unvalidated_"`；:1610 `if current_loss < best_train_loss:`；MACP p8 |
| 训练方式 | 在线自监督：深度从 RGB 预测，占据由 space carving 得到伪 GT，覆盖增益也是自监督；不用显式 3D 监督 | 【已核实】（论文）；深度确实被训练【推断】 | MACP p7："self-supervised pipeline that does not need any 3D input data"；MACGH 原始配置 :73 `"use_perfect_depth": false`。本次 CPU 比对：trained 与 pretrained 的 150 个 depth 张量**全部不同**（若深度冻结，可学习权重应保持不变，参见 MW `jiahui/stage_3/P0_baseline_audit.md:74-84`） |
| OC / FK 中的 `use_perfect_depth: true` | 是 MAGICIAN 为测试改的配置值，不代表训练时的设置 | 【推断】 | CM:73 `"use_perfect_depth": true`，而上游 :73 为 `false`；MW `jiahui/MACARONS_ALIGNMENT.md:49-55`（"这 7 处全是作者为**测试**改的"） |
| 训练起点 | 在整个 `pose_space` 上均匀随机抽一个 5D 位姿，只剔除视野为空或视野内没有 proxy 点的位姿；**不读** settings 的 `start_positions` | 【已核实】（代码） | OC `train_macarons.py:135-137`（"# Select a random, valid camera pose as starting pose" / `camera.get_random_valid_pose(...)`）；上游 :139 相同；OC `macarons/utility/macarons_utils.py:2171-2201`（`np.random.choice(list(self.pose_space.keys()))`）；MACP p15："We sample a random camera pose in the scene, and let our model ex[plore]" |
| 训练轨迹 | 先随机走一步（OC :146）；之后每步在合法邻居中选**模型预测**覆盖增益最大的（OC :290-311、:337）；每条 100 个位姿，每场景每 epoch 1 条 | 【已核实】（代码）；条数【推断】 | MACP p15："We perform up to 360 trajectories to make the model converge."；checkpoint 为 epoch 42、84 项损失 = 8 个场景 / 4 卡 = 每 epoch 2 次迭代 → 每个训练场景约 42 条轨迹（【推断】） |
| ShapeNet 预训练 | 主结果用 "Pretrained" 版本：论文表 1 的 MACARONS 列与表 4 的 "Pretrained 1 MRI" 列逐值相同（均值 0.726）；发布配置 `pretrained: true`、`pretrained_model_path: pretrained_macarons.pth`；Drive 另有 `trained_macarons_no_pretraining.pth` | 主结果【已核实】；`trained_macarons.pth` = Pretrained 版本【推断】 | MACP p20 表 4 题注："(Pretrained) … complete pretraining on ShapeNet with explicit 3D supervision"；MACGH 原始配置 :134、:136；W00-2:25-26 |
| 是 1 MRI 还是 3 MRI | 配置 `n_memory_scene_loops: 2`，与论文的 1 / 3 都对不上 | 【未知】 | MACGH 原始配置 :123；MACP p20 |
| 发布的 checkpoint | initialized `0d40b487`、pretrained `c7d66c9f`、trained `34c5247f`、trained_no_pretraining `3ee7dba5`、resnet depth `58bf0958`，以及 scone occ / vis 的 initialized / pretrained 共 4 个 | 【已核实】 | W00-2 全部 9 个条目 |
| MACARONS 论文的评估起点 | "We use the same starting camera poses … for each method"；图 10："starting from random camera poses"。是否就是 Macarons++ settings 的 `start_positions`，没有证据 | 【未知】 | MACP p8、p19 |
| MAGICIAN 论文中的 MACARONS 列 | 测试器冻结（`eval`、`no_grad`），按 settings 的 5 个起点运行，`model_name` 取 CT 默认的 `trained_macarons.pth`。论文用的是否就是这份配置，没有直接证据 | 代码【已核实】；论文实际所用【未知】 | OC `macarons/testers/scene.py:419,523,549,688-689`；OC `test_scenes.py:25` |

### A3. SCONE（Guédon et al., NeurIPS 2022）

| 项 | 事实 | 标记 | 证据 |
|---|---|---|---|
| 训练数据 | ShapeNetCore v1。训练集 4,000 个网格，取自 8 个类别：Airplane、Cabinet、Car、Chair、Lamp、Sofa、Table、Vessel。另有同类别的 val 400 / test 400，以及 8 个未见类别的 test 400 | 【已核实】 | SCP p7："We sample 4,000 training meshes from 8 specific categories of objects, 400 validation meshes and 400 test meshes from the same categories, and 400 additional test meshes from 8 categories unseen during training."；p17："with meshes from ShapeNetCore v1 [2]"；p19 类别列表；MACGH README 类别表（WebFetch 摘要，Watercraft = Vessel） |
| 相机采样 | 全部在物体周围的球面上。训练时每个网格随机取 n_i 个位姿作为历史；覆盖损失用球面上的稠密相机集 | 【已核实】 | SCP p19："All camera poses are sampled on a sphere around the object."；p17-18（B.1/B.2："we sample a random number ni of camera poses"；"we sample a dense set of Ncam camera poses … on a sphere around the object"）；OC `configs/scone/occupancy/occupancy_pretraining_config.json:37-41`（`camera_dist` 1.5、`pole_cameras` true、5 个仰角 × 10 个方位、`n_camera` 52） |
| 是否在场景上训练 | 没有。场景实验直接用 ShapeNet 模型，从随机位姿出发 | 【已核实】 | SCP p8："Note we did not retrain our model for such scenes and use the previous model trained on ShapeNet"；"The depth sensor starts from a random pose" |
| MAGICIAN 表 2 的 SCONE 列 | 用了哪份权重未知。上游 e4e08fe 的 `scone_planning.py` 通过 `setup_test` 加载 CT 的 `model_name`；e4e08fe 的 CT 默认值是 `trained_macarons.pth`。Drive 中另有只在 ShapeNet 上训练的 `pretrained_scone_occ/vis.pth` | 【未知】 | `git show e4e08fe:macarons/testers/scone_planning.py`:266-285、:611、:570-578；`git show e4e08fe:configs/test/test_in_default_scenes_config.json`（`"model_name": "trained_macarons.pth"`）；W00-2 的 scone 条目 |

### A4. FisherRF（Jiang et al.）

| 项 | 事实 | 标记 | 证据 |
|---|---|---|---|
| 是否有学习先验 | 原方法在当前已采图像上在线优化 3DGS（SplaTAM），用 Fisher 信息评估候选视角，用 frontier 产生路径；全文没有 "pretrain" / "pre-train" 字样 | 【推断】没有跨场景的学习先验 | FRP p2："compute the information gain between the training dataset and the candidate viewpoint pool using only the camera parameters of the candidate views"；p8："We build our active mapping system on top of existing passive mapping architecture SplaTAM … We employ frontier-based exploration [64] to propose path candidates"；全文 grep |
| MAGICIAN 的改编 | 改了 frontier 选择和动作空间 | 【已核实】 | MAGP p6 脚注 1："We adapt the released FisherRF code to outdoor scenes by modifying its frontier selection and adjusting its action space." |
| 改编代码 | OC HEAD 中没有。上游 e4e08fe 有 `macarons/testers/fish_planning.py`（853 行，"fisher" 出现 0 次），在 4df5063 删除。它通过 `setup_test` 加载 `trained_macarons.pth`，用 MACARONS 的 SCONE 模块预测覆盖增益，给 300 个采样位姿打分，再走 Dijkstra 路径 | 代码【已核实】；是否就是论文 FisherRF 列的来源【未知】 | `git show e4e08fe:macarons/testers/fish_planning.py`:266-285、:540-541、:552-561、:583；`git log -- macarons/testers/fish_planning.py` → 4df5063、e4e08fe |
| 含义 | 如果论文 FisherRF 列确实来自这个文件，那么这一列也继承了 MACARONS 在 8 个场景上的训练 | 【推断】（条件成立时） | 同上 |

### A5. Random Walk

| 项 | 事实 | 标记 | 证据 |
|---|---|---|---|
| 是否训练 | 不需要训练。上游测试器虽然加载权重，但动作是对邻居 `random.shuffle` 后选取，不调用模型打分 | 【已核实】（e4e08fe 代码） | `git show e4e08fe:macarons/testers/random_planning.py`:162（`load_state_dict`）、:486（`random.shuffle(neighbor_indices)`）；OC HEAD 的 `test_random_planning.py` 导入的模块已不存在（OC `macarons/testers/` 只剩 `magician_planning.py`、`scene.py`） |
| 论文列的来源 | 没有证据 | 【未知】 | — |

### A6. 关键问题：基线是否在评估场景上训练过、起点是否可能重叠

| 场景（Table 1 的 14 个） | MACARONS 权重中的角色（CM） | 位姿空间大小 | MACARONS 训练起点与 5 个评估起点重合的可能 |
|---|---|---|---|
| dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | **train** | 27,000 / 28,000 / 42,000 / 18,000 / 39,000 / 30,000 / 15,000 / 49,000 | 起点在全网格上随机抽，不读 `start_positions`。按每场景约 42 条轨迹估算，起点**恰好**落在 5 个评估起点之一的期望次数约为 42 × 5 / |pose_space|，即 0.004–0.014 / 场景，8 个场景合计约 0.06（【推断】：忽略合法性剔除，并假设 MACARONS 原始数据的 settings 网格与 Macarons++ 相同，后者【未知】）。训练轨迹在 100 步内**途经**评估起点的概率无法估计，因为轨迹没有发布（【未知】） |
| alhambra、neuschwanstein | val（是否真的用于选择【未知】，见 A2） | 33,000 / 39,200 | 按发布代码没有在这两个场景上训练（【已核实】）；论文也说没有见过（MACP p8） |
| bridge、eiffel | test | 30,000 / 21,600 | 没有训练 |
| barts、sestino_museum（以及 sofia_church） | 不在 MACARONS 数据中（MAGICIAN 新增的室内场景） | 7,560 / 7,560 /（8,640） | 没有训练 |

- 位姿空间大小 = `pose_l × pose_w × pose_h × n_θ × n_azim`，本次从 `MAGICIAN/data/Macarons++/<scene>/settings.json` 计算（【已核实】）。
- SCONE（只用 ShapeNet）和 Random Walk 不在任何场景上训练。FisherRF 原方法没有离线训练，但 MAGICIAN 改编版可能依赖 MACARONS 权重（A4，【未知】）。

---

## B. 我们的模型

### B1. 总表（sha 取前 8 位；完整值见 REG 的对应行）

| 行 ID（REG） | sha | 初始化 | 室外训练数据 | MP3D | 本模型训练步数 | 训练密度 | checkpoint 选择 | 主要证据 |
|---|---|---|---|---|---|---|---|---|
| V2O（REG:102-109） | a379c0fb | 随机，seed 0 | train6_minus_Sc，138 条轨迹 / 98,887 视角 | 无 | 60,000 | S10 | 固定 EMA@60k | `experiments/exp5_future_view/task1/configs/train_main.json:6-8,13`；`task1/scripts/v2_data.py:176-191`（`load_split`）；CHECKPOINT_AUDIT.md:79 |
| B0 = B1 = B2（REG:147-152、444-449、189-194） | 343f41ab | V2O EMA@60k（sha 断言） | 同上，S10 | MP3D 官方 train：**59 个房屋 / 114 个楼层 / 228 条轨迹** | 运行 20,000，选出 17,500 | S10 | ES 比值规则，选择集 = MP3D valsel（6 个房屋 / 9 个楼层）+ S-c；不看闭环 | `task2/configs/train_mix_v1.json:6-9,14`；`task2/scripts/t22_train_mix.py:68-69,88-89`；`task2/protocol.md:26-27,57-61`；`work_dirs/exp5_future_view/task2/mp3d/manifests/traj_jobs.json`（本次统计：train 228 jobs / 114 层 / 59 房屋，valsel 9 / 9 / 6） |
| E2D（REG:224-229） | 61439612 | V2-Mix EMA@17.5k（sha 断言） | 同上，D100 重建分片 | 同上（D100） | +5,000 | D100 | 固定 EMA@5k | `task3_full_depth_p1/e2_pilot/configs/e2_d100_v1.json:6-9,14`；`experiments/exp6_v2/E2/configs/e2.json:66` |
| E2S（REG:415-420） | 578855bb | 同上 | 同上，S10 | 同上（S10） | +5,000 | S10 | 固定 EMA@5k | `e2_pilot/configs/e2_s10_v1.json:6-9,14` |
| V3（= V3-T1 40K，REG:267-272） | 639c9d3a | 随机，seed 0 | `e2/out_views_d100` 中非 heldout 部分（= train6_minus_Sc）+ TSDF 缓存 | **有**（混合，32 + 32 视角） | 40,000 | D100 + TSDF | ES 比值（MP3D in_val + S-c，≥ 5k 取最小）→ 40k | `experiments/exp6_v3/configs/train_v3.json:5-6,47-51`；`seen2scene/v3/trainer.py:117-118`；`seen2scene/v3/data.py:43-58`。运行记录 `work_dirs/exp6_v3/task1_xattn/runs/main/run_none.json` 的 `data_counts` = {in_train 228, in_val 9, out_train 138, out_sc 42}，`data_files` 中 out_train / out_sc 只有 6 个 Train-6 场景 |
| V3-Y45K-H0 / V3-Y50K-H0（REG:653-658、665-670） | 37da75b2 / 64cea1d6 | V3-T1 真实 40K 训练状态（`ckpt_last` `3d819782…`，含 AdamW） | 与 V3 **完全相同**（混合） | **有** | +5,000 / +10,000（在 50K 停止） | D100 + TSDF | **按 Dev-5 闭环 AUC@100 选**：Y45K 0.732 < Y50K 0.741 → Y50K | `experiments/exp6_v3/Y/configs/train_y.json:59-61,67-73`；`Y/FROZEN_PROTOCOL.md:7,21,29,33`；`Y/scripts/train_y.py:2-8`（共享训练器不改）；运行记录 `work_dirs/exp6_v3/Y/runs/main/run_none_1793791215.json` 的 `data_counts` 与 V3 相同，`resumed_at` 40000；`Y/reports/G3_DEV_AND_ALL14.md:9-10` |
| V3-H3、V3-D1-H3-NORISK（REG:590-595、605-610） | 639c9d3a | = V3 | — | — | **0**（只换规划器） | — | D1 是在 D1 / D2 / D3 中按 Dev-5 AUC@100 选出的 | `exp6_v3/H/MODEL_AUDIT.md:10`（"本实验中的训练 **0 次**"）；`exp6_v3/ablation_study/experiment_D_h3_lookahead/FROZEN_PROTOCOL.md:29`；`H/FROZEN_PROTOCOL.md:51` |
| V3-YBEST-H3（REG:677-682） | 64cea1d6 | = Y50K | — | — | 0（只换规划器） | — | 权重是按 Dev-5 选出的（见上） | `Y/FROZEN_PROTOCOL.md:33-36` |
| V3.1（REG:468-473） | 13df0a45 | 随机（`scratch`） | 只用室外（138 条轨迹 + S-c 42 条作监控），D100 + TSDF | 0 | 40,000 | D100 + TSDF | 固定 EMA@40k | `exp6_v3.1/configs/outdoor_train.json:5-9,12,63-65`；`exp6_v3.1/scripts/trainer_outdoor.py:53-59`；运行记录 `data_counts` {out_train 138, out_sc 42} |
| V4（REG:308-313） | 1b31afff | 随机（`scratch`） | train6_minus_Sc，`out_views_d100` | 0 | 40,000 | D100 | 固定 EMA@40k | `exp6_v4/configs/outdoor_train.json:5,8-12,20` |
| V4C（REG:497-502） | c4190f1d | V4 EMA@40k，全部参数 | 同 V4 | 0 | +12,000 | D100 | 固定 EMA@12k（"no selection on S-c, Dev-3, Test-3 or MP3D"） | `exp6_v5/configs/train_v4_continue.json:2-7,43` |
| V5（REG:349-354） | 5ef30ac4 | V4 EMA@40k（encoder + U-Net） | 同 V4，另加 Train-6 多表面标签 | 0 | +12,000（Bridge 2k + Joint 10k） | D100 | 固定 EMA@12k | `exp6_v5/configs/train_v5_outdoor.json:2-7,43,102,114` |
| F1-GQ-v1@5k（REG:545-550） | a3f47c03 | B0 的 encoder / UNet2D 逐位复制 + 随机 decoder（因此**间接**继承了 MP3D 混合训练） | train6_minus_Sc，S10 | 本阶段 0 | +5,000 | S10 | **按 Dev-3 闭环 AUC@20 在 2k / 5k 中选** | `exp6_v5/F1/PROTOCOL.md:26,30-31,42,81-84`；`F1/reports/TABLE1_FOLLOWUP.md:4`；运行记录 `work_dirs/exp6_v5/F1/runs/B/run_none.json`：`train_trajs` 138、`train_views` 98,887、`mp3d_training_samples` 0 |

- **外部预训练**：所有行都没有 ShapeNet 或 ImageNet 预训练权重（【推断】）。依据：
  - 配置是 `scratch`，或从我们自己的父 checkpoint 初始化；
  - `seen2scene/futureview`、`seen2scene/v3` 以及上述配置目录中 grep "shapenet"、"imagenet"、`from_pretrained`、`torch.hub` 都没有结果。
- **监督信号**：标签是用光栅化器渲染 GT 网格得到的第一交点 t*（`task2/protocol.md:43`）。这是 GT 3D 监督，不同于 MACARONS 的 RGB 自监督（【推断】：依据是"光栅化器 … 第一交点"这一描述，没有逐行追代码）。V5 另用 GT 网格的多表面标签（`train_v5_outdoor.json:114`）。

### B2. Train-6 的起点核对（本次对 BEL 的 `sim_meta.json` 与 settings 的 `start_positions` 逐条比对；划分取自 FV）【已核实】

| 场景 | S-c 留出的官方起点 | 训练集中以官方起点 k 开始的轨迹数 | 训练集中途经官方起点 k 的轨迹数（任意步） | 起点 4 |
|---|---|---|---|---|
| bannerman | 0 | k1: 5、k2: 4、k3: 4（另有 9 条非官方起点） | 1: 5、2: 4、3: 4 | 从未出现 |
| dunnottar | 3 | k0: 6、k1: 5、k2: 4 | 0: 6、1: 5、2: 4 | 从未出现 |
| fushimi | 0 | k1: 5、k2: 4、k3: 4 | 1: 5、2: 4、3: 4 | 从未出现 |
| liberty | 2 | k0: 6、k1: 5、k3: 4 | 0: 6、1: 5、3: 4 | 从未出现 |
| pisa | 3 | k0: 6、k1: 5、k2: 4 | 0: 6、1: 5、2: **5** | 从未出现 |
| redeemer | 0 | k1: 5、k2: 4、k3: 4 | **0: 1**、1: 5、2: 5、3: 4 | 从未出现 |

- **划分规模**：每个场景 30 条轨迹，train / S-c = 22–24 / 6–8 条；合计 train 138 条，S-c **42** 条（FV `split.scenes.*`）。
- **训练集途经 S-c 起点的唯一一例**：redeemer 的 `rwm-A-0001` 在 t = 17 经过官方起点 0。
- **DATASET_AUDIT §2.4 的原话**是"官方起点 0–3 都有训练轨迹（6、5、4、4 条），起点 4 从未用作起点"。这对 180 条全集成立；在 train6_minus_Sc 中，每个场景只剩 3 个官方起点作为训练起点。
- **Dev-5、Test-3、sofia_church** 没有任何训练轨迹（【已核实】）：
  - BEL 的 `trajectories/` 下只有 6 个 Train-6 目录；
  - `work_dirs/exp5_future_view/task1/data/{views,tstar}` 与 `…/task3_full_depth_p1/e2/out_views_d100` 下也都只有这 6 个场景目录。

### B3. 训练轨迹由哪些策略生成（需要披露）

| 策略 | 是否用 GT | 每个场景的条数（全集 / train） | 证据 |
|---|---|---|---|
| RW、RW-M、FR | 否 | 8 / 6 / 6 | `experiments/exp4_multiworld_belief/task0/trajectory_generation.md:42-44` |
| CG-ε（真实一步增益贪心，ε = 0.2） | **是**（只用来选动作） | 8 | 同上 :45；`sim_meta.json` 中 `"eps": 0.2` |
| **MG = MAGICIAN beam 10×10 + 官方 `trained_macarons.pth`** | 否 | 2；train 中共 **9** 条：bannerman 1、dunnottar 2、fushimi 1、liberty 2、pisa 2、redeemer 1 | 同上 :46；`work_dirs/exp4_multiworld_belief/task0/pilot/mg_runs/bannerman/episodes/ep_magician_bannerman_s0_r0.json` 的 `model.trained_sha256` = `34c5247f…`、`beam` [10, 10] |

- **含义**【推断】：我们所有模型的训练观测分布中，有一小部分来自 MAGICIAN / MACARONS 权重驱动的相机轨迹，以及 GT 驱动的轨迹。这两类只决定"看哪里"，标签仍来自 GT 网格。

### B4. 评估场景在训练之外的使用（选择与开发）

| 用途 | 涉及的行 | 证据 | 标记 |
|---|---|---|---|
| 用 Dev-5 闭环选 checkpoint | V3-Y50K-H0、V3-YBEST-H3 | `Y/FROZEN_PROTOCOL.md:33`；`Y/reports/G3_DEV_AND_ALL14.md:9-10` | 【已核实】 |
| 用 Dev-3 闭环选 checkpoint | F1-GQ-v1@5k | `F1/PROTOCOL.md:81-84`；`F1/reports/TABLE1_FOLLOWUP.md:4` | 【已核实】 |
| 用 Dev-5 选规划器变体 | V3-D1-H3-NORISK | `experiment_D_h3_lookahead/FROZEN_PROTOCOL.md:29` | 【已核实】 |
| Dev-3 是 V2 系列的开发集；B2 / E2 的设计是在看过 B0 / B1 在 Dev-3 + Test-3 的结果之后定的 | V2O、B0/B1/B2、E2D/E2S 及后续 | `task2/protocol.md:35`；CHECKPOINT_AUDIT.md:167 | 【已核实】（文档记载） |
| Test-3 已多次评估，不是盲测 | 全部 | DATASET_AUDIT.md:311-314 | 【已核实】（文档记载） |
| 不用评估场景选择 | V2O、E2D、E2S、V3.1、V4、V4C、V5（固定步数）；B0/B1/B2、V3（只用 S-c + MP3D valsel） | 见 B1 | 【已核实】 |

### B5. 附注：MP3D（Table 1 之外）
- 我们的 MP3D 训练房屋包含 MAGICIAN 5 个 MP3D 评估场景中的 2 个：GdvgFV5R1Z5、HxpKQynjfin；pLe4wQe7qrG 在我们的 valsel 中。
- 影响：只在拿我们的模型与 MAGICIAN 在 MP3D 上比较时需要披露；对 Macarons++ Table 1 没有影响。
- 证据（【已核实】）：`work_dirs/exp5_future_view/task2/mp3d/manifests/traj_jobs.json`（例如 :624、:709、:2100）；OC README.md:63。

---

## C. 公平性披露表（只是披露：训练数据不同本身不等于比较不公平）

| 方法 | ShapeNet 预训练 | 用 Macarons++ 场景数据训练 | 用 MP3D 训练 | 训练过的场景 ∩ 14 个评估场景 | 评估起点重叠 | 评估场景用于选择 |
|---|---|---|---|---|---|---|
| MAGICIAN（MAG-B / MAG-G，`trained_macarons.pth`） | 是（论文 + 配置；【推断】正是这份权重） | 是：8 个场景，RGB 自监督，无 GT 3D | 否（表 5 的微调版本未发布、未使用） | **8 个**：dunnottar、pisa、bannerman、redeemer、pantheon、fushimi、liberty、colosseum | 训练起点是全网格随机位姿；精确重合的期望很小（【推断】），途经【未知】 | alhambra、neuschwanstein 标为 val，实际是否使用【未知】 |
| MACARONS（论文列） | 同上 | 同上 | 否 | 同上 | 同上 | 同上 |
| SCONE（论文列） | 是（v1，8 个类别） | 原方法：否；MAGICIAN 实际所用权重【未知】 | 否 | 原方法：0；实际【未知】 | 原方法：不适用 | — |
| FisherRF（论文列） | 原方法：否 | 原方法：否；MAGICIAN 改编版可能用了 MACARONS 权重（【未知】） | 否 | 原方法：0；改编版【未知】 | — | — |
| Random Walk | 否 | 否 | 否 | 0 | 不适用 | — |
| V2O | 否 | 是：Train-6 去 S-c（GT 监督） | 否 | 6 个 Train-6 场景 | 每个 Train-6 场景有 3 个官方起点是训练起点；起点 4 没有；Dev / Test 没有 | 否（Dev-3 作开发参照） |
| B0 / B1 / B2、E2D、E2S | 否 | 同上 | 是 | 同上 | 同上 | 否（B2 / E2 的设计参考过 Dev-3 + Test-3 结果） |
| V3、V3-H3 | 否 | 同上 | 是 | 同上 | 同上 | 否 |
| V3-D1-H3-NORISK | 否 | 同上 | 是 | 同上 | 同上 | 规划器按 Dev-5 选 |
| V3-Y50K-H0、V3-YBEST-H3 | 否 | 同上 | 是 | 同上 | 同上 | checkpoint 按 Dev-5 选 |
| V3.1、V4、V4C、V5 | 否 | 同上 | 否 | 同上 | 同上 | 否 |
| F1-GQ-v1@5k | 否 | 同上 | 间接（经 B0 初始化） | 同上 | 同上 | checkpoint 按 Dev-3 选 |

- 双方都"见过"的场景：Train-6 对我们和 MACARONS 权重都是训练场景。pantheon、colosseum 只对 MACARONS 权重是训练场景，对我们是 Dev。
- 双方都没训练过的场景：eiffel、bridge、barts、sestino_museum。alhambra、neuschwanstein 两边都没训练过，但 MACARONS 一方的 val 用途【未知】。

---

## D. 与已有审计的出入（建议在正式报告中更正，本次没有改原文件）

1. **DATASET_AUDIT.md:123**："共 44 / 180 条轨迹（bannerman 8、dunnottar 6、fushimi 8、liberty 6、pisa 6、redeemer 8）"。
   - 括号内各场景之和为 42；
   - FV 的 `heldout_trajs` 合计 42，V3 运行记录的 `out_sc` 也是 42。
   - 应为 **42 / 180**，与 CHECKPOINT_AUDIT.md:90 一致。
2. **CHECKPOINT_AUDIT.md:78**（"完美深度（官方）"）与 **REG:69**（"官方训练用完美深度"）没有证据支持。
   - 上游 MACARONS 训练配置是 `use_perfect_depth: false`（MACGH 原始文件 :73）；
   - `trained_macarons.pth` 的 150 个 depth 张量与 `pretrained_macarons.pth` 全部不同（本次 CPU 比对）；
   - OC / FK 中的 `true` 是 MAGICIAN 为测试改的值（A2）。
   - 应改为"RGB 自监督深度（【推断】）"。
3. **DATASET_AUDIT.md:143**："alhambra、neuschwanstein 被 MACARONS 用来选 checkpoint"。
   - 配置中只有 `val_scenes` 这个标签；
   - 发布的训练器丢弃了 val loader，并按训练损失保存 checkpoint（A2）。
   - 应改为"标为 val，实际是否用于选择未知"。
4. **CHECKPOINT_AUDIT.md:80**："MP3D train（61 个房屋，114 个楼层，228 条）"。
   - 61 是官方 train 房屋总数；
   - 实际进入训练的是其中 **59** 个房屋的 114 个楼层（`task2/protocol.md:26` 的写法是准确的）。
5. **CHECKPOINT_AUDIT §4 / REG 的数据描述**没有提到训练轨迹中有 MAGICIAN（MACARONS 权重）生成的 MG 轨迹和 GT 驱动的 CG-ε 轨迹（B3）。建议补充披露。

---

## E. 未知项清单（及缺少的证据）

| # | 未知项 | 缺少的证据 |
|---|---|---|
| U1 | MAGICIAN 论文表 2 中 SCONE、FisherRF、Random Walk 三列实际用的代码和权重 | 作者的运行配置 / 日志；e4e08fe 中的测试器默认加载 `trained_macarons.pth`，但不能证明论文就是这样跑的 |
| U2 | MAGICIAN 表 2 的 MAGICIAN / MACARONS 列是否就是 `trained_macarons.pth` | 论文没有点名权重文件（【推断】是） |
| U3 | 表 5 MP3D 微调的细节（房屋、步数、初始化）以及权重 | 论文与发布物中都没有 |
| U4 | MACARONS 是否用 alhambra / neuschwanstein 选 epoch 42；epoch 42（配置为 105 个 epoch）是怎么定的 | 作者的训练日志；checkpoint 中没有 `val_losses` |
| U5 | `trained_macarons.pth` 对应论文表 4 的 1 MRI 还是 3 MRI | 配置值为 2，与两者都不符 |
| U6 | MACARONS 训练轨迹是否途经 Macarons++ 的 5 个评估起点；MACARONS 原始数据的 settings 网格是否与 Macarons++ 相同 | 训练轨迹 / 记忆目录没有发布；原始 settings 没有核对 |
| U7 | MACARONS 论文评估用的起点是否就是 Macarons++ 的 `start_positions` | 论文只说"same starting camera poses"和"random camera poses" |
| U8 | MAGICIAN 是否以 MACARONS 以外的方式在 Macarons++ 上做过训练或微调 | 仓库和论文中都没有发现；只能确认发布代码不训练（【推断】没有） |
| U9 | 我们的标签是否全部由 GT 网格渲染（未逐行追溯标签生成代码） | 需要读 T1.1 标签生成脚本（`task1/data/run_labels.sh` 等） |

---

## F. 本次操作记录
- **只读**：OC、FK、MW 的代码、配置和权重；R 下的配置、运行记录（`run_none*.json`）、FV 与 BEL 的 `sim_meta.json`；MAGP 文本与 PDF。
- **CPU 读取**：`trained_macarons.pth`、`trained_macarons_no_pretraining.pth`、`pretrained_macarons.pth`。
  - 先用 `pickletools` 确认 pickle 中只有 numpy scalar / dtype 与 torch 重建全局；
  - 再用 `weights_only=True` 加 safe globals 读取。
- **下载**（不可信数据，只做文本提取）：
  - 论文 PDF：MACP、SCP、FRP → `/tmp/audit_dl_{macarons,scone,fisherrf}/`；
  - MACGH 原始文件 `configs/macarons/macarons_default_training_config.json`、`configs/test/test_in_default_scenes_config.json`、`macarons/trainers/train_macarons.py` → `/tmp/audit_dl_macarons_repo/`。
  - 这些都在 /tmp，会被清理，复核时以 URL 为准。
- **WebFetch**（结果经工具摘要，引文可能不是逐字的）：MACGH README；Drive 文件夹顶层。
- **辅助脚本**：`/tmp/audit_scripts_tf/`（`peek*.py`、`keys*.py`、`starts*.py`、`runinfo.py`，均为只读统计）。
