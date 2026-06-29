# MACARONS 训练复现笔记（冒烟测试）

> 目标：先做**冒烟测试**——让作者的训练流水线（`macarons/trainers/train_macarons.py`）在我们的 L40 节点上真正跑起来几个 step，确认能跑通；正式训练再上 4×L40。
> 关键结论：**作者发布的训练入口无法直接运行**——test 路径用不到的那些函数里藏了多个 bug（循环导入、DDP-only 假设、重构未同步、数据网格不一致）。冒烟测试逐个把它们暴露并修掉。

---

## 1. 怎么跑冒烟测试

```bash
cd /bsuscratch/jiahuizhang/projects/occupancy/MAGICIAN
source activate_magician.sh
python -u train_macarons_run.py -c smoke_test_config.json > jiahui/log/smoke_train.log 2>&1
```

- **启动脚本**（自建，仓库原本没有训练入口）：`train_macarons_run.py`
  载入配置 → 单卡直接 `run_training`；`ddp:true` 时走 `mp.spawn`（为 4 卡准备）。
- **冒烟配置**（自建）：`configs/macarons/smoke_test_config.json`
  基于 `macarons_default_training_config.json`，仅改：
  - `data_path → ./data/Macarons++`（原 `./data/scenes`）
  - `train_scenes → ["pantheon"]`、`val/test → 各 1 个`
  - `numGPU: 0`（原训练配置**缺**这个键，单卡 `setup_device` 会用到）
  - `n_poses_in_trajectory: 5`、`epochs: 1`、`empty_cache_every_n_batch: 1`
  - `macarons_model_name: smoke_trained` ⚠️ 改名，**绝不覆盖作者的 `trained_macarons.pth`**

---

## 2. 修掉的作者代码 bug（test 路径都不会触发）

| # | 现象 | 文件:行 | 根因 | 修复 |
|---|---|---|---|---|
| 1 | `NameError: setup_device` / `import *` 拿到空壳 | `utility/macarons_utils.py:10` ↔ `trainers/train_macarons.py:5` | **循环导入**：两文件互相 import；`macarons_utils` 顶部 `from ...train_macarons import recompute_mapping`，而该名字只在一段 `'''注释'''` 里用到（死代码） | 删除那行无用导入，环断开 |
| 2 | `AttributeError: 'RandomSampler' object has no attribute 'set_epoch'` | `trainers/train_macarons.py:1565` | **DDP-only 假设**：`set_epoch` 只有 `DistributedSampler` 有；单卡是 `RandomSampler` | 加 `if params.ddp or params.jz:` 守卫 |
| 3 | `TypeError: '>=' not supported between 'tuple' and 'Tensor'` | `trainers/train_macarons.py:48` (`setup_scene`) | **重构未同步**：`get_scene_gt_surface` 改成返回 `(gt_surface, gt_normals)`，test 路径已解包，训练路径仍当单值 | 解包：`gt_surface, gt_normals = get_scene_gt_surface(...)` |
| 4 | `KeyError: '[5, 5, 10]'`（`check_if_pose_is_occupied`） | `utility/macarons_utils.py:4135` | **数据网格不一致**：`settings.json` 相机网格 10×6×13，但 `occupied_pose.pt` 的 `X_idx` 只到 [7,3,6]（8×4×7 旧网格）。训练靠 `get_random_valid_pose` 随机起点才会查这个字典；test 用固定 `start_positions`+`update_camera`，从不查 | 查不到的位置当未占据：`self.pose_is_occupied.get(X_key, False)`（fov 检查仍过滤坏位姿）。正式训练若要严格，应重新生成匹配网格的 `occupied_pose.pt` |
| 5 | `FileNotFoundError: .../frames/-1.pt` | `trainers/train_macarons.py:155`（`setup_camera`）+ 配置 | **起始帧不足**：`setup_camera` 只采 `1+n_interpolation_steps` 帧，但深度模型要 `n_alpha=2`（预测）/`n_alpha_for_supervision=3`（监督）帧历史。配置里 `n_interpolation_steps=1` → 只 2 帧 → 第一次 loop 读 `-1.pt` | 冒烟配置把 `n_interpolation_steps` 提到 3（setup 采 4 帧）。⚠️ 这是**配置 bug**（作者默认训练配置的 1 与 n_alpha 不相容），正式训练也得改 |
| 6 | `RuntimeError: Parent directory .../0/surface does not exist` | `utility/macarons_utils.py:679`（`save_surface_scene_in_memory`） | 每条轨迹建了 `frames/occupancy/depths/imgs` 但**漏建 `surface/`**，轨迹末尾 `torch.save` 因父目录不存在失败 | 保存前 `os.makedirs(surface_dir_path, exist_ok=True)` |
| 7 | `_pickle.UnpicklingError: Weights only load failed ... numpy.core.multiarray.scalar`（**第 2 个 epoch 才崩**） | `train_macarons_run.py`（垫片，覆盖所有 `torch.load`） | **PyTorch 2.6 兼容**：代码写于 torch<2.6（`torch.load` 默认 `weights_only=False`），2.6 默认翻成 `True`，memory 的 surface/occupancy/depth 文件含 numpy 标量被拒。只有 epoch≥2 的 memory-replay 回放才加载这些文件，所以冒烟(1 epoch)没撞到 | 启动器加兼容垫片：`torch.load` 默认恢复 `weights_only=False`（我们自己生成的可信文件）。模块级设置，DDP 子进程 re-import 时也生效 |
| 8 | `ValueError: Cannot take a larger sample than population when 'replace=False'`（epoch 2 深度回放） | `utility/macarons_utils.py:5358`（`get_random_batch_for_depth_model`） | **条件写反**：`replace = n_sample <= len(scene_memory_paths)`。应是"样本数 > 场景数时才有放回"。场景少时（如冒烟 1 场景）`n_memory_samples=4` 触发无放回采 4 个 → 崩；场景多时反而有放回（采到重复场景，非预期） | 改成 `replace = n_sample > len(self.scene_memory_paths)` |

> #9（非 bug，配置 sizing）：`get_random_scene_for_scone_model` 有显式守卫 `traj_depth_nb + n_memory_scene_loops*n_poses_in_memory_scene_loops <= total_depth_nb`。深度图只在 `pose_i % remap_every_n_poses(=95)==0` 时保存——冒烟 5 步存了 **0 张**，必然触发。满训 100 步 > 95，第 95 步存约 280 张，守卫轻松通过。**不要改源码**，验证时用真实轨迹长度（100 步）即可。

> 说明：#1/#2/#3/#6 是代码 bug；#4 是**数据问题**（occupied_pose 网格旧）；#5 是**配置 bug**（`n_interpolation_steps` 与 `n_alpha` 不相容）；#7 是 **torch 2.6 兼容**（默认值变更）。
> 改动的作者源码文件：`macarons/utility/macarons_utils.py`（#1 删导入、#4 .get、#6 makedirs）、`macarons/trainers/train_macarons.py`（#2 守卫、#3 解包）。#5/#7 不改源码（#5 在配置、#7 在启动器垫片）。
> ⚠️ **关键教训**：bug 是逐 epoch 暴露的——#1~#6 在 epoch 1 内，#7 要到 epoch 2 才现（memory replay 只在 `current_epoch>0` 触发）。所以"冒烟 1 epoch 通过"≠"满训没问题"，至少要跑通 **2 个 epoch** 才覆盖回放路径。

---

## 3. 4 卡 DDP 正式训练（与论文一致）

**作者用 4 张卡训练**（证据一致）：所有配置 `WORLD_SIZE=4`；真正从头训练配置(`*_no_pretraining_*`)和全部 SCONE 预训练配置都 `ddp:true` + `CUDA_VISIBLE_DEVICES:"0,1,2,3"`；README 明写 `total_batch_size = GPU 总数`（MACARONS 训练=4，SCONE 预训练 total=12=3×4）；`idr_torch` 桩 `size=4`。

- **走 `ddp` 分支即可，不需要 idr_torch**：`idr_torch` 只在 `jz`(Jean Zay)分支用；通用多卡走 `ddp` 分支，`setup_device` 用 `dist.init_process_group(nccl)` + `cuda:ddp_rank`，`train_macarons_run.py` 用 `mp.spawn` 起 N 进程。已实测可用。
- **正式训练入口 = `train.sh`（已改为 4 卡 DDP）** → 跑 `configs/macarons/train_ddp_config.json`（`ddp:true, WORLD_SIZE:4, CUDA_VISIBLE_DEVICES:"0,1,2,3"`, 8 train 场景, total_batch_size=4=每卡1场景, epochs=105, 模型名 `jiahui_macarons_ddp`）。
- `-n 1`（只起 1 个 launcher 进程，内部 `mp.spawn` 起 4 个 DDP 进程占 4 卡）；**不要** `srun --ntasks=4`。
- 首次先 `--epochs 2` 验证（train.sh 里有注释行），确认全 8 场景+replay 在 4 卡上干净，再满训 105。

**DDP 实测**（2026-06-22, gpu114, 空闲 GPU 0/3）：
- 2 卡 × 1 epoch（`ddp_debug_config.json`）：✅ exit 0 —— mp.spawn 多进程 + NCCL + DistributedSampler 切场景(rank0=pantheon, rank1=redeemer) + DDP 模型包装 + 梯度同步全通，**无 `find_unused_parameters` 等 DDP 报错**。
- 2 卡 × 2 epoch（`ddp_validate_config.json`, 100 步）：进行中 —— 验证 DDP + memory replay 组合（结果见 §4）。
- 这两个 GPU 数无关的代码路径过了，4 卡只是 `WORLD_SIZE`/`CUDA_VISIBLE_DEVICES` 不同，逻辑一致。

---

## 4. 进度：✅ 冒烟测试通过（2026-06-22, gpu114 / L40S）

修掉上面 6 个 bug 后，`smoke_test_config.json`（pantheon, 5 pose, 1 epoch）**端到端跑通，退出码 0**：

- 加载预训练权重 ✓ → 优化器 ✓ → Epoch 1 ✓ → pantheon setup ✓（~21s）
- 5 个 pose 全部前向+反向：三个 loss（深度/占据/覆盖）正常，`grad_fn` 在、梯度回传、loss 下降（首 pose 0.45 → 后续 0.35…）
- 轨迹末尾保存 surface ✓ → 完成 epoch ✓ → 保存 checkpoint ✓
- 训练耗时 ~33 秒（`Done in 0.0091 hours`）

**产物**（都在 `weights/macarons/`，模型名 `smoke_trained`，**作者的 `trained_macarons.pth` 未被覆盖**，时间戳仍 2023-05-16）：
- `unvalidated_smoke_trained.pth` / `best_unval_smoke_trained.pth` / `epoch_0_smoke_trained.pth`（各 70M）
- `losses_data_smoke_trained.json`

**结论**：训练流水线本身可用；作者发布版的训练入口因上述 6 个问题无法直接运行，已逐一修复。下一步可按 §3 推进单卡全场景 1-epoch → 4 卡正式训练。
