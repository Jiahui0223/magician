# 00_STATUS_AND_RESUME — Y（2026-11-04 12:00）

- **状态**：训练在 **50000** 步按 control.json 有序停止（STOPPED_AT_TARGET）。控制器（gpu114，`controller_loop.sh Y`）仍在运行，每 60 秒更新 `work_dirs/exp6_v3/Y/STATUS.md`。
- **可用 checkpoint**：
  - `work_dirs/exp6_v3/Y/runs/main/ema_045000.pt`（sha `37da75b2bd95…`）
  - **`ema_050000.pt`（sha `64cea1d62e7b…`，入选）**
  - 滚动的完整训练状态 `ckpt_step0{48,49,50}000.pt` 与 `ckpt_last.pt`（step 50000）；sha 见 `checkpoint_manifest.json`
- **评估**：
  - Y45K + H0：Dev-5（25）；
  - Y50K + H0：All-14（70）；
  - YBEST-H3（Y50K + H3）：All-14（70）；
  - YD 组合（Y50K + D1）：Dev-5（25）。
- **续训到 55K**：把 `work_dirs/exp6_v3/Y/control.json` 的 stop_at 改为 55000，然后在 gpu114 上运行 `bash experiments/exp6_v3/Y/scripts/train_y.sh work_dirs/exp6_v3/Y/runs/main`（从 ckpt_last 的 50000 继续，学习率日程不变）。
- **停止控制器**：`touch work_dirs/exp6_v3/Y/STOP_CONTROLLER`。
