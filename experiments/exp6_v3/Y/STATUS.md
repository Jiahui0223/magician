# STATUS — EXP6 V3 / Y（2026-11-04T11:21:05Z，controller 自动更新）

- **阶段**：RUNNING
- **节点 / 作业**：gpu114 / 3461041 (gpu114)
- **已完成步数**：40001（ckpt_last 步数 40000；计划在 50000 停止，配置上限为 55000）
- **训练指标**：loss 0.09468207620753707，fm 0.06017085480241838，rec 0.027704195585101843，kl 0.34035129100084305，梯度范数 0.5766686797142029，学习率 4e-08
- **速度与显存**：3.1607866287231445 s/步，峰值显存 12.792995929718018 GB / rank；预计还需 8.78 h 到达 50000 步
- **里程碑 EMA**：—；最近的滚动 checkpoint：parent 40000；自动重启次数：1
- **续跑命令**：`bash experiments/exp6_v3/Y/scripts/train_y.sh work_dirs/exp6_v3/Y/runs/main`

| eval step | FM in_val | FM out_sc | fair ES in_val | fair ES out_sc |
|---|---|---|---|---|
