# EXP6 V5 偏差与实现取舍记录

"冻结前"= 在 `FROZEN_PROTOCOL.md` v1.0（`ad8dc5c`，正式训练启动前）之前确定，属于对设计文档的实现取舍；"冻结后"= 冻结后发生，需单独说明影响。

| # | 时间 | 项 | 设计 / 任务书 | 实际 | 理由 / 影响 |
|---|---|---|---|---|---|
| D1 | 冻结前 | 深度尺度 | 保留度量尺度（t 或 log(1 + t / s0)）+ 归一化深度 | 模型单位 / 体素单位（S0 = 8 VS）+ t / tcap | Macarons++ 运行时单位非公制（OBJ × 10），V2 / V4 的全部几何本来在模型单位；MP3D 与 Macarons++ 经 k_M / k_I 统一到同一模型尺度 |
| D2 | 冻结前 | 隐藏维度 | 约 192，4 头 | 128，4 头 | 显存 / 速度审计后取 128（新模块 242,757 参数，单卡 4.5 GB） |
| D3 | 冻结前 | 局部检索门控 | δ ≤ ρ_l(τ) = clip(a·s_l + τ·tanθ, ρmin, ρmax)，确定性 top-k 或分层选择 | 对数深度分层：每层 3 个探针格中取 δ 最小者（δ ≤ 半格对角线）；F0 ≤ 24、F1 ≤ 12；粗 token = F1 池化到 2 × 2 × 2 | 设计允许"分层选择"；分层保证远处也有 token；锥形扩张由 F1 与粗 token 承担 |
| D4 | 冻结前 | Bridge 阶段的损失 | 设计 §9.3 推荐 Bridge 中新 Decoder 接受 L_bridge + L_surf；总损失式含 L_Flow | Bridge 也包含 L_FM（Flow / U-Net 冻结，FM 梯度只到新 decoder） | 总损失式 L = L_Flow + λ_surf L_surf + λ_bridge L_bridge 适用于全程；冻结的 Flow 让新条件直接对齐 Flow 的需要，不改变继承权重 |
| D5 | 冻结前 | L_bridge 归一化 | ‖normalize(H_new) − normalize(H_old)‖²，监控原始尺度 | mean(((H_new − H_old) / RMS(H_old))²)（标量 RMS） | 按通道标准化在 G0 中把近常数通道放大到 3.8e5；标量 RMS 同时匹配方向与原始幅值；H_new / H_old 均值、标准差、余弦每 50 步记录 |
| D6 | 冻结前 | L_surf 查询与均衡 | masked / balanced BCE，首 / 后表面分开报告 | 正例 + 首表面前负例 + 首表面后负例三组均衡；加入首表面后均匀查询与表面旁 Δ..2Δ 难负例 | G0 overfit 发现只用随机查询时学成"首表面后都是表面"（首表面后负例正确率 0.014）；修复后 0.76 |
| D7 | 冻结前 | 标签层数 | 网格多交点 t_1..t_m | 深度剥离 16 层、存 ≤ 8 个表面；16 层用满时监督区间截到第 16 层；第 8 与第 9 个表面之间截断；Δ = 8 VS | K = 8 时 liberty 层数用满 22 %；16 层时 ≤ 3.9 %；截断只缩小监督区间，不产生负例 |
| D8 | 冻结前 | V4-Continue 学习率 | 同 checkpoint、同数据、约 12k 步 | lr 1e-5（500 步 warmup，余弦到 0），与 V5 继承部分的 lr 相同 | V4 原调度已退火到 0，不能沿用；1e-5 与 V5 对继承权重的步长一致 |
| D9 | 冻结前 | G0 冒烟的代码版本 | 每门记录 Git SHA | 冒烟在 V5 文件首次提交前运行（manifest HEAD `cb74c40`，V5 文件未跟踪）；修复后的代码全部在 `ad8dc5c`，正式训练从 `ad8dc5c` 启动 | 冒烟后的代码改动只有 D5 / D6 与 NaN 截断、冒烟轨迹固定、belief 导出形状，均在 G0 报告中列出 |
| D10 | 冻结前 | MP3D 评估端 GT 相机 | — | 评估端多表面 GT 用 T1.1 标签相机（z clip 0.5 运行时单位），MP3D 模拟器的 FoV 相机 znear = 1 | 只影响 0.5–1 运行时单位（7–14 cm）内的表面；只用于 belief 指标 |
| D11 | 冻结后 | V5 训练 manifest 的 git dirty = True | 每门记录 Git SHA | V5 正式训练（19:42）启动时工作区有已跟踪文件的改动：`STATUS.md` 与 `scripts/label_quality.py`（图标题）；`git diff --stat ad8dc5c HEAD` 对 `v5_model.py`、`train_v5.py`、`train_arm.sh`、`configs/` 为空 | 训练代码与配置与 `ad8dc5c` 完全相同；不影响结果 |
| D12 | 冻结后 | 评估加速 | — | 用户要求加速评估；`gpu` 分区 QOS 每用户上限 8 GPU（已占满），空闲 L40S 仅 gpu118 的 2 张（不可用节点），无法再申请；改为每 GPU 2 个 G3 worker（claim 文件防重复） | 只影响墙钟与记录的每步时间（共享负载），不影响闭环结果（同起点同种子同动作规则） |
