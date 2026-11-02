# V5.2-M 参数审计（实测，`VM.param_table` + 逐子模块计数，2026-11-02）

全部参数可训练，且全部随机初始化（seed 0）。模型没有 buffer。

| 模块 | 子模块 | 参数量 |
|---|---|---:|
| 3D U-Net encoder（V5 Encoder3D 结构，ch 32/64/96，每级 2 个 ResBlock，L2 dilation 2，GN 8） | stem 3,488 · l0 110,976 · d1 55,360 · l1 443,136 · d2 165,984 · l2 996,480 · u1 276,544 · r1 221,568 · u0 82,976 · r0 55,488 | **2,412,000** |
| Ray-to-Memory Query（V5 射线解码器去掉表面 / 摘要 / adapter 头；d 128、4 头、24 + 12 token、6 频） | q_mlp 27,520 · f_proj 20,864 · geo_mlp 19,712 · geo_bias 100 · LN 512 · Wq/Wk/Wv/Wo 66,048 · FFN 66,176 | **201,060** |
| Ray map（Linear 128 → 64） | | **8,256** |
| 2D Spatial U-Net（输入 76 = 64 + 11 + valid；[96,192,384,384]，每级 2 个 ResBlock；输出 64） | inp 65,760 · down 11,551,104 · pool 1,742,496 · up_proj 2,157,216 · up 6,975,360 · out 55,552 | **22,547,488** |
| Depth-Aware 1D Decoder（宽 192，6 block，k 5，FFN×4） | ray_in 37,056 · bin_in 5,760 · blocks 1,786,752 · LN 384 · head 386 · known_bias 1 | **1,830,339** |
| **总计** | | **26,999,143** |

## 与预算对比
- **任务书原型**：约 26.4M，其中空间 U-Net 约 21.7M、深度 decoder 约 2.0M。
- **实测**：27.0M，比原型多 2.3%，在目标 25–27M 内。
- **35M 上限**：未超过，因此没有触发“先缩小空间 U-Net”的规则，结构未做调整。
- **对比旧 V5.2 pilot**：2,668,710（其中 HazardHead 13,953）。V5.2-M 约为其 10.1 倍。

## 显存 / 吞吐（G0 实测，L40S 46 GB）
| 设置 | 每 rank batch | s/step | 峰值显存 / rank |
|---|---|---:|---:|
| 1 卡 smoke | 8 views / 1 state | 0.458 | 6.7 GB |
| 4 rank（gpu115） | 16 views / 2 states | 0.999 | 6.9 GB |
| 8 rank 跨节点（gpu115 + gpu112，ib0） | 8 views / 1 state | 0.574 | 6.8 GB |

正式训练使用 8 rank × 8 views，全局 batch 为 64 views / 8 states。

显存设置：
- 射线潜变量（ray query）和 depth decoder 都开启 activation checkpointing；
- 2D U-Net 不开（`ckpt_spatial false`）。

显存余量很大，因此不需要 OOM 降级规则。
