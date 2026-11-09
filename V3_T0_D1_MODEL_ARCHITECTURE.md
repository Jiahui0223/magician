# V3-T0@40K + D1：模型架构与完整 pipeline（含真实代码）

> 核心系统：**V3-T0@40K**（`work_dirs/exp6_v3/task0_vae/checkpoints/final.pt` = `work_dirs/checkpoints/exp6_v3/T0_vae/v3t0_ema40000.pt`，sha256 `7ed0ddd72bb91d1e8c6563f3192201c24a49fe862b49d2387ea8770915145ab6`，**没有交叉注意力**）+ **D1 规划器**（两步前瞻、λ = 0）。
> CVPR Table 1 记录为 `V3-T0-D1`（2026-11-08 由 Table 1 流水线补跑 70 条，数值见 §7.2）。它与 V3-T1 的关系、以及为什么把核心定为 T0，见 §0.1。
> 本文所有代码块都是从仓库里原样摘出的（文件路径和行号标在每段前面），不是示意代码；T0 与 T1 共用同一份代码，T0 只是 `xattn.enabled = false`，所以带 `xa` / `xattn` / `mem` 的分支在 T0 里都不执行。由 `V3_T1_D1_MODEL_ARCHITECTURE.md`（2026-11-07）派生，写于 2026-11-08，对应提交 `5f049ca`。架构图：`experiments/exp6_v3/figures/V3_T0_D1_architecture.{pdf,png}`。

---

## 目录

0. [一页总览](#0-一页总览)
1. [整体 pipeline](#1-整体-pipeline)
2. [数据与输入表示](#2-数据与输入表示)
3. [模型架构](#3-模型架构)
4. [训练算法](#4-训练算法)
5. [推理：采样与解码](#5-推理采样与解码)
6. [D1 规划器](#6-d1-规划器)
7. [关键数值](#7-关键数值)
8. [文件索引](#8-文件索引)

---

## 0. 一页总览

| 项 | 内容 |
|---|---|
| 任务 | 下一最佳视角（NBV）主动重建：每一步从模拟器给的合法候选位姿里选一个，使 100 步内的表面覆盖率尽快上升 |
| 模型输入 | 当前已观测的部分 TSDF（P1 地图）聚合成的 4× 体素网格 `[sdf, band, free] + valid`；每个候选视角的 7,296 条打分射线（64 × 114）及其 11 个投影通道 |
| 模型输出 | 对每条 UNKNOWN 射线预测 `(b, y)`：b = 是否会在未知区域命中表面，y = 命中位置在该射线 UNKNOWN 弧长上的相对深度；用整流流（rectified flow）生成，K = 4 个样本 |
| 网络 | 3D 卷积 VAE 编码器（μ：16 通道 @ 8× 体素）→ 潜变量射线编码器（沿射线采 32 点 → 64 维）→ 2D 条件整流流 U-Net（64/128/192/256，瓶颈自注意力）；**没有交叉注意力、没有场景记忆**：μ 只通过沿射线采样进入条件图 |
| 参数量 | 23,539,622（VAE 解码器 585,188 仅训练用；推理 22,954,434）；比 T1 少 191,364，即 T1 的交叉注意力块 |
| 训练 | 从零 40,000 步，4 × L40S，每步 64 个视角（室内 32 + 室外 32），AdamW 1e-4 余弦，bf16，EMA 0.999；`L = L_FM + L_recon + 0.02·KL`；15 h 54 min（与 T1 同 seed、同数据流、同配方，第 1 步损失逐位相同） |
| 规划器 D1 | 每个候选：K = 4 个样本解码成首次命中 → 新覆盖的体素集合（1 m 格、离观测点 ≥ 1 m）→ 对每个候选再展开一层设置网格邻居，取两步并集的样本均值的最大值作为分数 → argmax |
| 评估 | 冻结的 Macarons++ 模拟器（MAGICIAN 协议），14 个场景 × 5 个起点 × 100 步，官方覆盖 |

### 0.1 为什么核心是 T0，它和 T1 的关系

| | V3-T0（本文） | V3-T1 |
|---|---|---|
| 结构 | 3D VAE 编码器 → 沿射线采样 μ → 射线编码器 → 2D 整流流 U-Net（瓶颈自注意力） | 同左，**加**瓶颈处一个"相机射线 → 场景记忆"的交叉注意力块（`unet.xattn`，191,364 参数，输出投影零初始化） |
| 配置 | `experiments/exp6_v3/configs/model_v3.json`，`xattn.enabled = false` | 同一文件，只把 `xattn.enabled` 置 true |
| 训练 | 同数据、同 seed、同配方、同 40,000 步；公共模块初始权重逐位相同（哈希 `6a9d6071…`）；第 1 步损失逐位相同（3.674514578） | 同左 |
| 验证 fair ES @40k（室内 / 室外） | **6.37 / 6.44** | 6.42 / 6.48 |
| 权重 | `task0_vae/checkpoints/final.pt`，sha `7ed0ddd7…`，23,539,622 参数 | `task1_xattn/checkpoints/final.pt`，sha `639c9d3a…`，23,730,986 参数 |
| Table 1 行 | `V3-T0-D1`（2026-11-08 补跑，§7.2） | `V3-D1-H3-NORISK`（AUC@20 0.425 / AUC@100 0.765 / Final 0.910） |

把核心定为 T0 的依据是 Experiment XB 的 Task A（`experiments/exp6_v3/ablation_study/experiment_XB_bottleneck_audit/A_ATTENTION_AUDIT.md`，2026-11-08）：在 230 个固定状态（室内 90、室外 Dev 80、室外 Train 60）上对训练好的 T1 做六种推理期干预（交叉注意力输出置零、token 打乱、位置编码打乱、去掉 band / free token），**预测误差与候选选择全部与未干预完全相同**；注意力熵 5.84 对均匀熵 5.88，残差只占输入 RMS 的 6.9%。也就是说 T1 在推理时功能上等于一个 T0，交叉注意力没有贡献。T0 结构更简单、少 19 万参数、不需要构造场景记忆，因此作为核心系统；T1 保留为"加交叉注意力"的对照。两者在 D1 规划器下的闭环差别由 Table 1 的 `V3-T0-D1` 行给出（§7.2）。

---

## 1. 整体 pipeline

### 1.1 两条链

```
┌──────────────────────────── 离线：数据与训练 ────────────────────────────┐
│                                                                          │
│  模拟器观测记录 (svc_*.npz: 10% 采样点, 深度图, 位姿)                      │
│        │                                                                 │
│        ▼  m_p1.P1Fusion / p1_full.P1FusionFull (D100)                    │
│  部分 TSDF 地图 (细体素 VS = 0.011, BAND / EMPTY / UNKNOWN)                │
│        │                                   │                             │
│        ▼ partial_tsdf.aggregate            ▼ geometry.project / encode_labels (E2 构建器)
│  4× 网格缓存 tsdf4/<traj>.npz              视角分片 out_views_d100/<traj>.npz  │
│  [sdf, band, free, dom]                    (cls, tk, dlo, dhi, lu, nseg, b, y, cat)
│        └──────────────┬─────────────────────┘                            │
│                       ▼ seen2scene/v3/data.py (V3Views, build_cond)       │
│               trainer.py (torchrun × 4)                                  │
│        V3Model: VAE → LatentRayEncoder → FlowUNet (无 CrossAttn)         │
│               L = L_FM + L_recon + β·KL                                  │
│                       ▼                                                  │
│        ema_040000.pt ──选模──▶ checkpoints/final.pt                       │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────── 在线：闭环评估 ──────────────────────────────┐
│  模拟器进程 (magician env)                 规划器子进程 (seen2scene env)    │
│  t24_out_sim.py → v2_sim.run              v3_out_planner.py → v2_planner  │
│    observe(t) ──svc 记录──────────────▶   P1Fusion.integrate_record       │
│    legal() 候选 (6 移动 + 4 旋转,          Geometry.from_fusion → coarse() │
│            未访问优先, GT 碰撞检测)        R10: encode + K 噪声 (t ∈ {0,10,…})│
│                                           score_state: cond → sample →    │
│                                             decode → value_cov (每样本)    │
│                                           d_policy.lookahead: 两步并集     │
│    move(keys[pick]) ◀──pick────────────   m_score.choose (argmax)         │
│    coverage ← MAGICIAN 官方覆盖                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.2 训练入口调用链

```
experiments/exp6_v3/shared/train_arm.sh t0
  └─ torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/task0_vae/scripts/train.py --out <run dir>
       └─ seen2scene/v3/trainer.py : main()          ← 真正的训练脚本
            ├─ configs: experiments/exp6_v3/configs/train_v3.json, model_v3.json
            ├─ seen2scene/v3/data.py   : V3Views (分片 + 4× 缓存), build_mixed
            ├─ seen2scene/v3/model.py  : V3Model
            ├─ seen2scene/v3/losses.py : fm_terms / recon_terms / kl_terms
            └─ seen2scene/v3/eval_dev.py : 每 5,000 步验证 → ema_<step>.pt
```

文件：`experiments/exp6_v3/task0_vae/scripts/train.py`（全文）

```python
"""EXP6 V3 Task 0 -- scratch training of V3-VAE (T0): the shared trainer with --arm t0.
torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/task0_vae/scripts/train.py --out <run dir> [--smoke overfit|short|profile]
"""
import sys

from seen2scene.v3 import trainer

if __name__ == "__main__":
    sys.argv[1:1] = ["--arm", "t0"]
    trainer.main()
```

文件：`experiments/exp6_v3/shared/train_arm.sh`（全文）

```bash
#!/bin/bash
# train_arm.sh t0|t1 : formal V3 run of one arm on the 4 GPUs of the node it runs on (torchrun, 4 ranks)
# Tracked copy of the launcher; the running copy is work_dirs/exp6_v3/train_arm.sh (identical content).
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
S2E=/bsuscratch/jiahuizhang/envs/seen2scene
W=$R/work_dirs/exp6_v3
arm=$1; T=$([ $arm = t0 ] && echo task0_vae || echo task1_xattn)
cd $R
export LD_PRELOAD=$S2E/lib/libtbb.so LD_LIBRARY_PATH=$S2E/lib PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=6
echo "start $arm $(hostname -s) GPUs ${CUDA_VISIBLE_DEVICES:-0,1,2,3} job ${SLURM_JOB_ID:-interactive} $(date '+%F %T')" >> $W/$T/logs/train.log
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} $S2E/bin/torchrun --standalone --nproc_per_node 4 experiments/exp6_v3/$T/scripts/train.py --out $W/$T/runs/main >> $W/$T/logs/train.log 2>&1
echo "EXIT $? $arm $(hostname -s) $(date '+%F %T')" >> $W/$T/logs/train.log
```

### 1.3 闭环入口调用链

```
experiments/exp6_v3/shared/sim_out_v3.py --v3-ckpt final.pt --group V2M --scene S --start K --root DIR   (magician env)
  └─ experiments/exp5_future_view/task2/scripts/t24_out_sim.py (冻结)
       └─ experiments/exp5_future_view/task1/scripts/v2_sim.py : run()   逐步检查点、逐位续跑、覆盖曲线
            └─ 子进程 (seen2scene env) experiments/exp6_v3/shared/v3_out_planner.py
                 ├─ p1_full.patch()  T3_P1_MODE=b2 : 编码器输入用 D100 地图，DistP1 打分用 10% 稀疏地图
                 ├─ planner_v3.install() : PC.Core → V3Core (V3Predictor)
                 ├─ d_policy.install("d1") : PC.score_state → 两步前瞻（见 §6.4）
                 └─ task2/scripts/t24_v2_out_planner.py → task1/scripts/v2_planner.py : main()
experiments/exp5_future_view/task2/scripts/t25_out_cov.py   官方覆盖重放（必须与存档曲线逐位一致）
```

---

## 2. 数据与输入表示

### 2.1 坐标系、体素与打分射线

- **模型坐标系**：z-up，单位 = 运行时单位 / `k_M`，细体素 `VS = 0.011`；截断带 `TRUNC = 3 VS`。
- **打分射线**：MAGICIAN 传感器 456 × 256、fov 60°，取 4 × 4 像素块的中心 → 64 × 114 = 7,296 条射线；每条射线带一个亚像素抖动 `jitter`（按 scene / traj / t / a 的哈希确定）。网络里射线图按 `[64, 114]` 排列并零填充到 `[64, 128]`。

文件：`seen2scene/futureview/rays.py` 58–81

```python
def model_rays_torch(R, T, jit, origin_z, k_M, sensor_range=70.0):
    """Batched torch version of m_common.pixel_rays + SceneFrame.runtime_to_model + SceneCaster.rays for V views:
    R [V, 3, 3], T [V, 3], jit [V, 2] (float64 tensors), origin_z [3], k_M float -> o [V, 3], d [V, 7296, 3] (model
    frame, unit), tcap [V, 7296] (model units), dz [V, 7296] (z-depth per runtime ray length). float64."""
    import torch
    dev = R.device
    j = torch.arange(GW, device=dev, dtype=torch.float64)
    i = torch.arange(GH, device=dev, dtype=torch.float64)
    v, u = torch.meshgrid(BLOCK * i + (BLOCK - 1) / 2.0, BLOCK * j + (BLOCK - 1) / 2.0, indexing="ij")
    u = u.reshape(1, -1) + jit[:, 0:1]
    v = v.reshape(1, -1) + jit[:, 1:2]
    nx = W / M_SHORT - 2.0 * u / (M_SHORT - 1)
    ny = H / M_SHORT - 2.0 * v / (M_SHORT - 1)
    view = torch.stack([nx / S_PROJ, ny / S_PROJ, torch.ones_like(nx)], -1)          # z = 1
    p1 = (view - T[:, None]) @ R.transpose(1, 2)
    c = -(T[:, None] @ R.transpose(1, 2))[:, 0]
    dd = p1 - c[:, None]
    n = dd.norm(dim=-1, keepdim=True)
    d_y = dd / n
    dz = 1.0 / n[..., 0]
    zu = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=torch.float64, device=dev)
    o = (c @ zu.T - origin_z) / k_M
    d = d_y @ zu.T
    return o, d, sensor_range / dz / k_M, dz
```

`tcap` 是每条射线在 70 个运行时单位量程下的最大模型长度；`dz` 把射线长度换算成 z 深度。

### 2.2 P1 融合：观测 → 部分 TSDF

每一步的观测记录含相机位姿、深度图 `zbuf`/`mask`，以及模拟器按 MAGICIAN 规则抽取的 10% 表面采样点。P1 把它们融合到场景网格：

- **BAND 层**（`integrate_band`）：从相机中心向每个采样点做 DDA 体素遍历，在距离点 `±TRUNC` 的体素上累加带符号距离（定点整数）和计数；
- **EMPTY 层**（`integrate_free`）：对全部有效像素，从近裁剪面沿射线到 `z − trunc` 的样本点标记为空闲（free carving）；
- **输入体**：BAND 优先，其次 EMPTY（+trunc），其余 UNKNOWN（−trunc）。

文件：`experiments/exp3b_macaronspp_transfer/m_p1.py` 61–169（`P1Fusion` 的核心方法）

```python
    def integrate_band(self, points_y_up: np.ndarray, center_y_up: np.ndarray, chunk: int = 1 << 15):
        if len(points_y_up) == 0:
            return
        P = self._to_model(np.asarray(points_y_up, np.float64))
        o = self._to_model(np.asarray(center_y_up, np.float64).reshape(1, 3))[0]
        for s in range(0, len(P), chunk):
            self._band_rays(o, P[s:s + chunk])

    def _band_rays(self, o: torch.Tensor, p: torch.Tensor):
        vs, tr = MC.VS, MC.TRUNC
        dims = torch.as_tensor(self.dims, device=self.dev, dtype=torch.float64)
        d = p - o
        L = d.norm(dim=1)
        d = d / L[:, None]
        t_a = (L - tr - np.sqrt(3) * vs).clamp(min=0.0)
        t_b = L + tr
        c0 = o / vs                                   # cell coordinates (cell a spans [a, a+1))
        dc = d / vs
        safe = torch.where(dc.abs() < 1e-12, torch.full_like(dc, 1e-12), dc)
        inv = 1.0 / safe
        ta, tb = (0.0 - c0) * inv, (dims - c0) * inv
        t_in = torch.maximum(torch.minimum(ta, tb).amax(1), t_a)
        t_out = torch.minimum(torch.maximum(ta, tb).amin(1), t_b)
        alive = t_out > t_in
        if not alive.any():
            return
        idx = torch.nonzero(alive)[:, 0]
        p, d, dc, safe, t_in, t_out = p[idx], d[idx], dc[idx], safe[idx], t_in[idx], t_out[idx]
        pos = c0[None] + dc * (t_in[:, None] + 1e-9)
        dims_l = torch.as_tensor(self.dims, device=self.dev)
        cell = torch.minimum(torch.floor(pos).long().clamp(min=0), dims_l - 1)
        step = torch.sign(dc).long()
        step = torch.where(step == 0, torch.ones_like(step), step)
        nxt = cell + (step > 0).long()
        t_max = (nxt.double() - c0[None]) / safe
        t_max = torch.where(dc.abs() < 1e-12, torch.full_like(t_max, float("inf")), t_max)
        t_delta = (1.0 / dc.abs()).clamp(max=1e30)
        active = torch.ones(len(idx), dtype=torch.bool, device=self.dev)
        while active.any():
            a = torch.nonzero(active)[:, 0]
            c = cell[a]
            centre = (c.double() + 0.5) * vs
            vo = centre - o[None]
            pv = p[a] - centre
            dist = pv.norm(dim=1)
            sdf = torch.where((vo * pv).sum(1) >= 0, dist, -dist)
            ok = (sdf.abs() < tr) & self._in_domain(c)
            if ok.any():
                fl = self._flat(c[ok])
                self.sum.index_add_(0, fl, torch.round(sdf[ok] * FIXED).long())
                self.cnt.index_add_(0, fl, torch.ones_like(fl, dtype=torch.int32))
            tm = t_max[a]
            ax = torch.argmin(tm, dim=1)
            t_next = tm.gather(1, ax[:, None])[:, 0]
            cell[a, ax] += step[a, ax]
            t_max[a, ax] += t_delta[a, ax]
            inside = ((cell[a] >= 0) & (cell[a] < dims_l)).all(1)
            active[a] = inside & (t_next <= t_out[a])

    # ------------------------------------------------------------------ EMPTY layer
    @torch.no_grad()
    def integrate_free(self, zbuf: np.ndarray, mask: np.ndarray, R, T, chunk: int = 2048):
        u, v = MC.full_pixel_grid()
        d_y, dz, c = MC.pixel_rays(R, T, u, v)
        z = np.clip(np.asarray(zbuf, np.float64).reshape(-1), MC.NEAR_CLIP, 750.0)
        hit = np.asarray(mask, bool).reshape(-1)
        t0 = MC.NEAR_CLIP / dz                                         # runtime ray length at z = 0.5
        t1 = np.where(hit, np.minimum(z / dz - self.trunc_rt, MC.SENSOR_RANGE), MC.SENSOR_RANGE)
        t1 = np.minimum(t1, MC.SENSOR_RANGE * (1.0 - 1e-12))           # strictly Euclidean < 70
        keep = t1 > t0
        if not keep.any():
            return
        k = self.f.k_M
        o = self._to_model(c.reshape(1, 3))[0]
        D = torch.as_tensor(MC.to_z_up(d_y[keep]), dtype=torch.float64, device=self.dev)   # unit (rotation)
        T0 = torch.as_tensor(t0[keep] / k, dtype=torch.float64, device=self.dev)           # model units
        T1 = torch.as_tensor(t1[keep] / k, dtype=torch.float64, device=self.dev)
        h = 0.5 * MC.VS
        for s in range(0, len(D), chunk):
            d_, a_, b_ = D[s:s + chunk], T0[s:s + chunk], T1[s:s + chunk]
            m = int(torch.ceil((b_ - a_).max() / h).item()) + 1
            ts = a_[:, None] + h * torch.arange(m, device=self.dev, dtype=torch.float64)[None]
            ok = ts <= b_[:, None]
            pts = o[None, None] + d_[:, None] * ts[..., None]
            ijk = torch.floor(pts[ok] / MC.VS).long()
            ijk = ijk[self._in_domain(ijk)]
            if len(ijk):
                self.empty[self._flat(ijk)] = True

    # ------------------------------------------------------------------ frames and outputs
    def integrate_record(self, rec: Dict):
        """One whitelisted service record: BAND from its 10% samples, EMPTY from its full depth."""
        self.integrate_band(rec["sample_points"], rec["cam_center"])
        self.integrate_free(rec["zbuf"], rec["mask"], rec["R"], rec["T"])
        self.n_frames += 1

    def band_value(self) -> torch.Tensor:
        return torch.where(self.cnt > 0, self.sum.double() / FIXED / self.cnt.clamp(min=1).double(),
                           torch.zeros(self.sum.shape, dtype=torch.float64, device=self.dev))

    def input_volume(self) -> torch.Tensor:
        """Dense float32 TSDF in the make_batch convention: BAND value / +trunc (EMPTY) / -trunc (UNKNOWN)."""
        has_band = self.cnt > 0
        v = torch.full(self.sum.shape, -MC.TRUNC, dtype=torch.float64, device=self.dev)
        v = torch.where(self.empty, torch.full_like(v, MC.TRUNC), v)
        v = torch.where(has_band, self.band_value().clamp(-MC.TRUNC, MC.TRUNC), v)
        return v.float().reshape(self.dims)
```

**D100 变体**（V3 的编码器输入用它）：BAND 层改为用全部有效像素反投影的点，其余不变。

文件：`experiments/exp5_future_view/task3_full_depth_p1/scripts/p1_full.py` 1–38

```python
"""EXP5 Task 3 / Experiment 1 -- FullDepth-P1: the frozen P1 fusion (experiments/exp3b_macaronspp_transfer/m_p1.py) with
the BAND layer fed by ALL valid depth pixels of each observation instead of the simulator's 10 % sample.

Minimal change: P1FusionFull.integrate_record = the frozen integrate_record except that the BAND points are the stored
10 % sample points (bitwise, as in B0) PLUS the back-projection of every other valid pixel of the same record.
  valid pixel   = mask & clamp(zbuf, 0.5, 750) < 70 -- the exact pixel set the simulator samples its 10 % from
                  (mp3d_sim.Sensor.sample / MAGICIAN compute_partial_point_cloud after apply_perfect_depth_simple)
  back-projection = m_common.pixel_rays (the camera model of integrate_free): p = c + d * z / dz
Everything else is inherited unchanged: integrate_band (same voxel traversal, truncation, signed distance, fixed-point
sums and counts), integrate_free (EMPTY carving from the full depth), input_volume / structure (BAND over EMPTY),
frames, domains. No interpolation, no point beyond a measured ray end, no future information.
patch(): replace m_p1.P1Fusion in the current process (planner children only; mode from env T3_P1_MODE=full).
"""
...
def full_points(rec):
    """Points [N, 3] (y-up runtime, float32) of all valid pixels of a service record: the stored 10 % sample points first
    (bitwise), then the back-projection of the other valid pixels."""
    depth = np.clip(np.asarray(rec["zbuf"], np.float64), 0.5, 750.0).reshape(-1)
    valid = np.asarray(rec["mask"], bool).reshape(-1) & (depth < MC.SENSOR_RANGE)
    sf = np.asarray(rec["sample_pixel_flat"], np.int64)
    rest = valid.copy()
    rest[sf] = False
    idx = np.flatnonzero(rest)
    u, v = idx % MC.W, idx // MC.W
    d_y, dz, c = MC.pixel_rays(rec["R"], rec["T"], u, v)
    p = c[None] + d_y * (depth[idx] / dz)[:, None]
    return np.concatenate([np.asarray(rec["sample_points"], np.float32), p.astype(np.float32)]), int(valid.sum())
```

部署时的模式是 **B2**：编码器输入看 D100 地图（稠密），但 DistP1 / `value_cov` 的“已观测”判定用原始 10% 稀疏地图（与训练分片一致）。

### 2.3 4× 聚合：细体素 P1 → VAE 输入网格

把细体素融合结果以 4 × 4 × 4 的块聚合成 4× 网格：每块的 `band` 比例、`free` 比例、`dom` 比例和带权 sdf 均值。网格在高端填充到 8 的倍数，因此 stride-2 的潜变量覆盖同样范围。

文件：`seen2scene/v3/partial_tsdf.py` 51–81、123–131、157–159

```python
def aggregate(sum_, cnt, empty, dims, dom1d, factor=FACTOR, trunc=TRUNC, fixed=FIXED):
    """Fused arrays (flat, on one device) -> dict of 4x grids (sdf int8, band / free / dom uint8), torch on the same
    device. dom1d: three bool arrays / tensors (separable domain of the fine grid). Memory-lean: one pooled quantity
    at a time; the indicator pools run in half precision (k / factor^3 is exact in half)."""
    dev = sum_.device
    dims = [int(x) for x in dims]
    d4 = padded_dims(dims, factor)
    pad = [0, d4[2] * factor - dims[2], 0, d4[1] * factor - dims[1], 0, d4[0] * factor - dims[0]]
    dom = [torch.as_tensor(np.asarray(m, bool), device=dev) for m in dom1d]
    dom3 = dom[0][:, None, None] & dom[1][None, :, None] & dom[2][None, None, :]
    has_band = (cnt > 0).reshape(dims)

    ind = torch.half if dev.type == "cuda" else torch.float32          # CPU avg_pool3d has no half kernel

    def P(x):
        return _pool(F.pad(x, pad) if any(pad) else x, factor).float()
    n_dom = P(dom3.to(ind))
    n_band = P((has_band & dom3).to(ind))
    n_free = P((empty.reshape(dims) & ~has_band & dom3).to(ind))
    val = torch.where(has_band, sum_.reshape(dims).float() / fixed, torch.zeros(dims, dtype=torch.float32, device=dev))
    val = (val / cnt.reshape(dims).clamp(min=1).float()).clamp(-trunc, trunc) / trunc
    val = torch.where(has_band & dom3, val, torch.zeros_like(val))
    s_sum = P(val)                                    # mean of val over the block = sum / factor^3
    del val, dom3, has_band
    safe = n_dom.clamp(min=1.0 / factor ** 3)
    out = {"sdf": torch.round((s_sum / n_band.clamp(min=1.0 / factor ** 3)) * SDF_Q).clamp(-SDF_Q, SDF_Q).to(torch.int8),
           "band": torch.round(n_band / safe * 255.0).to(torch.uint8),
           "free": torch.round(n_free / safe * 255.0).to(torch.uint8),
           "dom": torch.round(n_dom * 255.0).to(torch.uint8)}
    out["sdf"] = torch.where(n_band > 0, out["sdf"], torch.zeros_like(out["sdf"]))
    return out


def unpack(grid, device, dtype=torch.float32):
    """numpy / torch grids -> (x [1, 3, X4, Y4, Z4] in [-1, 1] / [0, 1], valid [1, 1, X4, Y4, Z4] in {0, 1})."""
    t = lambda k: torch.as_tensor(np.asarray(grid[k]) if not torch.is_tensor(grid[k]) else grid[k], device=device)  # noqa: E731
    sdf = t("sdf").to(dtype) / SDF_Q
    band = t("band").to(dtype) / 255.0
    free = t("free").to(dtype) / 255.0
    valid = (t("dom") > 0).to(dtype)
    return torch.stack([sdf, band, free])[None] * valid[None, None], valid[None, None]


def extent(shape4, factor=FACTOR):
    """Model-unit extent [3] of a 4x grid (its padded fine grid)."""
    return torch.tensor([s * factor * VS for s in shape4], dtype=torch.float32)
```

训练缓存（`shared/build_tsdf_cache.py`）和部署（`V3Predictor.encode_fusion`）调用的是同一个 `aggregate`，所以训练与部署的输入按构造相同。最大场景 dunnottar 的 4× 网格是 278 × 398 × 64。

### 2.4 候选视角的投影通道（11 通道）

对一个候选位姿，沿每条打分射线在当前 P1 的 `struct`（EMPTY / UNKNOWN / BAND）上以 `H_STEP = 0.5 VS` 行进，只看 `t_known` 之前（已知表面之前）的部分，统计 UNKNOWN 样本数、首末 UNKNOWN 位置和段数，并给射线分类。

文件：`seen2scene/futureview/geometry.py` 50–100、169–191

```python
def unknown_samples(o, d, tmin, tmax, valid, M, t_known, struct, domain):
    """Boolean [R, M] of UNKNOWN samples, and ts [R, M]. struct int8 [X, Y, Z] (P1 structure), domain bool [X, Y, Z]."""
    dims = struct.shape
    ts = sample_ts(tmin, M)
    alive = (ts <= tmax[:, None]) & valid[:, None] & (ts < t_known[:, None])
    p = o[None, None] + d[:, None] * ts[..., None]
    ijk = torch.floor(p / VS).long()
    inside = alive & (ijk >= 0).all(-1) & (ijk[..., 0] < dims[0]) & (ijk[..., 1] < dims[1]) & (ijk[..., 2] < dims[2])
    unk = torch.zeros_like(inside)
    if inside.any():
        q = ijk[inside]
        flat = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
        unk[inside] = (struct.reshape(-1)[flat] == UNKNOWN) & domain.reshape(-1)[flat]
    return unk, ts


def ray_stats(unk, ts):
    """n_u, d_lo, d_hi (ts of the first / last UNKNOWN sample; inf if none), n_seg."""
    n_u = unk.sum(1)
    any_ = n_u > 0
    first = torch.argmax(unk.int(), dim=1)
    last = unk.shape[1] - 1 - torch.argmax(unk.flip(1).int(), dim=1)
    inf = torch.full_like(ts[:, 0], float("inf"))
    d_lo = torch.where(any_, ts.gather(1, first[:, None])[:, 0], inf)
    d_hi = torch.where(any_, ts.gather(1, last[:, None])[:, 0], inf)
    n_seg = (unk[:, 1:] & ~unk[:, :-1]).sum(1) + unk[:, 0].long()
    return n_u, d_lo, d_hi, n_seg


def ray_class(n_u, t_known):
    cls = torch.full_like(n_u, CLS_EMPTY)
    cls = torch.where(torch.isfinite(t_known), torch.full_like(cls, CLS_KNOWN), cls)
    return torch.where(n_u > 0, torch.full_like(cls, CLS_UNKNOWN), cls)


def project(o, d, tcap, ext, t_known, struct, domain, chunk=2048):
    """All projection quantities of one view. o [3], d [R, 3], tcap [R] (caster dtype, model units). Returns a dict of
    per-ray tensors + the march range; unknown samples are recomputed per chunk (not kept)."""
    tmin, tmax, valid, M = march_range(o, d, tcap, ext)
    R = d.shape[0]
    n_u = torch.zeros(R, dtype=torch.long, device=d.device)
    d_lo = torch.full((R,), float("inf"), dtype=d.dtype, device=d.device)
    d_hi, n_seg = d_lo.clone(), torch.zeros_like(n_u)
    for s in range(0, R, chunk):
        sl = slice(s, min(R, s + chunk))
        unk, ts = unknown_samples(o, d[sl], tmin[sl], tmax[sl], valid[sl], M, t_known[sl], struct, domain)
        n_u[sl], d_lo[sl], d_hi[sl], n_seg[sl] = ray_stats(unk, ts)
    return {"tmin": tmin, "tmax": tmax, "valid": valid, "M": M, "t_known": t_known, "n_u": n_u, "d_lo": d_lo,
            "d_hi": d_hi, "n_seg": n_seg, "cls": ray_class(n_u, t_known)}


def stats_fp16(pr, tcap):
    """Stored projection statistics (protocol §2.3): t_known, d_lo, d_hi, L_u relative to t_cap, as float16 (the exact
    values the training data holds; the planner rounds the same way, so train and deploy inputs are identical)."""
    one = torch.ones_like(tcap)

    def rel(x):
        return torch.where(torch.isfinite(x), (x / tcap).clamp(0.0, 1.0), one)
    lu = (pr["n_u"].to(tcap.dtype) * H_STEP / tcap).clamp(0.0, 1.0)
    return torch.stack([rel(pr["t_known"]), rel(pr["d_lo"]), rel(pr["d_hi"]), lu]).half()


def channels_from_stats(cls, st16, nseg, d_z, cam_z, tcap):
    """Input channels [..., 11, R] (architecture §2) from class [..., R], float16 stats [..., 4, R], n_seg [..., R],
    ray direction z [..., R], camera model z [...] and t_cap [..., R]; float32."""
    oh = torch.nn.functional.one_hot(cls.long(), 3).movedim(-1, -2).float()
    rest = torch.stack([nseg.clamp(max=3).float() / 3.0, d_z.float(),
                        (cam_z.float() / PATCH_H)[..., None].expand_as(tcap), torch.log(tcap.float())], -2)
    return torch.cat([oh, st16.float(), rest], -2)


def channels(pr, tcap, d, o):
    """Input channels [11, R] of one view, through the same float16 statistics as the training data."""
    return channels_from_stats(pr["cls"], stats_fp16(pr, tcap), pr["n_seg"], d[:, 2], o[2], tcap)
```

11 个通道：

| # | 通道 | 含义 |
|---|---|---|
| 0–2 | one-hot(cls) | 射线类别 EMPTY / KNOWN / UNKNOWN |
| 3 | t_known / tcap | 已知表面（BAND）的位置，无则 1 |
| 4 | d_lo / tcap | 第一个 UNKNOWN 样本的位置 |
| 5 | d_hi / tcap | 最后一个 UNKNOWN 样本的位置 |
| 6 | L_u | UNKNOWN 总弧长 / tcap |
| 7 | n_seg / 3 | UNKNOWN 段数（截断到 3） |
| 8 | d_z | 射线方向的 z 分量 |
| 9 | cam_z / PATCH_H | 相机高度 |
| 10 | log tcap | 射线量程 |

### 2.5 标签 (b, y)：GT 首次命中在 UNKNOWN 弧长上的位置

训练标签由 E2 构建器用 GT 网格沿同一组射线计算（`tstar` = 首个 GT 交点）。规则：命中落在 UNKNOWN 样本上 → `b = +1`，`y = 2 s − 1`，其中 `s` 是该命中在这条射线 UNKNOWN 样本序列里的相对序号；GT 在 `t_known` 之后或没有命中 → `b = −1`，`y = 1`；其余（被遮挡、近裁剪、命中在非 UNKNOWN 体素）→ `b = 0`，不监督。

文件：`seen2scene/futureview/geometry.py` 101–167

```python
def encode_labels(o, d, pr, tstar, dz_rt, k_M, struct, domain, chunk=2048):
    """Label category, b in {-1, 0, +1} (0 = no target) and y of every ray (protocol §2.4).
    tstar: first GT intersection along the ray in model units (inf = none); dz_rt: z-depth per runtime ray length."""
    tmin, tmax, valid, M = pr["tmin"], pr["tmax"], pr["valid"], pr["M"]
    R, dims = d.shape[0], struct.shape
    cat = torch.full((R,), CAT["NOHIT"], dtype=torch.long, device=d.device)
    s_val = torch.ones(R, dtype=d.dtype, device=d.device)
    fin = torch.isfinite(tstar)
    blocked = fin & (tstar < tmin - VS)
    near = fin & ~blocked & (tstar * k_M * dz_rt.to(tstar.dtype) < NEAR_CLIP)
    nohit = ~fin | (tstar > tmax)
    known = ~blocked & ~near & ~nohit & (tstar >= pr["t_known"] - VS)
    rest = ~blocked & ~near & ~nohit & ~known
    hit = torch.zeros_like(rest)
    for s in range(0, R, chunk):
        sl = slice(s, min(R, s + chunk))
        r_idx = torch.nonzero(rest[sl])[:, 0]
        if len(r_idx) == 0:
            continue
        unk, ts = unknown_samples(o, d[sl], tmin[sl], tmax[sl], valid[sl], M, pr["t_known"][sl], struct, domain)
        unk, ts, tq = unk[r_idx], ts[r_idx], tstar[sl][r_idx]
        j0 = torch.round((tq - ts[:, 0]) / H_STEP).long()
        best_j = torch.full_like(j0, -1)
        best_e = torch.full_like(tq, float("inf"))
        for off in (0, -1, 1, -2, 2, -3, 3):          # nearest UNKNOWN sample within one voxel (ties -> earlier)
            j = j0 + off
            ok = (j >= 0) & (j < M)
            jc = j.clamp(0, M - 1)
            e = (ts.gather(1, jc[:, None])[:, 0] - tq).abs()
            good = ok & unk.gather(1, jc[:, None])[:, 0] & (e <= VS)
            better = good & ((e < best_e) | ((e == best_e) & (j < best_j)))
            best_e = torch.where(better, e, best_e)
            best_j = torch.where(better, j, best_j)
        found = best_j >= 0
        cs = torch.cumsum(unk.long(), dim=1)
        rank = cs.gather(1, best_j.clamp(min=0)[:, None])[:, 0] - 1
        n_u = unk.sum(1)
        g = r_idx + s
        hit[g[found]] = True
        s_val[g[found]] = ((rank[found].to(d.dtype) + 0.5) / n_u[found].to(d.dtype))
    infeas = rest & ~hit
    if infeas.any():
        q = torch.nonzero(infeas)[:, 0]
        p = o[None] + d[q] * tstar[q][:, None]
        ijk = torch.floor(p / VS).long()
        inside = (ijk >= 0).all(-1) & (ijk[:, 0] < dims[0]) & (ijk[:, 1] < dims[1]) & (ijk[:, 2] < dims[2])
        sub = torch.full_like(q, CAT["INF_OOD"])
        if inside.any():
            qi = ijk[inside]
            flat = (qi[:, 0] * dims[1] + qi[:, 1]) * dims[2] + qi[:, 2]
            st = struct.reshape(-1)[flat].long()
            dm = domain.reshape(-1)[flat]
            c = torch.where(st == EMPTY, CAT["INF_EMPTY"], torch.where(st == BAND, CAT["INF_BAND"], CAT["INF_OTHER"]))
            sub[inside] = torch.where(dm, c, torch.full_like(c, CAT["INF_OOD"]))
        cat[q] = sub
    cat[known] = CAT["KNOWN"]
    cat[hit] = CAT["HIT"]
    cat[near] = CAT["NEAR"]
    cat[blocked] = CAT["BLOCKED"]
    cat[~valid] = CAT["OUT"]
    m_unk = pr["n_u"] > 0
    b = torch.zeros(R, dtype=torch.int8, device=d.device)
    b[m_unk & (cat == CAT["HIT"])] = 1
    b[m_unk & ((cat == CAT["NOHIT"]) | (cat == CAT["KNOWN"]))] = -1
    y = torch.where(cat == CAT["HIT"], 2.0 * s_val - 1.0, torch.ones_like(s_val))
    return {"cat": cat, "b": b, "y": y}
```

因此网络的目标图 `x1 ∈ [V, 2, 64, 128]`：通道 0 是 `b`，通道 1 是 `y`；监督掩码 `lmask = UNKNOWN ∧ b ≠ 0`。

### 2.6 训练样本的组装

数据集：

| 域 | 训练 | 选模 / 验证 | 分片 | 4× 缓存 |
|---|---|---|---|---|
| 室内 MP3D | 228 条轨迹（66,803 个候选视角行） | VALSEL 9 条（10,473 行） | `work_dirs/exp5_future_view/task2/mp3d/...` | `work_dirs/exp6_v3/shared_cache/tsdf4/mp3d/` |
| 室外 Macarons++ | Train-6 去 S-c 138 条（25,225 行） | S-c 42 条（31,027 行） | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/out_views_d100/<scene>/<traj>.npz` | `.../tsdf4/outdoor/` |

一个样本 = (轨迹, 时刻 t, 陈旧度 Δ, 候选行)。编码器输入是状态 `t − Δ` 的网格（Δ ∈ [1, 9]，概率 0.5；MP3D 不跨 10 步的刷新块），射线通道和标签是状态 t 的；这模拟部署时 R10 刷新（编码只在 t ∈ {0, 10, …} 做）。

文件：`seen2scene/v3/data.py` 117–128、152–200

```python
    def sample_batch(self, rng, n_views=32, n_states=4, max_cands=8, stale_p=0.5, stale_max=9, weights=None):
        out, n = [], 0
        while n < n_views:
            ti, t = self.states[rng.choice(len(self.states), p=weights)]
            rows = self.state_rows[(ti, t)]
            k = min(len(rows), max_cands, n_views - n)
            rows = np.sort(rng.choice(rows, k, replace=False))
            cap = self.stale_cap(t, stale_max)
            dlt = int(rng.integers(1, cap + 1)) if (cap > 0 and rng.random() < stale_p) else 0
            out.append((ti, t, dlt, rows))
            n += k
        return out

    def state_tensors(self, ti, t, dlt, rows, dev):
        """Rays, channels, masks, targets of the views (state t) and the VAE input grid (state t - dlt)."""
        cache = self.caches[ti]
        assert (t - dlt) // cache.refresh == t // cache.refresh, (t, dlt)
        origin = torch.as_tensor(cache.block(t)["origin"], dtype=torch.float64, device=dev)
        v = self.v
        R = torch.as_tensor(v["R"][rows], dtype=torch.float64, device=dev)
        T = torch.as_tensor(v["T"][rows], dtype=torch.float64, device=dev)
        J = torch.as_tensor(v["jitter"][rows], dtype=torch.float64, device=dev)
        o, d, tcap, _ = RY.model_rays_torch(R, T, J, origin, self.k_M[ti])
        o, d, tcap = o.float(), d.float(), tcap.float()
        g = {k: torch.as_tensor(v[k][rows], device=dev) for k in VIEW_KEYS}
        st16 = torch.stack([g["tk"], g["dlo"], g["dhi"], g["lu"]], 1)
        ch = G.channels_from_stats(g["cls"], st16, g["nseg"], d[..., 2], o[:, 2], tcap)
        m_unk = g["cls"] == G.CLS_UNKNOWN
        bb = g["b"].float()
        lmask = m_unk & (g["b"] != 0)
        x1 = torch.stack([torch.where(g["b"] == 0, torch.full_like(bb, -1.0), bb), g["y"].float()], 1)
        x4, valid4 = cache.state(t - dlt, dev)
        return {"o": o, "d": d, "tcap": tcap, "ch": ch, "m_unk": M.pad_rays(m_unk), "lmask": M.pad_rays(lmask),
                "x1": M.pad_rays(x1), "cat": g["cat"], "lu_vox": g["lu"].float() * tcap / G.VS, "x4": x4, "valid4": valid4}


def build_cond(model, data, batch, dev, autocast=True):
    """-> dict: cond [V, C, 64, 128], mem (T1) or None, m_unk / lmask / x1 / cat / lu_vox (per view), vae: per-state
    [{mu, logvar, valid8, x4, valid4}]. The scene encoder runs ONCE per state; gradients flow through everything."""
    conds, mems, states = [], [], []
    keep = {"m_unk": [], "lmask": [], "x1": [], "cat": [], "lu_vox": []}
    for ti, t, dlt, rows in batch:
        st = data.state_tensors(ti, t, dlt, rows, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            h = model.encode(st["x4"], st["valid4"])
            out = model.view_cond(h, st["o"], st["d"], st["tcap"], st["ch"])
        if model.xa:
            cond, mem = out
            mems.append(mem)
        else:
            cond = out
        conds.append(cond)
        for k in keep:
            keep[k].append(st[k])
        states.append({"mu": h["mu"], "logvar": h["logvar"], "valid8": h["valid8"], "x4": st["x4"], "valid4": st["valid4"]})
    bt = {k: torch.cat(v) for k, v in keep.items()}
    bt["cond"] = torch.cat(conds)
    bt["mem"] = tuple(torch.cat([m[i] for m in mems]) for i in range(3)) if mems else None
    bt["vae"] = states
    return bt
```

---

## 3. 模型架构

### 3.1 架构图

```
                       一个状态（一张地图）                              每个候选视角 (V 个)
  ┌──────────────────────────────────────────────┐        ┌──────────────────────────────────────────┐
  │ 4× 网格 x [1,3,X,Y,Z] + valid [1,1,X,Y,Z]     │        │ 射线 o [3], d [7296,3], tcap [7296]       │
  │    ch: sdf/TRUNC, band 比例, free 比例         │        │ 投影通道 ch [11, 7296]                     │
  └───────────────┬──────────────────────────────┘        └───────────────┬──────────────────────────┘
                  ▼ VAEEncoder (3D conv)                                  │
  stem 3x3x3 (4→32) → Res3D(32) ──── F4 (32 ch @4×) ──┐                   │
  SiLU → stride-2 conv (32→64) → 2×Res3D(64)          │                   │
  GN → SiLU → 1x1 conv (64→32) → chunk → μ, logvar     │                   │
                  │                                   │                   │
                  ▼                                   │                   │
  μ [1,16,X/2,Y/2,Z/2] (@8× 体素, valid8 掩码)         │                   │
   ├──────────────────────────────┐                   │                   │
   │ (训练) reparameterize → z    │                   │                   │
   │        VAEDecoder → [sdf_hat, 3 类 logits] @4×   │   L_recon, KL     │
   │                              │                   │                   │
   ▼ LatentRayEncoder                                 │                   │
  沿每条射线采 32 点 grid_sample(μ) + valid8·inside      │                   │
  → [7296, 17, 32] → Conv1d 17→64 → Conv1d s2 → Conv1d s2 → fc 512→64       │
  → rf [7296, 64]                                     │                   │
   │                                                  │                   │
   ▼ cond = [ch 11 ‖ rf 64 ‖ valid 1] → [V, 76, 64, 128]  (114 列 + 零填充) │
   │                                                                      │
   ▼ FlowUNet（整流流，输入 [x_τ (2) ‖ cond (76)] = 78 通道）                  │
  inp 3x3 (78→64) @64×128
  ├ L0 64 ch  @64×128 : Res2D ×2                       ─┐ skip
  ├ ↓ conv s2
  ├ L1 128 ch @32×64  : Res2D ×2                       ─┤
  ├ ↓ conv s2
  ├ L2 192 ch @16×32  : Res2D ×2                       ─┤
  ├ ↓ conv s2
  ├ L3 256 ch @8×16   : (Res2D + Attn2D) ×2            ─┤
  ├ mid: Res2D → Attn2D(自注意力) → Res2D          （T0：xattn = None，这里没有交叉注意力）
  ├ up L3 : (Res2D(cat skip) + Attn2D) ×3 → ↑ nearest + conv
  ├ up L2 : Res2D ×3 → ↑ ; up L1 : Res2D ×3 → ↑ ; up L0 : Res2D ×3
  └ out: GN → SiLU → conv 3x3 (64→2)   (零初始化)  → 速度 v [V, 2, 64, 128]
                                                                            
  采样: x_0 = ε·m_unk → 10 步 Euler x ← (x + v/10)·m_unk → (b, y) 每条射线
```

每个 Res2D 都接收时间步嵌入 `te`（FiLM 调制）；`m_unk` 把非 UNKNOWN 射线上的状态始终置零。

### 3.2 V3Model 装配

T0 与 T1 共用这份代码。T0 的配置里 `xattn.enabled = false`，所以 `self.xa = None`：`memory()` 返回 None，`view_cond` 只返回 `cond`，`velocity` 收到的 `mem` 恒为 None。

文件：`seen2scene/v3/model.py` 42–96

```python
class V3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        v, r, u, xa = cfg["vae"], cfg["ray"], cfg["unet"], cfg.get("xattn", {"enabled": False})
        self.vae_enc = VAEEncoder(v["in_ch"], tuple(v["ch"]), v["latent_ch"], v["groups"], v["act_ckpt"])
        self.vae_dec = VAEDecoder(v["latent_ch"], tuple(v["ch"]), v["groups"], 4, v["act_ckpt"])
        self.rays = LatentRayEncoder(r["n_points"], v["latent_ch"], r["hidden"], r["out_ch"])     # base adds the +1 valid channel
        self.x_ch, self.proj_ch, self.ray_ch = cfg["flow"]["x_ch"], cfg["proj_ch"], r["out_ch"]
        self.cond_ch = self.proj_ch + self.ray_ch + 1
        self.xa = dict(xa) if xa.get("enabled") else None
        if self.xa:
            self.xa["mem_dim"] = v["latent_ch"] + 2 + 6 * xa["n_freq"]
            self.xa["q_pe_dim"] = 12 * xa["n_freq"]
        self.unet = FlowUNet(self.x_ch + self.cond_ch, self.x_ch, tuple(u["ch"]), u["blocks_per_level"], tuple(u["attn_levels"]),
                             u["heads"], u["t_emb"], u["groups"], xattn=self.xa)
        self.rays_ckpt = True

    # ------------------------------------------------------------------------------------------------ scene side
    def encode(self, x, valid):
        """x [1, 3, X4, Y4, Z4], valid [1, 1, ...] -> h (mu is the flow condition; logvar for the training terms)."""
        mu, logvar, valid8 = self.vae_enc(x, valid)
        band8 = F.max_pool3d(x[:, 1:2].float(), 2)
        free8 = F.max_pool3d(x[:, 2:3].float(), 2)
        return {"mu": mu, "logvar": logvar, "valid8": valid8, "band8": band8, "free8": free8, "ext": PT.extent(x.shape[2:])}

    def decode(self, z):
        return self.vae_dec(z)

    def memory(self, h):
        return XA.SceneMemory(h["mu"], h["valid8"], h["band8"], h["free8"], h["ext"]) if self.xa else None

    # ------------------------------------------------------------------------------------------------- view side
    def view_cond(self, h, o, d, tcap, ch, mem=None):
        """o [V, 3], d [V, R, 3], tcap [V, R], ch [V, 11, R] -> cond [V, C, 64, 128]; T1 also returns the memory
        tuple (tokens [V, budget, F], mask [V, budget], query pe [V, L, P]) built from mem (SceneMemory of h)."""
        V, R = d.shape[:2]
        args = (h["mu"], h["valid8"], h["ext"], o[:, None].expand(V, R, 3).reshape(-1, 3), d.reshape(-1, 3), tcap.reshape(-1))
        if self.training and torch.is_grad_enabled() and self.rays_ckpt:
            rf = torch.utils.checkpoint.checkpoint(self.rays, *args, use_reentrant=False)
        else:
            rf = self.rays(*args)
        c = torch.cat([ch.to(rf.dtype), rf.reshape(V, R, -1).transpose(1, 2)], 1).reshape(V, -1, GH, GW)
        c = F.pad(c, (0, GW_PAD - GW))
        valid = torch.zeros(V, 1, GH, GW_PAD, device=c.device, dtype=c.dtype)
        valid[..., :GW] = 1
        cond = torch.cat([c, valid], 1)
        if not self.xa:
            return cond
        mem = mem if mem is not None else self.memory(h)
        tok, mask = mem.batch([(o[i], d[i], tcap[i]) for i in range(V)], self.xa["budget"], self.xa["fov_margin_deg"], self.xa["n_freq"])
        qpe = torch.stack([XA.query_pe(o[i], d[i], h["ext"], self.xa["n_freq"]) for i in range(V)])
        return cond, (tok, mask, qpe)

    def velocity(self, x, tau, cond, mem=None, return_features=False):
        return self.unet(torch.cat([x, cond.to(x.dtype)], 1), tau, mem, return_features)
```

模型配置（`experiments/exp6_v3/configs/model_v3.json`；T0 用的就是下面这份，`xattn.enabled = false`；T1 只改这一个字段）：

```json
{
 "input": {"factor": 4, "pad_to": 8, "channels": ["sdf", "band", "free"], "valid_channel": true},
 "vae": {"in_ch": 4, "ch": [32, 64], "latent_ch": 16, "groups": 8, "act_ckpt": true, "latent_stride_fine_voxels": 8},
 "ray": {"n_points": 32, "hidden": 64, "out_ch": 64},
 "proj_ch": 11,
 "flow": {"x_ch": 2, "euler_steps": 10, "K": 4},
 "unet": {"ch": [64, 128, 192, 256], "blocks_per_level": 2, "attn_levels": [3], "heads": 4, "t_emb": 256, "groups": 16},
 "xattn": {"enabled": false, "heads": 4, "budget": 512, "n_freq": 8, "fov_margin_deg": 10.0}
}
```

### 3.3 VAE 编码器 / 解码器

文件：`seen2scene/v3/vae.py` 20–81；`seen2scene/futureview/model.py` 19–36（`gn`、`Res3D`）

```python
def gn(c, groups):
    g = min(groups, c)
    while c % g:
        g -= 1
    return nn.GroupNorm(g, c)


class Res3D(nn.Module):
    def __init__(self, c, groups=8, dilation=1):
        super().__init__()
        self.n1, self.c1 = gn(c, groups), nn.Conv3d(c, c, 3, padding=dilation, dilation=dilation)
        self.n2, self.c2 = gn(c, groups), nn.Conv3d(c, c, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + self.c2(F.silu(self.n2(self.c1(F.silu(self.n1(x))))))
```

```python
class VAEEncoder(nn.Module):
    def __init__(self, in_ch=4, ch=(32, 64), latent_ch=16, groups=8, act_ckpt=True):
        super().__init__()
        c0, c1 = ch
        self.stem = nn.Conv3d(in_ch, c0, 3, padding=1)
        self.r0 = Res3D(c0, groups)
        self.down = nn.Conv3d(c0, c1, 3, stride=2, padding=1)
        self.r1 = nn.Sequential(Res3D(c1, groups), Res3D(c1, groups))
        self.head = nn.Sequential(gn(c1, groups), nn.SiLU(), nn.Conv3d(c1, 2 * latent_ch, 1))
        self.latent_ch, self.act_ckpt = latent_ch, act_ckpt
        nn.init.zeros_(self.head[-1].bias)

    def _full_res(self, x):
        return self.r0(self.stem(x))

    def forward(self, x, valid):
        """x [B, 3, X4, Y4, Z4], valid [B, 1, X4, Y4, Z4] -> mu, logvar [B, latent_ch, X4/2, Y4/2, Z4/2], valid8 [B, 1, ...]."""
        inp = torch.cat([x, valid.to(x.dtype)], 1)
        if self.act_ckpt and self.training and torch.is_grad_enabled():
            h = torch.utils.checkpoint.checkpoint(self._full_res, inp, use_reentrant=False)
        else:
            h = self._full_res(inp)
        h = self.r1(self.down(F.silu(h)))
        mu, logvar = self.head(h).chunk(2, 1)
        valid8 = (F.avg_pool3d(valid.float(), 2) > 0).to(x.dtype)
        return mu * valid8, logvar.clamp(-10.0, 10.0) * valid8, valid8


class VAEDecoder(nn.Module):
    def __init__(self, latent_ch=16, ch=(32, 64), groups=8, out_ch=4, act_ckpt=True):
        super().__init__()
        c0, c1 = ch
        self.inp = nn.Conv3d(latent_ch, c1, 3, padding=1)
        self.r1 = nn.Sequential(Res3D(c1, groups), Res3D(c1, groups))
        self.up = nn.Conv3d(c1, c0, 3, padding=1)
        self.r0 = Res3D(c0, groups)
        self.out = nn.Sequential(gn(c0, groups), nn.SiLU(), nn.Conv3d(c0, out_ch, 3, padding=1))
        self.act_ckpt = act_ckpt

    def _full_res(self, h):
        return self.out(self.r0(self.up(F.interpolate(h, scale_factor=2, mode="nearest"))))

    def forward(self, z):
        h = self.r1(self.inp(z))
        h = F.silu(h)
        if self.act_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(self._full_res, h, use_reentrant=False)
        return self._full_res(h)


def reparameterize(mu, logvar, generator=None):
    eps = torch.randn(mu.shape, device=mu.device, dtype=mu.dtype, generator=generator)
    return mu + torch.exp(0.5 * logvar) * eps


def kl_per_unit(mu, logvar, valid8):
    """Mean KL(q || N(0, I)) per valid latent unit and channel (scalar, float32); also the per-channel mean [C]."""
    kl = 0.5 * (mu.float() ** 2 + torch.exp(logvar.float()) - logvar.float() - 1.0)       # [B, C, ...]
    m = valid8.float()
    n = m.sum().clamp(min=1.0)
    per_ch = (kl * m).sum(dim=(0, 2, 3, 4)) / n
    return per_ch.mean(), per_ch
```

要点：解码器只在训练时存在，重建的是**同一张观测地图**（不是补全场景）；流匹配的条件用 `μ`（不采样），重建和 KL 通过重参数化同时约束 `μ, logvar`。

### 3.4 Latent Ray Encoder

文件：`seen2scene/v3/model.py` 24–39；基类 `seen2scene/futureview/model.py` 27–41、59–64

```python
class RayEncoder(nn.Module):
    """Features of N points t_j = (j + 0.5) / N * t_cap along every ray, sampled from (F0, F1) -> [R, out_ch]."""

    def __init__(self, n_points=32, feat_ch=96, hidden=64, out_ch=64):
        super().__init__()
        self.n = n_points
        self.c1 = nn.Conv1d(feat_ch + 1, hidden, 3, padding=1)
        self.c2 = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
        self.c3 = nn.Conv1d(hidden, hidden, 3, stride=2, padding=1)
        self.fc = nn.Linear(hidden * (n_points // 4), out_ch)

    def points(self, o, d, tcap):
        t = (torch.arange(self.n, device=d.device, dtype=d.dtype) + 0.5) / self.n
        o = o[None, None] if o.dim() == 1 else o[:, None]
        return o + d[:, None] * (tcap[:, None] * t[None])[..., None]                       # [R, N, 3]
```

```python
class LatentRayEncoder(M.RayEncoder):
    """The V2 ray encoder (same Conv1d stack) fed by the VAE latent: per point [mu (C), valid8 * inside]."""

    def forward(self, mu, valid8, ext, o, d, tcap):
        p = self.points(o, d, tcap)
        g = 2.0 * p / ext.to(p.device, p.dtype) - 1.0
        grid = torch.stack([g[..., 2], g[..., 1], g[..., 0]], -1)[None, :, :, None].float()       # [1, R, N, 1, 3]
        with torch.autocast("cuda", enabled=False):
            f0 = F.grid_sample(mu.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            fv = F.grid_sample(valid8.float(), grid, mode="nearest", padding_mode="zeros", align_corners=False)
        inside = ((g > -1) & (g < 1)).all(-1)[None, None, ..., None].float()
        f = torch.cat([f0, (fv > 0).float() * inside], 1)[0, ..., 0].permute(1, 0, 2)              # [R, C + 1, N]
        h = F.silu(self.c1(f))
        h = F.silu(self.c2(h))
        h = F.silu(self.c3(h))
        return self.fc(h.flatten(1))
```

V3 里 `feat_ch = latent_ch = 16`，所以 Conv1d 输入 17 通道、32 点；两次 stride-2 后 8 点 × 64 = 512 → fc → 64 维射线特征。

### 3.5 条件图 cond 的组成

| 通道 | 来源 | 数量 |
|---|---|---|
| 0–10 | 投影通道 `ch`（§2.4） | 11 |
| 11–74 | 射线特征 `rf`（§3.4） | 64 |
| 75 | valid（114 列为 1，填充列为 0） | 1 |
| 合计 | `cond [V, 76, 64, 128]`；U-Net 输入再拼上 `x_τ [V, 2, …]` | 78 |

### 3.6 整流流 U-Net

文件：`seen2scene/futureview/model.py` 87–178（`timestep_embedding`、`Res2D`、`Attn2D`、`UNet2D`）

```python
def timestep_embedding(tau, dim=64):
    half = dim // 2
    f = torch.exp(-math.log(10000.0) * torch.arange(half, device=tau.device, dtype=torch.float32) / half)
    a = (tau.float() * 1000.0)[:, None] * f[None]
    return torch.cat([torch.sin(a), torch.cos(a)], 1)


class Res2D(nn.Module):
    def __init__(self, cin, cout, temb, groups):
        super().__init__()
        self.n1, self.c1 = gn(cin, groups), nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(temb, 2 * cout)
        self.n2, self.c2 = gn(cout, groups), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, te):
        h = self.c1(F.silu(self.n1(x)))
        sc, sh = self.film(te)[:, :, None, None].chunk(2, 1)
        h = self.n2(h) * (1 + sc) + sh
        return self.skip(x) + self.c2(F.silu(h))


class Attn2D(nn.Module):
    def __init__(self, c, heads, groups):
        super().__init__()
        self.n, self.h = gn(c, groups), heads
        self.qkv, self.proj = nn.Conv2d(c, 3 * c, 1), nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, self.h, C // self.h, H * W).unbind(1)
        o = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
        return x + self.proj(o.transpose(-1, -2).reshape(B, C, H, W))


class UNet2D(nn.Module):
    def __init__(self, in_ch, out_ch=2, ch=(64, 128, 192, 256), blocks=2, attn_levels=(3,), heads=4, t_emb=256, groups=16):
        super().__init__()
        self.temb = nn.Sequential(nn.Linear(64, t_emb), nn.SiLU(), nn.Linear(t_emb, t_emb))
        self.inp = nn.Conv2d(in_ch, ch[0], 3, padding=1)
        self.down, self.skips_ch = nn.ModuleList(), [ch[0]]
        c = ch[0]
        for lv, co in enumerate(ch):
            for _ in range(blocks):
                m = nn.ModuleList([Res2D(c, co, t_emb, groups)])
                if lv in attn_levels:
                    m.append(Attn2D(co, heads, groups))
                self.down.append(m)
                c = co
                self.skips_ch.append(c)
            if lv < len(ch) - 1:
                self.down.append(nn.ModuleList([nn.Conv2d(c, c, 3, stride=2, padding=1)]))
                self.skips_ch.append(c)
        self.mid = nn.ModuleList([Res2D(c, c, t_emb, groups), Attn2D(c, heads, groups), Res2D(c, c, t_emb, groups)])
        self.up = nn.ModuleList()
        skips = list(self.skips_ch)
        for lv in reversed(range(len(ch))):
            co = ch[lv]
            for _ in range(blocks + 1):
                m = nn.ModuleList([Res2D(c + skips.pop(), co, t_emb, groups)])
                if lv in attn_levels:
                    m.append(Attn2D(co, heads, groups))
                self.up.append(m)
                c = co
            if lv > 0:
                self.up.append(nn.ModuleList([nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c, c, 3, padding=1)]))
        self.out = nn.Sequential(gn(c, groups), nn.SiLU(), nn.Conv2d(c, out_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
```

V3 的 `FlowUNet` 继承它。T1 在 `mid` 的自注意力之后插入一个交叉注意力块；**T0 时 `xattn=None`，下面 `forward` 里的 `if self.xattn is not None` 分支不执行，前向与 V2 的 U-Net 完全相同**（输入 78 通道 → 四级下行 / 上行 → 速度 2 通道）：

文件：`seen2scene/v3/flow_unet.py` 12–44

```python
class FlowUNet(UNet2D):
    def __init__(self, in_ch, out_ch=2, ch=(64, 128, 192, 256), blocks=2, attn_levels=(3,), heads=4, t_emb=256, groups=16,
                 xattn=None):
        super().__init__(in_ch, out_ch, ch, blocks, attn_levels, heads, t_emb, groups)
        self.xattn = CrossAttnBlock(ch[-1], xattn["mem_dim"], xattn["q_pe_dim"], xattn["heads"]) if xattn else None

    def forward(self, x, tau, mem=None, return_features=False):
        """mem: None or (tokens [B, N, F], mask [B, N], query pe [B, L, P]) for the cross-attention block."""
        te = self.temb(timestep_embedding(tau))
        h = self.inp(x)
        hs = [h]
        for m in self.down:
            if isinstance(m[0], Res2D):
                h = m[0](h, te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[0](h)
            hs.append(h)
        h = self.mid[1](self.mid[0](h, te))
        if self.xattn is not None:
            assert mem is not None, "T1 needs the scene memory"
            h = self.xattn(h, *mem)
        h = self.mid[2](h, te)
        for m in self.up:
            if isinstance(m[0], Res2D):
                h = m[0](torch.cat([h, hs.pop()], 1), te)
                for a in m[1:]:
                    h = a(h)
            else:
                h = m[1](m[0](h))
        out = self.out(h)
        return (out, h) if return_features else out
```

### 3.7 T0 没有交叉注意力（与 T1 的差别所在）

T1 在上面 `FlowUNet.forward` 的 `mid[1]` 之后调用 `CrossAttnBlock`：Query = 瓶颈 8 × 16 = 128 个 token 加相机 / 射线方向的 Fourier 编码，Key / Value = `SceneMemory`（μ 网格里有观测支持的格子，特征 `[μ16 ‖ band8 ‖ free8 ‖ PE48]`，每个候选取视锥内 ≤ 512 个），输出投影 `W_o` 零初始化。T0 不构造 `SceneMemory`、不实例化 `CrossAttnBlock`（`seen2scene/v3/cross_attention.py` 在 T0 里没有任何对象被创建）。

因此在 T0 里，**μ 进入生成器只有一条路**：§3.4 的沿射线采样 → 64 维射线特征 → §3.5 的条件图。这也是 Task A 发现 T1 的交叉注意力没有贡献的原因：它读的是同一份 μ，而 7,296 条射线 × 32 个采样点已经把候选视锥铺满。T1 版本的完整代码见 `V3_T1_D1_MODEL_ARCHITECTURE.md` §3.7。

### 3.8 参数量

文件：`seen2scene/v3/model.py` 132–136；数值来自 T0 的运行清单 `work_dirs/exp6_v3/task0_vae/runs/main/run_none.json`

```python
def param_counts(model):
    tot = sum(p.numel() for p in model.parameters())
    xa = sum(p.numel() for n, p in model.named_parameters() if n.startswith("unet.xattn."))
    dec = sum(p.numel() for n, p in model.named_parameters() if n.startswith("vae_dec."))
    return {"total": tot, "xattn": xa, "vae_decoder_train_only": dec, "inference": tot - dec}
```

| 部分 | 参数 |
|---|---|
| 总计 | 23,539,622 |
| 其中交叉注意力 `unet.xattn` | 0（T0 没有这个模块） |
| 其中 VAE 解码器（仅训练） | 585,188 |
| 推理时 | 22,954,434 |
| T1（对照：多一个交叉注意力块） | 23,730,986（+191,364） |

---

## 4. 训练算法

### 4.1 损失

$$L = L_\text{FM} + \lambda_\text{rec} L_\text{recon} + \beta(\text{step}) \cdot \text{KL},\qquad \lambda_\text{rec}=1,\ \beta: 0 \to 0.02\ (5{,}000\ \text{步线性})$$

- **L_FM**（整流流）：`x_τ = (1 − τ) ε + τ x1`，目标速度 `x1 − ε`；只在 `lmask` 射线上计算，除以 2 × 全局有效射线数；
- **L_recon**：VAE 解码器对输入网格的重建——三类（EMPTY / UNKNOWN / BAND）类别均衡交叉熵 + BAND 格子上 sdf 的 L1；
- **KL**：每个有效潜变量单元每通道的 KL(q ‖ N(0, I)) 均值。

文件：`seen2scene/v3/losses.py` 18–71

```python
def fm_terms(model, bt, eps, tau, autocast=True):
    """(sum of squared velocity errors over labelled M_unknown rays, number of such rays)."""
    m = bt["m_unk"][:, None].float()
    x1 = bt["x1"] * m
    xt = ((1.0 - tau)[:, None, None, None] * eps + tau[:, None, None, None] * x1) * m
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        v = model.velocity(xt, tau, bt["cond"], bt.get("mem"))
    err = ((v.float() - (x1 - eps)) ** 2).sum(1)
    lm = bt["lmask"].float()
    return (err * lm).sum(), lm.sum()


def recon_terms(model, states, generator=None, autocast=True):
    """states: [{mu, logvar, valid8, x4, valid4}] -> (L_recon mean over states, diagnostics dict)."""
    ce_tot, l1_tot, n = 0.0, 0.0, 0
    diag = {"acc_empty": [], "acc_unknown": [], "acc_band": [], "sdf_mae": []}
    for st in states:
        z = VAE.reparameterize(st["mu"], st["logvar"], generator)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model.decode(z)
        sdf_hat, logits = out[:, 0:1].float(), out[:, 1:4].float()
        cls, sdf_t, band_m, vmask = PT.recon_targets(st["x4"], st["valid4"])
        ce = F.cross_entropy(logits, cls, reduction="none")
        terms = []
        for c, nm in ((PT.EMPTY, "acc_empty"), (PT.UNKNOWN, "acc_unknown"), (PT.BAND, "acc_band")):
            mc = vmask & (cls == c)
            if mc.any():
                terms.append(ce[mc].mean())
                diag[nm].append(float((logits.argmax(1)[mc] == c).float().mean()))
        ce_c = torch.stack(terms).mean() if terms else logits.sum() * 0.0
        nb = band_m.sum().clamp(min=1.0)
        l1 = ((sdf_hat - sdf_t).abs() * band_m).sum() / nb
        diag["sdf_mae"].append(float(l1))
        ce_tot, l1_tot, n = ce_tot + ce_c, l1_tot + l1, n + 1
    n = max(n, 1)
    return (ce_tot + l1_tot) / n, {"ce": float(ce_tot) / n, "l1": float(l1_tot) / n,
                                    **{k: (sum(v) / len(v) if v else None) for k, v in diag.items()}}


def kl_terms(states):
    """-> (mean KL per unit over the states, per-channel mean [C])."""
    kls, chs = [], []
    for st in states:
        k, per_ch = VAE.kl_per_unit(st["mu"], st["logvar"], st["valid8"])
        kls.append(k)
        chs.append(per_ch)
    return torch.stack(kls).mean(), torch.stack(chs).mean(0)


def beta_at(step, cfg):
    k = cfg["kl"]
    if step < k["start"]:
        return 0.0
    return k["beta"] * min(1.0, (step - k["start"] + 1) / max(1, k["warmup"]))
```

### 4.2 训练循环

每步：两个域各抽 1 个状态、8 个候选视角（每 rank），4 个 rank → 全局 64 视角；`build_mixed` 对每个状态跑一次编码器；`fm_terms` 的射线数做全局 all-reduce 后归一化；梯度手动 all-reduce（平铺一次），裁剪 1.0，AdamW，EMA 0.999。

文件：`seen2scene/v3/trainer.py` 61–65（学习率）、206–212（抽样）、236–275（主循环）、312–330（验证 / 存档）

```python
def lr_at(step, cfg):
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    p = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    return cfg["lr"] * 0.5 * (1.0 + math.cos(math.pi * min(1.0, p)))
```

```python
    def draw():
        return [(data["in_train"], data["in_train"].sample_batch(rng, half, hs, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_in)),
                (data["out_train"], data["out_train"].sample_batch(rng, half, hs, cfg["max_cands_per_state"], cfg["stale_p"], cfg["stale_max"], w_out))]
```

```python
        lr = cfg["lr"] if a.smoke == "overfit" else lr_at(step, cfg)
        for gp in opt.param_groups:
            gp["lr"] = lr
        beta = L.beta_at(step, cfg)
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        acc = {"esum": 0.0, "nray": 0.0, "rec": 0.0, "kl": 0.0, "rays_in": 0.0, "rays_out": 0.0}
        rec_diag = {}
        for _ in range(accum):
            parts = fixed if fixed is not None else draw()
            bt = D.build_mixed(model, parts, dev)
            V = bt["cond"].shape[0]
            eps = torch.randn(V, 2, M.GH, M.GW_PAD, device=dev)
            tau = torch.rand(V, device=dev)
            esum, nray = L.fm_terms(model, bt, eps, tau)
            rec, rec_diag = L.recon_terms(model, bt["vae"])
            kl, kl_ch = L.kl_terms(bt["vae"])
            tot = torch.stack([esum.detach(), nray]).double()
            if ddp:
                dist.all_reduce(tot)
            nray_g = tot[1].clamp(min=1.0)
            loss = esum / (2.0 * nray_g.to(esum.dtype)) + (cfg["lambda_rec"] * rec + beta * kl) / (world * accum)
            if not torch.isfinite(loss):
                raise RuntimeError(f"STOP: non-finite loss at step {step} (fm {float(esum)}, rec {float(rec)}, kl {float(kl)})")
            loss.backward()
            n_in = bt["n_views"][0]
            with torch.no_grad():
                lm = bt["lmask"].float()
                acc["rays_in"] += float(lm[:n_in].sum()); acc["rays_out"] += float(lm[n_in:].sum())
                acc["esum"] += float(tot[0]); acc["nray"] += float(tot[1]); acc["rec"] += float(rec) / accum; acc["kl"] += float(kl) / accum
        if ddp:
            params = [p_ for p_ in model.parameters()]
            for p_ in params:
                if p_.grad is None:
                    p_.grad = torch.zeros_like(p_)
            flat = torch._utils._flatten_dense_tensors([p_.grad for p_ in params])
            dist.all_reduce(flat)
            for g_, s_ in zip(params, torch._utils._unflatten_dense_tensors(flat, [p_.grad for p_ in params])):
                g_.grad.copy_(s_)
```

```python
        stop = torch.zeros(1, device=dev)
        if a.smoke == "none" and rank == 0:
            if (step + 1) % cfg["eval_every"] == 0 or step + 1 == steps:
                evaluate(step + 1)
                if early_stop(ev_p, cfg, step + 1):
                    say(f"early stop at step {step + 1} (frozen rule: no selection-score improvement for "
                        f"{cfg['early_stop']['patience_evals']} evaluations after {cfg['early_stop']['after_fraction']:.2f} of the budget)")
                    stop += 1
            if (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps:
                save_ckpt(ck_last, model, ema, opt, step + 1, rng)
        if ddp and a.smoke == "none" and ((step + 1) % cfg["eval_every"] == 0 or (step + 1) % cfg["ckpt_every"] == 0 or step + 1 == steps):
            dist.all_reduce(stop)
            dist.barrier()
        if float(stop) > 0:
            run["early_stopped_at"] = step + 1
            break
```

训练配置（`experiments/exp6_v3/configs/train_v3.json`）的关键字段：

| 字段 | 值 |
|---|---|
| seed / steps | 0 / 40,000 |
| lr / warmup / 调度 | 1e-4 / 1,000 步 / 余弦到 0 |
| AdamW betas / weight_decay / grad_clip | (0.9, 0.99) / 0.01 / 1.0 |
| precision / EMA | bf16 / 0.999 |
| batch_views / states_per_batch | 64（室内 32 + 室外 32）/ 8（每 rank 1 + 1） |
| stale_p / stale_max | 0.5 / 9 |
| state_weight（t ≤ 20 的状态权重） | ×2.0 |
| lambda_rec / kl | 1.0 / β 0.02，5,000 步预热 |
| eval_every / ckpt_every | 5,000 / 1,000 |
| early_stop | 过 2/3 预算后连续 2 次验证无改善则停（T0 未触发；选模分数单调下降） |

### 4.3 验证与选模

验证集：VALSEL 2,048 个视角、S-c 2,048 个视角（固定子集，确定性噪声）。主要指标 **fair ES**（energy score，单位：UNKNOWN 弧长的体素数）：
`ES = E‖ℓ_k − ℓ*‖ − (1/2) E‖ℓ_i − ℓ_j‖`，其中 ℓ 是样本（或标签）的命中弧长（未命中记为全长）。

文件：`seen2scene/v3/eval_dev.py` 66–118（节选）

```python
def eval_samples(model, data, states, name, dev, K=4, steps=10):
    """Sample metrics on M_unknown rays with a valid label, in unknown-arclength voxels (the V2 definitions)."""
    model.eval()
    acc = {"brier": [0.0, 0.0], "fw": [0.0, 0.0], "ph": [0.0, 0.0], "lh": [0.0, 0.0], "es": [0.0, 0.0],
           "spread": [0.0, 0.0], "ambig": [0.0, 0.0], "s_sq": [0.0, 0.0], "s_sum": [0.0, 0.0]}
    for ch in chunks(states, 32):
        bt = D.build_cond(model, data, ch, dev)
        V = bt["cond"].shape[0]
        g = gen(dev, f"exp6-V3|evalsample|{name}|{ch[0][0]}|{ch[0][1]}")
        eps = torch.randn(K, V, 2, M.GH, M.GW_PAD, generator=g, device=dev).reshape(K * V, 2, M.GH, M.GW_PAD)
        cond = bt["cond"].repeat(K, 1, 1, 1)
        mu = bt["m_unk"].repeat(K, 1, 1)
        mem = None if bt["mem"] is None else tuple(m.repeat(K, *([1] * (m.dim() - 1))) for m in bt["mem"])
        x = V3.sample(model, cond, mu, eps, mem, steps).reshape(K, V, 2, M.GH, M.GW_PAD)
        x = M.unpad_rays(x)
        b, y = x[:, :, 0], x[:, :, 1]
        lm = M.unpad_rays(bt["lmask"])
        cat = bt["cat"].reshape(V, -1)
        lab = (cat == G.CAT["HIT"]) & lm
        hit = (b > 0) & (y < 1.0)
        p = hit.float().mean(0)
        s = ((y + 1) / 2).clamp(0, 1)
        lu = bt["lu_vox"].reshape(V, -1)
        ell = torch.where(hit, s, torch.ones_like(s)) * lu
        s_lab = ((M.unpad_rays(bt["x1"])[:, 1] + 1) / 2).clamp(0, 1)
        ell_t = torch.where(lab, s_lab, torch.ones_like(s_lab)) * lu
        nohit = (cat == G.CAT["NOHIT"]) & lm
        for v in range(V):
            m = lm[v]
            if not m.any():
                continue
            rms = lambda a: torch.sqrt((a[..., m] ** 2).mean(-1))      # noqa: E731
            e1 = rms(ell[:, v] - ell_t[v][None]).mean()
            pair = sum(rms(ell[i, v] - ell[j, v]) for i in range(K) for j in range(K) if i != j)
            acc["es"][0] += float(e1 - pair / (2 * K * (K - 1)))
            acc["es"][1] += 1
        acc["brier"][0] += float(((p - lab.float()) ** 2)[lm].sum()); acc["brier"][1] += float(lm.sum())
        acc["fw"][0] += float(p[nohit].sum()); acc["fw"][1] += float(nohit.sum())
        ...
```

选模规则（冻结）：

文件：`seen2scene/v3/trainer.py` 81–103

```python
def selection_score(evals, base_step=5000):
    """Frozen checkpoint-selection score of every evaluated step >= base_step (None where the base is missing)."""
    base = next((e for e in evals if e["step"] == base_step), None)
    if base is None:
        return {}
    es = lambda e, k: e[f"samp_{k}"]["fair_es_arclength_vox"]  # noqa: E731
    return {e["step"]: es(e, "in_val") / es(base, "in_val") + es(e, "out_sc") / es(base, "out_sc") for e in evals if e["step"] >= base_step}


def early_stop(ev_p, cfg, step):
    """Protocol v1.0: after after_fraction of the budget, stop when the last patience_evals evaluations did not improve
    the best selection score."""
    es = cfg.get("early_stop")
    if not es or step < es["after_fraction"] * cfg["steps"]:
        return False
    evals = [e for e in map(json.loads, open(ev_p)) if e["step"] <= step]
    sc = selection_score(evals)
    if len(sc) < es["patience_evals"] + 1:
        return False
    steps_sorted = sorted(sc)
    best_before = min(sc[k] for k in steps_sorted[:-es["patience_evals"]])
    return all(sc[k] >= best_before for k in steps_sorted[-es["patience_evals"]:])
```

### 4.4 T0@40K 的实际训练记录

来源：`work_dirs/exp6_v3/task0_vae/logs/train.log`、`runs/main/eval.jsonl`、`checkpoints/final.json`、`experiments/exp6_v3/task0_vae/report/T0_TRAINING.md`。

| 项 | 值 |
|---|---|
| 节点 / GPU | gpu117，4 × NVIDIA L40S |
| 时间 | 2026-10-30 06:24 → 22:18（15 h 54 min），一次跑完 |
| 速度 / 显存 | 约 1.34–1.37 s / 步，峰值约 33 GB / 卡 |
| 公共模块初始化哈希 | `6a9d6071a22564c362177bcd2fac0ea8652fe83f3e60de655f299eed0363b513`（与 T1 相同）；第 1 步损失 3.6745（与 T1 逐位相同） |
| 验证 fair ES（室内 / 室外，体素） | 5k 9.58 / 9.00 → 10k 8.11 / 7.86 → 15k 7.36 / 7.40 → 20k 6.96 / 7.04 → 25k 6.73 / 6.84 → 30k 6.51 / 6.57 → 35k 6.38 / 6.51 → 40k **6.37 / 6.44** |
| 验证 L_FM（室内 VALSEL / 室外 S-c） | 5k 0.131 / 0.044 → 20k 0.096 / 0.029 → 40k 0.090 / 0.027 |
| 选模分数 ES_in/ES_in(5k) + ES_out/ES_out(5k) | 2.000 → 1.719 → 1.591 → 1.509 → 1.462 → 1.410 → 1.388 → **1.380**（40k 最小）→ `final.pt` = EMA @ 40,000 |
| 40k 其他指标（室内 / 室外） | Brier 0.079 / 0.029，误报墙面率 0.072 / 0.017，命中率比 1.01 / 0.97 |
| 对照 T1@40k | fair ES 6.42 / 6.48，选模分数 1.395：T0 在验证上略好于 T1 |

---

## 5. 推理：采样与解码

### 5.1 整流流采样（Euler 10 步）

T0 调用时 `mem=None`（§5.2 的 `V3Predictor` 里 `pack = (out, None)`），其余与 T1 完全相同。

文件：`seen2scene/v3/model.py` 103–115

```python
@torch.no_grad()
def sample(model, cond, m_unk, eps, mem=None, steps=10, autocast=True, return_features=False):
    """V3 version of futureview.model.sample: Euler integration with the optional scene memory."""
    x = eps * m_unk[:, None]
    feats = None
    for i in range(steps):
        tau = torch.full((x.shape[0],), i / steps, device=x.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            out = model.velocity(x, tau, cond, mem, return_features=return_features and i == steps - 1)
        if return_features and i == steps - 1:
            out, feats = out
        x = (x + out.float() / steps) * m_unk[:, None]
    return (x, feats) if return_features else x
```

### 5.2 规划器侧的预测器

一个状态编码一次；所有候选共用同一组 K = 4 张噪声图（公共随机数，让候选之间的比较不受噪声差异干扰）。

文件：`seen2scene/v3/predictor.py` 17–84

```python
class V3Predictor:
    def __init__(self, model, device, steps=10, max_views=64, autocast=True):
        self.model, self.dev, self.steps, self.max_views, self.autocast = model.eval(), device, steps, max_views, autocast

    @torch.no_grad()
    def encode(self, packed):
        """Planner interface: the uint8 [4, X4, Y4, Z4] tensor of partial_tsdf.pack_grid (Geometry.coarse() of the V3
        planner patch) -> h."""
        x, valid = PT.unpack_packed(packed.to(self.dev), self.dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            return self.model.encode(x, valid)

    @torch.no_grad()
    def cond(self, h, views):
        """views: [(o [3], d [R, 3], tcap [R], pr)] -> (pack, m_unk [V, 64, 128]); pack = (cond [V, C, 64, 128], mem or None)."""
        o = torch.stack([v[0] for v in views]).float()
        d = torch.stack([v[1] for v in views]).float()
        tc = torch.stack([v[2] for v in views]).float()
        ch = torch.stack([G.channels(v[3], v[2], v[1], v[0]) for v in views])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.autocast):
            out = self.model.view_cond(h, o, d, tc, ch)
        pack = out if self.model.xa else (out, None)
        m_unk = M.pad_rays(torch.stack([v[3]["cls"] == G.CLS_UNKNOWN for v in views]))
        return pack, m_unk

    @torch.no_grad()
    def sample(self, pack, m_unk, eps, return_features=False):
        """pack from cond(); eps [K, 2, 64, 128] shared by all views -> b, y [V, K, R] float32 (+ features)."""
        cond, mem = pack
        V, K = cond.shape[0], eps.shape[0]
        idx = torch.arange(V * K, device=cond.device)
        vi, ki = idx // K, idx % K
        xs, fs = [], []
        for s in range(0, V * K, self.max_views):
            sl = slice(s, min(V * K, s + self.max_views))
            r = V3.sample(self.model, cond[vi[sl]], m_unk[vi[sl]], eps[ki[sl]], V3.index_mem(mem, vi[sl]), self.steps, self.autocast, return_features)
            if return_features:
                xs.append(r[0]); fs.append(r[1].float())
            else:
                xs.append(r)
        x = M.unpad_rays(torch.cat(xs)).reshape(V, K, 2, -1)
        if return_features:
            return x[:, :, 0], x[:, :, 1], torch.cat(fs).reshape(V, K, *fs[0].shape[1:])
        return x[:, :, 0], x[:, :, 1]


def load_core(ckpt_path, device, steps=10, expect_sha=None):
    """EMA checkpoint of the trainer -> (model, predictor, sha256)."""
    import hashlib
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    if expect_sha is not None and sha != expect_sha:
        raise RuntimeError(f"checkpoint sha {sha} != expected {expect_sha}")
    ck = torch.load(ckpt_path, map_location=device)
    model = V3.V3Model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, V3Predictor(model, device, steps), sha
```

噪声：

文件：`seen2scene/futureview/predictor.py` 51–58

```python
def noise(device, K, seed_fn):
    """eps [K, 2, 64, 128] from per-k seeds seed_fn(k)."""
    out = []
    for k in range(K):
        g = torch.Generator(device=device)
        g.manual_seed(seed_fn(k) % (2 ** 63))
        out.append(torch.randn(2, M.GH, M.GW_PAD, generator=g, device=device))
    return torch.stack(out)
```

种子键 `seed_key(scene, start, rep, t_r, k)`，`t_r` 是最近一次刷新的时刻（R10：0, 10, 20, …），所以同一刷新块内每一步用同一组噪声。

### 5.3 解码：(b, y) → 首次命中距离

`b > 0 ∧ y < 1` 的射线视为在未知区域命中；`s = (y + 1) / 2` 映射到该射线第 `⌊s · n_u⌋` 个 UNKNOWN 样本的位置；其余射线取投影结果（已知表面 `t_known` 或无命中）。

文件：`seen2scene/futureview/geometry.py` 194–218

```python
def decode(b_hat, y_hat, o, d, pr, struct, domain, chunk=2048, eps=1e-6):
    """Hits of K samples (b_hat, y_hat [K, R]) -> ray distance t [K, R] (model units, inf = no hit), hit_in_unknown
    [K, R]. Non-M_unknown rays and samples without an UNKNOWN hit take the projection outcome (t_known or no hit)."""
    K, R = b_hat.shape
    tk = pr["t_known"]
    t_out = tk[None].expand(K, R).clone()
    hit_u = (b_hat > 0) & (y_hat < 1.0) & (pr["n_u"] > 0)[None]
    s = ((y_hat.to(tk.dtype) + 1.0) / 2.0).clamp(0.0, 1.0 - eps)
    for st in range(0, R, chunk):
        sl = slice(st, min(R, st + chunk))
        hs = hit_u[:, sl]
        if not hs.any():
            continue
        unk, ts = unknown_samples(o, d[sl], pr["tmin"][sl], pr["tmax"][sl], pr["valid"][sl], pr["M"], tk[sl],
                                  struct, domain)
        cs = torch.cumsum(unk.long(), dim=1)                                 # [r, M]
        n_u = pr["n_u"][sl]
        i = torch.floor(s[:, sl] * n_u[None].to(s.dtype)).long().clamp(min=0)
        i = torch.minimum(i, (n_u[None] - 1).clamp(min=0))                    # [K, r]
        j = torch.searchsorted(cs, (i + 1).T.contiguous())                   # [r, K]: first sample with cs >= i + 1
        j = j.clamp(max=pr["M"] - 1)
        t_hat = ts.gather(1, j).T                                            # [K, r]
        t_out[:, sl] = torch.where(hs, t_hat, t_out[:, sl])
    return t_out, hit_u
```

---

## 6. D1 规划器

### 6.1 闭环协议与规划器主循环

模拟器每步把观测记录（位姿、深度、10% 采样点、合法候选的 R / T）写成 `svc_<scene>_s<start>_t<t>.npz`，通过 JSON 行把路径发给规划器子进程；规划器只能读白名单内的记录（LeakGuard 禁止读取 GT 和评估侧结果），返回 `pick`。

文件：`experiments/exp5_future_view/task1/scripts/v2_planner.py` 146–227（主循环节选）

```python
        def refresh(t, geo):
            tr = time.time()
            torch.cuda.synchronize()
            coarse = geo.coarse()
            state["h"] = core.pred.encode(coarse)
            state["eps"] = PR.noise(dev, K, lambda kk: RY.sha_seed(LC.seed_key(s, k, rep, t, kk)))
            state["t_r"] = t
            torch.cuda.synchronize()
            r = {"t": t, "encode_s": time.time() - tr,
                 "coarse_sha256": VF.hashlib.sha256(coarse.cpu().numpy().tobytes()).hexdigest()}
            state["refreshes"].append(r)
            return r
        ...
        for line in sys.stdin:
            req = json.loads(line)
            if req.get("cmd") == "end":
                break
            t = int(req["t"])
            assert t == t_expect, (t, t_expect)
            t_expect += 1
            p = os.path.abspath(req["svc"])
            assert os.path.dirname(p) == svc_dir and os.path.basename(p) == f"svc_{s}_s{k}_t{t}.npz", p
            torch.cuda.synchronize()
            tic = time.time()
            rec = MC.load_service_record(p)
            assert int(rec["t"]) == t and str(rec["scene_key"]) == f"{s}/{k}"
            fu.integrate_record(rec)
            assert fu.n_frames == t + 1, (fu.n_frames, t)
            torch.cuda.synchronize()
            t_p1 = time.time()
            geo = PC.Geometry.from_fusion(fr, dims, dom, fu, dev)
            torch.cuda.synchronize()
            t_geo = time.time()
            rinfo = None
            if LC.refresh_time(a.exp, t) == t:
                rinfo = refresh(t, geo)
            n = len(rec["legal_R"])
            step = {"t": t, "n": n, "t_r": state["t_r"]}
            sc_all, tm = {}, {}
            if n == 0:
                scores, ch = [], {"pick": None, "fallback": "no_legal_action"}
            else:
                out, tm, _ = PC.score_state(core, geo, state["h"], rec["legal_R"], rec["legal_T"], state["eps"], z0,
                                            deltas=(delta,) if need_u else ())
                sc_all = {kk: v for kk, v in out.items()}
                if E["score"] == "distp1":
                    scores = out["distp1"]
                    ch = MS.choose(scores, s, k, t, True)
                ...
            send({"t": t, "pick": ch["pick"], "fallback": ch.get("fallback"), "decision_s": dt, "scores": scores})
```

每步流程：`P1 融合 → Geometry（当前地图的 struct / domain / 射线投射器）→ [R10 刷新：编码 + 噪声] → score_state（§6.3）→ D1 策略（§6.4，通过替换 `PC.score_state` 接入）→ choose`。

### 6.2 候选生成（模拟器侧，MAGICIAN 规则）

设置网格上的 6 个平移 + 4 个旋转邻居 → 未访问优先 → GT 线段碰撞检测（模拟器侧，规划器看不到 GT）。

文件：`experiments/exp3b_macaronspp_transfer/m_magician_replay.py` 84–104

```python
def legal_neighbours(camera, mesh, intersector):
    """Beam layer 1 of magician_planning.py: get_neighboring_poses -> get_valid_neighbors -> GT line-segment collision."""
    cur = camera.cam_idx
    nb = camera.get_neighboring_poses(pose_idx=cur)
    valid = camera.get_valid_neighbors(neighbor_indices=nb, mesh=mesh)
    cur_pose, _ = camera.get_pose_from_idx(cur)
    X_cur, _, _ = camera.get_camera_parameters_from_pose(cur_pose)
    loc = X_cur[0].cpu().numpy()
    out, flags = [], []
    for row in valid:
        pose, _ = camera.get_pose_from_idx(row)
        X, V, _ = camera.get_camera_parameters_from_pose(pose)
        hit = bool(line_segment_mesh_intersection(loc, X[0].cpu().numpy(), intersector))
        flags.append({"idx": row.cpu().numpy().astype(int).tolist(), "collides": hit})
        if not hit:
            R, T = get_camera_RT(X, V)
            out.append((row.cpu().numpy().astype(np.int64), R[0].cpu().numpy(), T[0].cpu().numpy(),
                        X[0].cpu().numpy()))
    return out, flags, [r.cpu().numpy().astype(int).tolist() for r in nb]
```

规划器侧展开第二层时用同样的邻居规则（没有 GT 碰撞检测，改用已知表面带的线段检测）：

文件：`experiments/exp6_v3/H/scripts/h_policy.py` 121–162

```python
SHIFTS = np.asarray([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -1, 0, 0],
                     [0, 0, 0, 1, 0], [0, 0, 0, -1, 0], [0, 0, 0, 0, 1], [0, 0, 0, 0, -1]], np.int64)


def neighbours(idx, grid):
    """MAGICIAN get_neighboring_poses: 6 moves + 4 rotations, clamp x / y / z / elevation, azimuth modulo, unique (rows
    sorted lexicographically, as torch.unique(dim=0))."""
    r = np.asarray(idx, np.int64)[None] + SHIFTS
    r[:, :4] = np.maximum(r[:, :4], 0)
    for a, n in enumerate(("pose_l", "pose_w", "pose_h", "pose_n_elev")):
        r[:, a] = np.minimum(r[:, a], grid[n] - 1)
    r[:, 4] = r[:, 4] % grid["pose_n_azim"]
    return np.unique(r, axis=0)


def segment_blocked(geo, c_from, c_to, step_vox=0.5):
    """True if the straight segment between two camera centres (runtime) crosses a voxel of the known D100 surface band
    of the current P1 (inference-available geometry; unknown space counts as free)."""
    a = np.asarray(geo.fr.runtime_to_model(np.asarray(c_from, np.float64)[None]))[0]
    b = np.asarray(geo.fr.runtime_to_model(np.asarray(c_to, np.float64)[None]))[0]
    L = np.linalg.norm(b - a)
    if L == 0:
        return False
    n = int(np.ceil(L / (0.011 * step_vox))) + 1
    pts = a[None] + (b - a)[None] * np.linspace(0, 1, n)[:, None]
    ijk = np.floor(pts / 0.011).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < np.asarray(geo.dims)[None]), 1)
    if not ok.any():
        return False
    ijk = torch.as_tensor(ijk[ok], device=geo.caster.obs_band.device)
    return bool(geo.caster.obs_band[ijk[:, 0], ijk[:, 1], ijk[:, 2]].any())
```

### 6.3 一步价值

**(a) 冻结的 `score_state`**：对所有合法候选做投影 → 条件 → K 个样本 → 解码为首次命中 → DistP1。

文件：`experiments/exp5_future_view/task1/scripts/v2_plan_core.py` 69–130（节选）

```python
def score_state(core, geo, h, legal_R, legal_T, eps, z0, deltas=(), want_flow=False, want_feat=False,
                keep_hits=False, jitter=(0.0, 0.0)):
    """eps [K, 2, 64, 128] (shared by all candidates). Returns dict of per-action lists and the timing parts."""
    tm = {}
    t0 = sync()
    views = [geo.view(R, T, jitter) for R, T in zip(legal_R, legal_T)]
    ...
    cond, m_unk = core.pred.cond(h, [(v["o"], v["d"], v["tcap"], v["pr"]) for v in views])
    ...
    r = core.pred.sample(cond, m_unk, eps, return_features=want_feat)
    b, y = r[0], r[1]
    ...
    k_M = geo.fr.k_M
    for m, v in enumerate(views):
        t_dec, hit_u = G.decode(b[m], y[m], v["o"], v["d"], v["pr"], geo.struct, geo.domain)
        ws, news = [], []
        for k in range(K):
            ids, wv = SCO.new_set_from_depth(t_dec[k], v["o"], v["d"], v["dz"], k_M, geo.dims, geo.domain,
                                            geo.caster.band_dil, z0)
            ws.append(float(wv.sum()))
            ...
        out["distp1"].append(float(np.mean(ws)))
        out["distp1_k"].append(ws)
        ...
        if keep_hits:
            out["hits"].append({"t": t_dec, "hit_u": hit_u, "b": b[m], "y": y[m]})
    tm["decode_score_s"] = sync() - t3
    return out, tm, views
```

**DistP1**：预测命中点所在体素若不在已观测的 BAND 膨胀带里就算“新”，按距离加权 `w = min((z / z0)^2, 1)`（`z0` = 场景尺度 / 2），每个新体素只计一次。

文件：`seen2scene/futureview/scoring.py` 14–41

```python
def new_mask_from_depth(t, o, d, dims, domain, band_dil):
    """Per-ray new-hit mask, flat voxel index, hit voxel (SceneCaster.cast post-processing; t [R], inf = no hit)."""
    hit = torch.isfinite(t)
    p_hit = o[None] + d * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p_hit / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, dims[a] - 1)
    in_dom = domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    hit = hit & in_dom
    new = hit & ~band_dil[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    flat = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    return new, flat, hit


def new_set_from_depth(t, o, d, dz, k_M, dims, domain, band_dil, z0=None):
    """(ids int64 sorted, weights float64) of the unique new voxels of one view (abl_score.new_set semantics)."""
    new, flat, _ = new_mask_from_depth(t, o, d, dims, domain, band_dil)
    sel = new
    f = flat[sel]
    if f.numel() == 0:
        return f.new_zeros(0), torch.zeros(0, dtype=torch.float64, device=f.device)
    ids, inv = torch.unique(f, return_inverse=True)
    if z0 is None:
        return ids, torch.ones(len(ids), dtype=torch.float64, device=ids.device)
    z = t[sel].double() * float(k_M) * dz[sel]
    w = torch.clamp((z / float(z0)) ** 2, max=1.0)
    wv = torch.zeros(len(ids), dtype=torch.float64, device=ids.device).scatter_reduce(0, inv, w, "amax", include_self=False)
    return ids, wv
```

**(b) D1 实际用的价值 `value_cov`**：与 DistP1 同样的命中点，但“新”的判定改为**离所有已观测的 10% 采样点都超过 1 m（KD-tree 查询）**，并按 1 m 的格子去重（这更接近官方覆盖的口径）。

文件：`experiments/exp6_v3/H/scripts/h_policy.py` 74–102

```python
def value_cov(t, v, geo, z0, cfg):
    """(cell ids int64 sorted unique, weights float64) of the predicted hits of one view / sample that are in the planning
    domain and farther than eps_cov from every observed S10 sample point (runtime frame)."""
    VS = 0.011
    hit = torch.isfinite(t)
    p = v["o"][None] + v["d"] * torch.where(hit, t, torch.zeros_like(t))[:, None]
    ijk = torch.floor(p / VS).long()
    for a in range(3):
        ijk[:, a] = ijk[:, a].clamp(0, geo.dims[a] - 1)
    hit = hit & geo.domain[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    sel = torch.nonzero(hit)[:, 0]
    if sel.numel() == 0:
        return np.zeros(0, np.int64), np.zeros(0)
    pm = p[sel].double().cpu().numpy()
    pr = geo.fr.model_to_runtime(pm)                                         # runtime (y-up) frame, float64
    eps = float(cfg["eps_cov"])
    dist, _ = obs_tree().query(pr, k=1, distance_upper_bound=eps)
    novel = ~np.isfinite(dist)                                               # no observed point within eps
    if not novel.any():
        return np.zeros(0, np.int64), np.zeros(0)
    pr = pr[novel]
    z = (t[sel].double() * float(geo.fr.k_M) * v["dz"][sel].double()).cpu().numpy()[novel]
    w = np.minimum((z / float(z0)) ** 2, 1.0)
    c = np.floor(pr / float(cfg["cell_size"])).astype(np.int64) + (1 << 19)
    key = (c[:, 0] << 40) | (c[:, 1] << 20) | c[:, 2]
    ids, inv = np.unique(key, return_inverse=True)
    wv = np.zeros(len(ids))
    np.maximum.at(wv, inv, w)
    return ids, wv
```

### 6.4 两步前瞻（D1 = depth 2，穷举，λ_risk = 0）

算法：

1. 对每个合法候选 c1，用冻结的 `score_state` 得到 K 个样本的新覆盖集合 `pred[c1][k]`；一步分数 `s_one = mean_k |set_k|`。
2. 对每个 c1，展开它在设置网格上的邻居 c2（未访问优先；排除路径上已有的；穿过已知表面带的线段丢弃）；对这些 c2 位姿也做一次预测（同一特征 h、同一组噪声，按 16 个一批缓存）。
3. 路径价值：对每个样本 k，取 c1 与 c2 的集合**并集**（每格取最大权重），`Q = mean_k − λ·std_k`（D1 的 λ = 0，即样本均值）。
4. c1 的分数 = 它所有两步路径中 Q 的最大值；没有可展开的 c2 时保留一步分数。
5. `choose`：取分数最大的候选（并列取最小下标；全 0 → 种子化随机回退）。

文件：`experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py` 26–110

```python
def q_of(sets_k, lam):
    vals = [float(s[1].sum()) for s in sets_k]
    return float(np.mean(vals)) - lam * float(np.std(vals))


def union_sets(a, b):
    """Per-sample union (ids, weights) of two cell sets (max weight per cell)."""
    if len(b[0]) == 0:
        return a
    if len(a[0]) == 0:
        return b
    ids = np.concatenate([a[0], b[0]])
    w = np.concatenate([a[1], b[1]])
    u, inv = np.unique(ids, return_inverse=True)
    wv = np.zeros(len(u))
    np.maximum.at(wv, inv, w)
    return u, wv


def lookahead(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats):
    """Per-root beam lookahead. sets: per legal view, per sample (ids, weights) of the frozen score_state predictions."""
    rec, P = HP.STATE["rec"], HP.STATE["pose"]
    legal = [tuple(int(x) for x in r) for r in np.asarray(rec["legal_idx"]).reshape(-1, 5)]
    depth, beam, lam = int(pc["depth"]), pc.get("beam"), float(pc["lambda_risk"])
    K = eps.shape[0]
    pred = {c: sets[i] for i, c in enumerate(legal)}
    vcfg = {"eps_cov": pc["eps_cov"], "cell_size": pc["cell_size"]}

    def ensure(poses):
        need = [c for c in dict.fromkeys(poses) if c not in pred]
        chunk = int(pc.get("c2_chunk", 16))
        for s0 in range(0, len(need), chunk):
            part = need[s0:s0 + chunk]
            o2, _, v2 = orig_score(core, geo, h, [P["R"][P["lut"][c]] for c in part], [P["T"][P["lut"][c]] for c in part], eps, z0, keep_hits=True)
            for q, c in enumerate(part):
                tk = o2["hits"][q]["t"]
                pred[c] = [HP.value_cov(tk[k], v2[q], geo, z0, vcfg) for k in range(K)]
            del o2, v2
        stats["predicted_views"] += len(need)

    def children(path):
        vis = HP.STATE["visited"] | set(path)
        nb = [tuple(int(x) for x in r) for r in HP.neighbours(path[-1], P["grid"])]
        nb = [n for n in nb if n not in path]
        new = [n for n in nb if n not in vis]
        cand = new if new else nb
        out = []
        for c in cand:
            if HP.segment_blocked(geo, P["C"][P["lut"][path[-1]]], P["C"][P["lut"][c]]):
                stats["blocked"] += 1
                continue
            out.append(c)
        return out

    score = np.asarray(s_one, float).copy()
    per_root = []
    # layer 2 for all roots first, predictions requested in the same order / chunks as H3 (bitwise reproduction of H3)
    ch2 = [children((c1,)) for c1 in legal]
    ensure([c for lst in ch2 for c in lst])
    for i, c1 in enumerate(legal):
        frontier = [((c1,), pred[c1])]                # (path, per-sample union sets)
        best, best_path, nodes = None, None, 0
        for L in range(2, depth + 1):
            exp = []
            for path, U in frontier:
                ch = ch2[i] if L == 2 else children(path)
                if L > 2:
                    ensure(ch)
                for c in ch:
                    U2 = [union_sets(U[k], pred[c][k]) for k in range(K)]
                    exp.append((path + (c,), U2, q_of(U2, lam)))
            nodes += len(exp)
            if not exp:
                break
            layer_best = max(exp, key=lambda e: e[2])  # first maximal element (generation order) on ties
            best, best_path = layer_best[2], layer_best[0]
            if L < depth:
                order = sorted(range(len(exp)), key=lambda j: -exp[j][2])   # stable: ties keep generation order
                keep = order if beam is None else order[:int(beam)]
                frontier = [(exp[j][0], exp[j][1]) for j in keep]
        if best is not None:
            score[i] = best
        per_root.append({"c1": list(c1), "nodes": nodes, "best_path": [list(p) for p in best_path] if best_path else None, "q": best, "s_one": float(s_one[i])})
    return score, per_root
```

策略接入方式（替换 `PC.score_state`，冻结的打分函数原样运行，只把 `out["distp1"]` 换成前瞻分数）：

文件：`experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/scripts/d_policy.py` 139–165

```python
    orig_score = PC.score_state
    lam = float(pc["lambda_risk"])

    def score_state(core, geo, h, legal_R, legal_T, eps, z0, *a, **kw):
        want_hits = kw.pop("keep_hits", False)
        tt = time.time()
        out, tm, views = orig_score(core, geo, h, legal_R, legal_T, eps, z0, *a, keep_hits=True, **kw)
        t_frozen = time.time() - tt
        if not views:
            if not want_hits:
                out.pop("hits", None)
            return out, tm, views
        K = eps.shape[0]
        sets = [[HP.value_cov(out["hits"][m]["t"][k], v, geo, z0, pc) for k in range(K)] for m, v in enumerate(views)]
        g = np.asarray([[float(s_[1].sum()) for s_ in row] for row in sets])
        mu = np.asarray([float(np.mean(list(r))) for r in g])
        sd = g.std(1)
        s_one = mu - lam * sd
        stats = {"predicted_views": 0, "blocked": 0}
        score, per_root = lookahead(core, geo, h, eps, z0, sets, s_one, pc, orig_score, stats)
        out["h0_distp1"] = list(out["distp1"])
        out["h_value_k"] = g.tolist()
        out["h_mean"], out["h_std"], out["d_one_step"] = mu.tolist(), sd.tolist(), s_one.tolist()
        out["d"] = {"per_root": per_root, **stats}
        out["distp1"] = [float(x) for x in score]
        tm = dict(tm, frozen_score_state_s=t_frozen, policy_s=time.time() - tt - t_frozen)
        if not want_hits:
            out.pop("hits", None)
        return out, tm, views
    PC.score_state = score_state
```

D1 的冻结配置（`experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/configs/d_config.yaml`，节选）：

```yaml
inference: "same as H-v1 / Table 1 V3 (b2: D100 input + S10 band_dil, R10, K 4, Euler 10, frozen noise keys shared by all candidates, simulator legal list, m_score.choose, 100 steps)"
value: {kind: cov, eps_cov: 1.0, cell_size: 1.0, novelty: hard}   # = H1 / H3 per-sample value (h_policy.value_cov)
policies:
  d1: {depth: 2, beam: null, lambda_risk: 0.0, doc: "D1_H3_NO_RISK: H3 with lambda 0"}
search:
  layer1: "all simulator-legal candidates c1 (no pruning)"
  layer2: "settings-grid neighbours of c1 (get_neighboring_poses), unvisited first with history + path poses counted visited, path poses excluded, segment crossing the known D100 surface band dropped; exhaustive (= H3)"
  value_of_path: "per sample k: union of the novel cells of all poses of the path (max weight per cell); Q = mean_k - lambda std_k (ddof 0)"
  score_c1: "max over all paths of root c1 at the deepest layer reached (a path that cannot be extended keeps its last Q); no layer-2 child -> the one-step Q of c1"
  samples: "sample k of every pose uses noise map k of the step (frozen keys) -> paths pair sample k across poses (Monte-Carlo pairing)"
  cache: "predictions of a pose computed once per step (same features h, same noise) and reused by every path"
  c2_chunk: 16
```

### 6.5 选择规则

文件：`experiments/exp3b_macaronspp_transfer/m_score.py` 129–135；`m_common.py` 59–69

```python
def choose(scores, scene: str, start: int, t: int, worlds_valid: bool = True) -> Dict:
    s = np.asarray(scores, dtype=np.float64)
    if not worlds_valid:
        return {"pick": MC.fallback_choice(scene, start, t, len(s)), "fallback": "world_invalid"}
    if s.max() == 0.0:
        return {"pick": MC.fallback_choice(scene, start, t, len(s)), "fallback": "all_zero"}
    return {"pick": MC.argmax_lowest(s), "fallback": None}
```

```python
def fallback_choice(scene: str, start: int, t: int, n: int) -> int:
    """S1-M / S2-M Random fallback (v1.3 E): seeded uniform choice among the n legal actions (same form as R's
    MacaronsRandomPlanner: default_rng([2027, key_hash(key), episode 0, step, 7]).random(n), argmax)."""
    r = np.random.default_rng([2027, key_hash(world_key(scene, start)), 0, int(t), 7]).random(int(n))
    return int(np.argmax(r))


def argmax_lowest(scores) -> int:
    """Deterministic tie-breaking: the lowest index of the maximum (numpy argmax semantics, made explicit)."""
    s = np.asarray(scores, dtype=np.float64)
    return int(np.flatnonzero(s == s.max())[0])
```

### 6.6 覆盖评估

模拟器每步记录 MAGICIAN 官方覆盖（`gt_scene.scene_coverage(covered, surface_epsilon)`，被覆盖格子有上限、点可被替换，所以曲线可能下降，不取 cummax）。评估后用 `t25_out_cov.py` 以相同种子重放执行过的位姿，重算的 `cov_official[0..100]` 必须与存档曲线逐位一致，否则 STOP。指标：`AUC@k = trapz(C[0..k]) / k`（k = 20, 50, 100）、`Cov@k = C[k]`、`Final = C[100]`；场景内 5 个起点平均，再场景等权。

---

## 7. 关键数值

### 7.1 张量形状（一个状态 + V 个候选）

| 张量 | 形状 |
|---|---|
| 4× 输入 `x`, `valid` | `[1, 3, X4, Y4, Z4]`, `[1, 1, X4, Y4, Z4]`（dunnottar：278 × 398 × 64） |
| `μ`, `logvar`, `valid8` | `[1, 16, X4/2, Y4/2, Z4/2]` |
| 射线 `o`, `d`, `tcap` | `[V, 3]`, `[V, 7296, 3]`, `[V, 7296]` |
| 投影通道 `ch` | `[V, 11, 7296]` |
| 射线特征 `rf` | `[V·7296, 64]` |
| `cond` | `[V, 76, 64, 128]` |
| 场景记忆（T1 专有） | T0 没有：`mem = None` |
| U-Net 输入 / 输出 | `[V, 78, 64, 128]` / `[V, 2, 64, 128]` |
| 采样结果 `(b, y)` | `[V, K=4, 7296]` 各一 |
| 解码的首次命中 `t` | `[K, 7296]`（inf = 无命中） |

### 7.2 Table 1（All-14，官方覆盖，场景等权）

| 行 | AUC@20 | Cov@20 | AUC@50 | AUC@100 | Final |
|---|---|---|---|---|---|
| **V3-T0 + D1**（本文核心，Table 1 行 `V3-T0-D1`） | 2026-11-08 补跑中（19:54 时 31 / 70；同起点配对中期：与 T1 + D1 的差在 ±0.006 以内），跑完后更新此行 | | | | |
| V3-T1 + D1（对照，`V3-D1-H3-NORISK`） | 0.425 | 0.654 | 0.638 | 0.765 | 0.910 |
| V3-T1 + H0（一步 DistP1） | 0.384 | 0.603 | 0.604 | 0.744 | 0.903 |
| B0（V2-Mix + H0） | 0.403 | 0.633 | 0.618 | 0.748 | 0.897 |
| MAGICIAN Beam | 0.307 | 0.537 | 0.544 | 0.711 | 0.906 |

D1 对 H0 的增益（同一网络、只换规划器，T1 上测得）：AUC@20 +0.041、AUC@100 +0.021。T0 + H0 只在 EXP6 V3 的 G3（Dev-3 + Test-3，30 条，非 Table 1 协议）上跑过，T1 − T0 在那里 Macarons++ ΔFinal +0.011 [+0.005, +0.017]、MP3D ΔAUC +0.020 [−0.003, +0.042]；按 Task A 的结论，这个差别不能归因于交叉注意力。

### 7.3 推理开销（D1，Table 1 存档）

| 项 | 值 |
|---|---|
| 规划器每步决策时间 | 4.7–6.8 s（场景不同；两步前瞻每步要预测 c1 加全部 c2，缓存复用） |
| 每条室外 episode（100 步） | 9–13 分钟 |
| 规划器显存峰值 | 约 2.7 GB（barts）到 31 GB（dunnottar，整图编码） |

---

## 8. 文件索引

| 路径 | 内容 |
|---|---|
| `experiments/exp6_v3/protocol.md` | V3 协议 v1.0（数据、模型、损失、关口、公平性） |
| `experiments/exp6_v3/configs/{model_v3,train_v3}.json` | 模型与训练配置（冻结） |
| `experiments/exp6_v3/shared/train_arm.sh t0` → `task0_vae/scripts/train.py` | 训练启动 |
| `seen2scene/v3/trainer.py` | 训练循环、验证、存档、选模规则 |
| `seen2scene/v3/data.py` | 分片 + 4× 缓存加载、采样、`build_cond` |
| `seen2scene/v3/partial_tsdf.py` | 4× 聚合、打包 / 解包、重建目标 |
| `seen2scene/v3/model.py` | `V3Model`、`LatentRayEncoder`、`sample` |
| `seen2scene/v3/vae.py` | VAE 编码器 / 解码器、KL |
| `seen2scene/v3/flow_unet.py` | `FlowUNet`（U-Net；T0 时 `xattn=None`） |
| `seen2scene/v3/cross_attention.py` | `SceneMemory`、`query_pe`、`CrossAttnBlock`（仅 T1 实例化，T0 不用） |
| `seen2scene/v3/losses.py` | `L_FM`、`L_recon`、KL、β 调度 |
| `seen2scene/v3/eval_dev.py` | 验证指标（fair ES 等） |
| `seen2scene/v3/predictor.py` | 规划器侧预测器 `V3Predictor`、`load_core` |
| `seen2scene/futureview/model.py` | V2 的 `RayEncoder`、`Res2D`、`Attn2D`、`UNet2D`、`pad_rays` |
| `seen2scene/futureview/rays.py` | 打分射线与相机模型 |
| `seen2scene/futureview/geometry.py` | 投影通道、标签编码、解码 |
| `seen2scene/futureview/scoring.py` | DistP1 |
| `experiments/exp3b_macaronspp_transfer/m_p1.py`、`task3_full_depth_p1/scripts/p1_full.py` | P1 融合（10% / D100） |
| `experiments/exp3b_macaronspp_transfer/{m_score,m_common,m_magician_replay,m_t6_sim}.py` | 选择规则、候选生成、模拟器 |
| `experiments/exp5_future_view/task1/scripts/{v2_sim,v2_planner,v2_plan_core}.py` | 闭环模拟器、规划器子进程、`score_state` |
| `experiments/exp5_future_view/task2/scripts/{t24_out_sim,t24_v2_out_planner,t25_out_cov}.py` | 室外入口、官方覆盖重放 |
| `experiments/exp6_v3/shared/{sim_out_v3,v3_out_planner,planner_v3}.py` | V3 的闭环接入 |
| `experiments/exp6_v3/H/scripts/h_policy.py` | `value_cov`、邻居规则、线段检测 |
| `experiments/exp6_v3/ablation_study/experiment_D_h3_lookahead/{scripts/d_policy.py,configs/d_config.yaml}` | D1 两步前瞻 |
| `experiments/cvpr_results/table1/` | Table 1 协议、记录、聚合脚本 |
| `work_dirs/exp6_v3/task0_vae/{checkpoints/final.pt,runs/main/}` | T0@40K 权重（= `work_dirs/checkpoints/exp6_v3/T0_vae/v3t0_ema40000.pt`）、训练日志、验证记录、运行清单 |
| `work_dirs/cvpr_results/table1/episodes/V3-T0-D1/` | T0 + D1 的 Table 1 规范记录（官方覆盖曲线，重放核对） |
| `experiments/exp6_v3/figures/V3_T0_D1_architecture.{pdf,png}`、`v3_t0_d1_architecture.py` | 本文的架构图及生成脚本 |
| `V3_T1_D1_MODEL_ARCHITECTURE.md` | T1 版本的同名文档（含 §3.7 交叉注意力的完整代码） |
