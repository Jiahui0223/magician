# Checkpoint 索引（V2 B2/E2 · V3 · V3.1 · V4 · V5 / V4C · V5.2 · V5.2-M）

由 `experiments/checkpoint_registry.py` 于 2026-11-02 18:56:03 生成。`work_dirs/checkpoints/<实验>/` 中全部是指向原文件的相对软链接：原文件未移动、复制或删除，冻结配置和旧脚本中的路径继续有效；每个链接都核对了 sha256（读穿链接）、step、模型配置，并对 EMA 权重做了严格加载（strict=True）。G0 测试运行（resume_A / resume_B、smoke、runs/g0）不登记。

## exp6_v2

V2 family: B2 (frozen V2-Mix, D100 input / S10 DistP1) and E2 (V2-Mix fine-tuned on D100 data)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v2/B2/v2mix_ema17500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-1_V2Mix_best/ema_017500.pt` | ema | 17500 | 24,822,114 | 343f41ab8d566c52 ✓ | ok | B2 MODEL: V2-Mix EMA@17.5k (frozen-rule selection; = Task 2 group V2M, B0 / B2 planners) |
| `exp6_v2/B2/v2mix_ema20000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-2_V2Mix_last/ema_020000.pt` | ema | 20000 | 24,822,114 | 5c68720ea6c64f1a | ok | V2-Mix last EMA (not selected) |
| `exp6_v2/B2/v2mix_raw20000.pt` | `work_dirs/exp5_future_view/task2/training/mix_v1/ckpt_last.pt` | raw | 20000 | 24,822,114 | 9805f081da3ac0d2 | model≡EMA 布局，含优化器 | V2-Mix raw training state at 20k (model + EMA + AdamW + RNG) |
| `exp6_v2/B2/lineage/v2orig_ema60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ema_060000.pt` | ema | 60000 | 24,822,114 | a379c0fb9d7d8026 ✓ | ok | V2 Original EMA@60k (Task 1; init of V2-Mix) |
| `exp6_v2/B2/lineage/v2orig_raw60000.pt` | `work_dirs/weights/W20_exp5_v2/W20-1_V2_original_ema60k/ckpt_last.pt` | raw | 60000 | 24,822,114 | 7e827ecc2ce5360c | model≡EMA 布局，含优化器 | V2 Original raw training state at 60k |
| `exp6_v2/B2/intermediate/v2mix_ema000000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_000000.pt` | ema | 0 | 24,822,114 | 607882ec6bae82c3 | ok | intermediate EMA @ 0 |
| `exp6_v2/B2/intermediate/v2mix_ema002500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_002500.pt` | ema | 2500 | 24,822,114 | 4e56981ef3b93077 | ok | intermediate EMA @ 2500 |
| `exp6_v2/B2/intermediate/v2mix_ema005000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_005000.pt` | ema | 5000 | 24,822,114 | ba99081a99749f6d | ok | intermediate EMA @ 5000 |
| `exp6_v2/B2/intermediate/v2mix_ema007500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_007500.pt` | ema | 7500 | 24,822,114 | 103486e06b51ceba | ok | intermediate EMA @ 7500 |
| `exp6_v2/B2/intermediate/v2mix_ema010000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_010000.pt` | ema | 10000 | 24,822,114 | 32bb5548d2abd59b | ok | intermediate EMA @ 10000 |
| `exp6_v2/B2/intermediate/v2mix_ema012500.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_012500.pt` | ema | 12500 | 24,822,114 | 11df90ab648e51ba | ok | intermediate EMA @ 12500 |
| `exp6_v2/B2/intermediate/v2mix_ema015000.pt` | `work_dirs/weights/W21_exp5_v2mix/W21-3_V2Mix_intermediate_ema/ema_015000.pt` | ema | 15000 | 24,822,114 | c8a795dba55c5bbc | ok | intermediate EMA @ 15000 |
| `exp6_v2/E2/e2d100_ema5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_005000.pt` | ema | 5000 | 24,822,114 | 61439612cb6132af ✓ | ok | E2 MODEL: E2-D100 EMA@5k (V2-Mix EMA@17.5k fine-tuned 5k steps on D100 shards; B2-style planner) |
| `exp6_v2/E2/e2d100_raw5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ckpt_last.pt` | raw | 5000 | 24,822,114 | f647eb5f7a7cad51 | model≡EMA 布局，含优化器 | E2-D100 raw training state at 5k |
| `exp6_v2/E2/control_e2s10/e2s10_ema5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_005000.pt` | ema | 5000 | 24,822,114 | 578855bb14995626 | ok | E2-S10 EMA@5k (control arm: same recipe on sparse S10 shards; B0-style planner) |
| `exp6_v2/E2/control_e2s10/e2s10_raw5000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ckpt_last.pt` | raw | 5000 | 24,822,114 | 2cb3f1357f027a67 | model≡EMA 布局，含优化器 | E2-S10 raw training state at 5k |
| `exp6_v2/E2/intermediate/e2d100_ema000000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_000000.pt` | ema | 0 | 24,822,114 | fd19fc33741e342a | ok | intermediate EMA @ 0 |
| `exp6_v2/E2/intermediate/e2d100_ema001000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_001000.pt` | ema | 1000 | 24,822,114 | 24d556983a525820 | ok | intermediate EMA @ 1000 |
| `exp6_v2/E2/intermediate/e2d100_ema002000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_002000.pt` | ema | 2000 | 24,822,114 | bc22a820b3d0ccdb | ok | intermediate EMA @ 2000 |
| `exp6_v2/E2/intermediate/e2d100_ema003000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_003000.pt` | ema | 3000 | 24,822,114 | 956b5e3d85b3ddcf | ok | intermediate EMA @ 3000 |
| `exp6_v2/E2/intermediate/e2d100_ema004000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/d100/ema_004000.pt` | ema | 4000 | 24,822,114 | a463ec9c78b71c38 | ok | intermediate EMA @ 4000 |
| （未单独链接，与 `exp6_v2/E2/intermediate/e2d100_ema000000.pt` 字节相同） | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_000000.pt` | ema | — | — | fd19fc33741e342a | — | intermediate EMA @ 0 |
| `exp6_v2/E2/control_e2s10/intermediate/e2s10_ema001000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_001000.pt` | ema | 1000 | 24,822,114 | cb514d47d550e729 | ok | intermediate EMA @ 1000 |
| `exp6_v2/E2/control_e2s10/intermediate/e2s10_ema002000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_002000.pt` | ema | 2000 | 24,822,114 | 60e238c34cac88f2 | ok | intermediate EMA @ 2000 |
| `exp6_v2/E2/control_e2s10/intermediate/e2s10_ema003000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_003000.pt` | ema | 3000 | 24,822,114 | 879f4bb5ac45ac45 | ok | intermediate EMA @ 3000 |
| `exp6_v2/E2/control_e2s10/intermediate/e2s10_ema004000.pt` | `work_dirs/exp5_future_view/task3_full_depth_p1/e2/training/s10/ema_004000.pt` | ema | 4000 | 24,822,114 | e6686f68c7203a8b | ok | intermediate EMA @ 4000 |

## exp6_v3

V3 (mixed scratch, VAE latent + ray encoder + 2D flow): T0 = V3-VAE, T1 = V3-VAE-XAttn

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v3/T0_vae/v3t0_ema40000.pt` | `work_dirs/exp6_v3/task0_vae/checkpoints/final.pt` | ema | 40000 | 23,539,622 | 7ed0ddd72bb91d1e ✓ | ok | V3-T0 final (EMA@40k) |
| `exp6_v3/T0_vae/v3t0_raw40000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ckpt_last.pt` | raw | 40000 | 23,539,622 | 063471a8ffbeac0a | model≡EMA 布局，含优化器 | V3-T0 raw training state at 40k |
| `exp6_v3/T0_vae/intermediate/v3t0_ema000000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_000000.pt` | ema | 0 | 23,539,622 | 987f848c34ee55e0 | ok | intermediate EMA @ 0 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema005000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_005000.pt` | ema | 5000 | 23,539,622 | 6befc76473ea297b | ok | intermediate EMA @ 5000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema010000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_010000.pt` | ema | 10000 | 23,539,622 | 699d0a4cc6a73ab3 | ok | intermediate EMA @ 10000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema015000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_015000.pt` | ema | 15000 | 23,539,622 | 32d26db851f71460 | ok | intermediate EMA @ 15000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema020000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_020000.pt` | ema | 20000 | 23,539,622 | 45c1838d28655797 | ok | intermediate EMA @ 20000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema025000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_025000.pt` | ema | 25000 | 23,539,622 | 6295fff423ee9136 | ok | intermediate EMA @ 25000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema030000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_030000.pt` | ema | 30000 | 23,539,622 | 67a4fa201ecc7fb1 | ok | intermediate EMA @ 30000 |
| `exp6_v3/T0_vae/intermediate/v3t0_ema035000.pt` | `work_dirs/exp6_v3/task0_vae/runs/main/ema_035000.pt` | ema | 35000 | 23,539,622 | b49198c4cb126d7a | ok | intermediate EMA @ 35000 |
| （未单独链接，与 `exp6_v3/T0_vae/v3t0_ema40000.pt` 字节相同） | `work_dirs/exp6_v3/task0_vae/runs/main/ema_040000.pt` | ema | — | — | 7ed0ddd72bb91d1e | — | intermediate EMA @ 40000 |
| `exp6_v3/T1_xattn/v3t1_ema40000.pt` | `work_dirs/exp6_v3/task1_xattn/checkpoints/final.pt` | ema | 40000 | 23,730,986 | 639c9d3ae3f3c069 ✓ | ok | V3-T1 final (EMA@40k) |
| `exp6_v3/T1_xattn/v3t1_raw40000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ckpt_last.pt` | raw | 40000 | 23,730,986 | 3d81978206c88068 | model≡EMA 布局，含优化器 | V3-T1 raw training state at 40k |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema000000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_000000.pt` | ema | 0 | 23,730,986 | 7f0488ceb0068e82 | ok | intermediate EMA @ 0 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema005000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_005000.pt` | ema | 5000 | 23,730,986 | d18b985e9f14df69 | ok | intermediate EMA @ 5000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema010000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_010000.pt` | ema | 10000 | 23,730,986 | 6569bb769bcab4ee | ok | intermediate EMA @ 10000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema015000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_015000.pt` | ema | 15000 | 23,730,986 | 04ba18f70cdbedf1 | ok | intermediate EMA @ 15000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema020000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_020000.pt` | ema | 20000 | 23,730,986 | 2d40fa12f5ecb859 | ok | intermediate EMA @ 20000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema025000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_025000.pt` | ema | 25000 | 23,730,986 | 9c37a0440b60773d | ok | intermediate EMA @ 25000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema030000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_030000.pt` | ema | 30000 | 23,730,986 | 7401589576ec8896 | ok | intermediate EMA @ 30000 |
| `exp6_v3/T1_xattn/intermediate/v3t1_ema035000.pt` | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_035000.pt` | ema | 35000 | 23,730,986 | d67c7402ceec3527 | ok | intermediate EMA @ 35000 |
| （未单独链接，与 `exp6_v3/T1_xattn/v3t1_ema40000.pt` 字节相同） | `work_dirs/exp6_v3/task1_xattn/runs/main/ema_040000.pt` | ema | — | — | 639c9d3ae3f3c069 | — | intermediate EMA @ 40000 |

## exp6_v3_1

V3.1 (V3-T0 architecture, outdoor-only scratch)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v3_1/V3_1/v31_ema40000.pt` | `work_dirs/exp6_v3_1/checkpoints/final.pt` | ema | 40000 | 23,539,622 | 13df0a45df56d125 | ok | V3.1 final (EMA@40k) |
| `exp6_v3_1/V3_1/v31_raw40000.pt` | `work_dirs/exp6_v3_1/runs/main/ckpt_last.pt` | raw | 40000 | 23,539,622 | c5a4332e1b986aad | model≡EMA 布局，含优化器 | V3.1 raw training state at 40k |
| `exp6_v3_1/V3_1/intermediate/v31_ema000000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_000000.pt` | ema | 0 | 23,539,622 | 6088219da8610af8 | ok | intermediate EMA @ 0 |
| `exp6_v3_1/V3_1/intermediate/v31_ema001000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_001000.pt` | ema | 1000 | 23,539,622 | 275aea2338c14c6c | ok | intermediate EMA @ 1000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema005000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_005000.pt` | ema | 5000 | 23,539,622 | 69b764fc7d07abef | ok | intermediate EMA @ 5000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema010000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_010000.pt` | ema | 10000 | 23,539,622 | f7264b0b713166cd | ok | intermediate EMA @ 10000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema015000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_015000.pt` | ema | 15000 | 23,539,622 | 2ab39b8c1b278923 | ok | intermediate EMA @ 15000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema020000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_020000.pt` | ema | 20000 | 23,539,622 | 21da1a1dbb0f731d | ok | intermediate EMA @ 20000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema025000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_025000.pt` | ema | 25000 | 23,539,622 | a0827c738b63e41b | ok | intermediate EMA @ 25000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema030000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_030000.pt` | ema | 30000 | 23,539,622 | 77b54d7ab5508d72 | ok | intermediate EMA @ 30000 |
| `exp6_v3_1/V3_1/intermediate/v31_ema035000.pt` | `work_dirs/exp6_v3_1/runs/main/ema_035000.pt` | ema | 35000 | 23,539,622 | 882b732f6c7bd299 | ok | intermediate EMA @ 35000 |
| （未单独链接，与 `exp6_v3_1/V3_1/v31_ema40000.pt` 字节相同） | `work_dirs/exp6_v3_1/runs/main/ema_040000.pt` | ema | — | — | 13df0a45df56d125 | — | intermediate EMA @ 40000 |

## exp6_v4

V4 (V2 modules + ray-aligned cross-attention, outdoor-only scratch)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v4/V4/v4_ema40000.pt` | `work_dirs/exp6_v4/checkpoints/final.pt` | ema | 40000 | 24,847,544 | 1b31afffbe4da2d9 | ok | V4 final (EMA@40k) |
| `exp6_v4/V4/v4_raw40000.pt` | `work_dirs/exp6_v4/runs/main/ckpt_last.pt` | raw | 40000 | 24,847,544 | f114e9c52ff1547b | model≡EMA 布局，含优化器 | V4 raw training state at 40k |
| `exp6_v4/V4/intermediate/v4_ema000000.pt` | `work_dirs/exp6_v4/runs/main/ema_000000.pt` | ema | 0 | 24,847,544 | 2a3883b77af0f806 | ok | intermediate EMA @ 0 |
| `exp6_v4/V4/intermediate/v4_ema001000.pt` | `work_dirs/exp6_v4/runs/main/ema_001000.pt` | ema | 1000 | 24,847,544 | 80e31e83de17e9fd | ok | intermediate EMA @ 1000 |
| `exp6_v4/V4/intermediate/v4_ema005000.pt` | `work_dirs/exp6_v4/runs/main/ema_005000.pt` | ema | 5000 | 24,847,544 | 41cfcdfd66c327ac | ok | intermediate EMA @ 5000 |
| `exp6_v4/V4/intermediate/v4_ema010000.pt` | `work_dirs/exp6_v4/runs/main/ema_010000.pt` | ema | 10000 | 24,847,544 | 53cd5002c057eb98 | ok | intermediate EMA @ 10000 |
| `exp6_v4/V4/intermediate/v4_ema015000.pt` | `work_dirs/exp6_v4/runs/main/ema_015000.pt` | ema | 15000 | 24,847,544 | 76f58515c0b60d45 | ok | intermediate EMA @ 15000 |
| `exp6_v4/V4/intermediate/v4_ema020000.pt` | `work_dirs/exp6_v4/runs/main/ema_020000.pt` | ema | 20000 | 24,847,544 | f6e7cbc9735e8439 | ok | intermediate EMA @ 20000 |
| `exp6_v4/V4/intermediate/v4_ema025000.pt` | `work_dirs/exp6_v4/runs/main/ema_025000.pt` | ema | 25000 | 24,847,544 | 3282ee2e88df03d7 | ok | intermediate EMA @ 25000 |
| `exp6_v4/V4/intermediate/v4_ema030000.pt` | `work_dirs/exp6_v4/runs/main/ema_030000.pt` | ema | 30000 | 24,847,544 | ded804db55268fe8 | ok | intermediate EMA @ 30000 |
| `exp6_v4/V4/intermediate/v4_ema035000.pt` | `work_dirs/exp6_v4/runs/main/ema_035000.pt` | ema | 35000 | 24,847,544 | 9f1b7ee24711b2bb | ok | intermediate EMA @ 35000 |
| （未单独链接，与 `exp6_v4/V4/v4_ema40000.pt` 字节相同） | `work_dirs/exp6_v4/runs/main/ema_040000.pt` | ema | — | — | 1b31afffbe4da2d9 | — | intermediate EMA @ 40000 |

## exp6_v5

V5 (geometry-aware ray-to-scene memory + multi-surface belief, V4@40k warm-start) and V4C (V4 continued)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v5/V5/v5_ema12000.pt` | `work_dirs/exp6_v5/checkpoints/v5_final.pt` | ema | 12000 | 24,988,647 | 5ef30ac4fd0b4a17 ✓ | ok | V5-Warm12k final (EMA@12k) |
| `exp6_v5/V5/v5_raw12000.pt` | `work_dirs/exp6_v5/runs/v5/ckpt_last.pt` | raw | 12000 | 24,988,647 | 0c6d5f72f4b9ea8d | model≡EMA 布局，含优化器 | V5 raw training state at 12k |
| `exp6_v5/V5/intermediate/v5_ema000000.pt` | `work_dirs/exp6_v5/runs/v5/ema_000000.pt` | ema | 0 | 24,988,647 | 81e3c66daf20f1f2 | ok | intermediate EMA @ 0 |
| `exp6_v5/V5/intermediate/v5_ema002000.pt` | `work_dirs/exp6_v5/runs/v5/ema_002000.pt` | ema | 2000 | 24,988,647 | 41634c61b338b9cb | ok | intermediate EMA @ 2000 |
| `exp6_v5/V5/intermediate/v5_ema004000.pt` | `work_dirs/exp6_v5/runs/v5/ema_004000.pt` | ema | 4000 | 24,988,647 | 2190978ee9119383 | ok | intermediate EMA @ 4000 |
| `exp6_v5/V5/intermediate/v5_ema006000.pt` | `work_dirs/exp6_v5/runs/v5/ema_006000.pt` | ema | 6000 | 24,988,647 | 0e60e66bbee526d8 | ok | intermediate EMA @ 6000 |
| `exp6_v5/V5/intermediate/v5_ema008000.pt` | `work_dirs/exp6_v5/runs/v5/ema_008000.pt` | ema | 8000 | 24,988,647 | 284a834b9c397cbc | ok | intermediate EMA @ 8000 |
| `exp6_v5/V5/intermediate/v5_ema010000.pt` | `work_dirs/exp6_v5/runs/v5/ema_010000.pt` | ema | 10000 | 24,988,647 | 641497161802e360 | ok | intermediate EMA @ 10000 |
| （未单独链接，与 `exp6_v5/V5/v5_ema12000.pt` 字节相同） | `work_dirs/exp6_v5/runs/v5/ema_012000.pt` | ema | — | — | 5ef30ac4fd0b4a17 | — | intermediate EMA @ 12000 |
| `exp6_v5/V4C/v4c_ema12000.pt` | `work_dirs/exp6_v5/checkpoints/v4c_final.pt` | ema | 12000 | 24,847,544 | c4190f1d05dd29d9 ✓ | ok | V4-Continue12k final (EMA@12k; control of V5) |
| `exp6_v5/V4C/v4c_raw12000.pt` | `work_dirs/exp6_v5/runs/v4c/ckpt_last.pt` | raw | 12000 | 24,847,544 | 621f82213d6dc1e0 | model≡EMA 布局，含优化器 | V4C raw training state at 12k |
| `exp6_v5/V4C/intermediate/v4c_ema000000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_000000.pt` | ema | 0 | 24,847,544 | a43a8015dfcb4f62 | ok | intermediate EMA @ 0 |
| `exp6_v5/V4C/intermediate/v4c_ema002000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_002000.pt` | ema | 2000 | 24,847,544 | 3d8b5e67869594dd | ok | intermediate EMA @ 2000 |
| `exp6_v5/V4C/intermediate/v4c_ema004000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_004000.pt` | ema | 4000 | 24,847,544 | 0a3679946c07ce1c | ok | intermediate EMA @ 4000 |
| `exp6_v5/V4C/intermediate/v4c_ema006000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_006000.pt` | ema | 6000 | 24,847,544 | ec9504f22691a116 | ok | intermediate EMA @ 6000 |
| `exp6_v5/V4C/intermediate/v4c_ema008000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_008000.pt` | ema | 8000 | 24,847,544 | 4864a40cff719eb0 | ok | intermediate EMA @ 8000 |
| `exp6_v5/V4C/intermediate/v4c_ema010000.pt` | `work_dirs/exp6_v5/runs/v4c/ema_010000.pt` | ema | 10000 | 24,847,544 | 40ffab7ce8a444f0 | ok | intermediate EMA @ 10000 |
| （未单独链接，与 `exp6_v5/V4C/v4c_ema12000.pt` 字节相同） | `work_dirs/exp6_v5/runs/v4c/ema_012000.pt` | ema | — | — | c4190f1d05dd29d9 | — | intermediate EMA @ 12000 |

## exp6_v5_2

V5.2 pilot (Idea 2: ray geometry belief + conditional first-hit hazard, V5 warm-start, 2.67M)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v5_2/V5_2/v52_ema6000.pt` | `work_dirs/exp6_v5_2/checkpoints/v52_final.pt` | ema | 6000 | 2,668,710 | 4982bc6673687f69 ✓ | ok | V5.2 pilot final (EMA@6000) |
| `exp6_v5_2/V5_2/v52_raw6000.pt` | `work_dirs/exp6_v5_2/runs/v52/ckpt_last.pt` | raw | 6000 | 2,668,710 | 8f87762cd2e5ac76 | model≡EMA 布局，含优化器 | V5.2 raw training state at 6000 |
| `exp6_v5_2/V5_2/intermediate/v52_ema000000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_000000.pt` | ema | 0 | 2,668,710 | b9bfbf24f7d242b0 | ok | intermediate EMA @ 0 |
| `exp6_v5_2/V5_2/intermediate/v52_ema001000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_001000.pt` | ema | 1000 | 2,668,710 | bb551df24d227b97 | ok | intermediate EMA @ 1000 |
| `exp6_v5_2/V5_2/intermediate/v52_ema002000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_002000.pt` | ema | 2000 | 2,668,710 | 02242f4143f53b92 | ok | intermediate EMA @ 2000 |
| `exp6_v5_2/V5_2/intermediate/v52_ema003000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_003000.pt` | ema | 3000 | 2,668,710 | 9501abee66ccace9 | ok | intermediate EMA @ 3000 |
| `exp6_v5_2/V5_2/intermediate/v52_ema004000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_004000.pt` | ema | 4000 | 2,668,710 | ff9152aa03c03621 | ok | intermediate EMA @ 4000 |
| `exp6_v5_2/V5_2/intermediate/v52_ema005000.pt` | `work_dirs/exp6_v5_2/runs/v52/ema_005000.pt` | ema | 5000 | 2,668,710 | 021dd233f62b3c55 | ok | intermediate EMA @ 5000 |
| （未单独链接，与 `exp6_v5_2/V5_2/v52_ema6000.pt` 字节相同） | `work_dirs/exp6_v5_2/runs/v52/ema_006000.pt` | ema | — | — | 4982bc6673687f69 | — | intermediate EMA @ 6000 |

## exp6_v5_2_scratch70k

V5.2-M Scratch70k (Idea 2, 27.0M, from scratch)

| 新路径（work_dirs/checkpoints/…） | 原路径 | 类型 | step | 参数 | sha256（前 16） | 严格加载 | 用途 |
|---|---|---|---|---|---|---|---|
| `exp6_v5_2_scratch70k/V5_2M/v52m_best_ema50000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/best_train_holdout.pt` | ema | 50000 | 26,999,143 | a540950738cc7191 ✓ | ok | V5.2-M best_train_holdout (EMA@50k; primary checkpoint of the run) |
| `exp6_v5_2_scratch70k/V5_2M/v52m_ema70000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_070000.pt` | ema | 70000 | 26,999,143 | e1b168945323a21c ✓ | ok | V5.2-M last70k (EMA@70k) |
| `exp6_v5_2_scratch70k/V5_2M/v52m_raw70000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/last.pt` | raw | 70000 | 26,999,143 | eee203bbd9925e01 | model≡EMA 布局，含优化器 | V5.2-M raw training state at 70k |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema002000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_002000.pt` | ema | 2000 | 26,999,143 | 3f64a9548747f0a8 | ok | intermediate EMA @ 2000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema005000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_005000.pt` | ema | 5000 | 26,999,143 | a92d0af361d3105d | ok | intermediate EMA @ 5000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema010000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_010000.pt` | ema | 10000 | 26,999,143 | 985af03835ce3f3a | ok | intermediate EMA @ 10000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema015000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_015000.pt` | ema | 15000 | 26,999,143 | 521578f8bc5bd930 | ok | intermediate EMA @ 15000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema020000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_020000.pt` | ema | 20000 | 26,999,143 | 4f60f4f5b1a4ac02 | ok | intermediate EMA @ 20000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema025000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_025000.pt` | ema | 25000 | 26,999,143 | 217437a390880e7a | ok | intermediate EMA @ 25000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema030000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_030000.pt` | ema | 30000 | 26,999,143 | a863672c8898955d | ok | intermediate EMA @ 30000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema035000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_035000.pt` | ema | 35000 | 26,999,143 | 1f7e0ac595acb966 | ok | intermediate EMA @ 35000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema040000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_040000.pt` | ema | 40000 | 26,999,143 | 7379341c20894753 | ok | intermediate EMA @ 40000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema045000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_045000.pt` | ema | 45000 | 26,999,143 | 9ecd9c3c9330484f | ok | intermediate EMA @ 45000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema050000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_050000.pt` | ema | 50000 | 26,999,143 | c49511be95cb18eb | ok | intermediate EMA @ 50000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema055000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_055000.pt` | ema | 55000 | 26,999,143 | 29f081cd81c4cfee | ok | intermediate EMA @ 55000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema060000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_060000.pt` | ema | 60000 | 26,999,143 | 6387e4d043a362f9 | ok | intermediate EMA @ 60000 |
| `exp6_v5_2_scratch70k/V5_2M/intermediate/v52m_ema065000.pt` | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_065000.pt` | ema | 65000 | 26,999,143 | 2be0921438de4c53 | ok | intermediate EMA @ 65000 |
| （未单独链接，与 `exp6_v5_2_scratch70k/V5_2M/v52m_ema70000.pt` 字节相同） | `work_dirs/exp6_v5_2_scratch70k/checkpoints/step_070000.pt` | ema | — | — | e1b168945323a21c | — | intermediate EMA @ 70000 |

