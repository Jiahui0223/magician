# FisherRF-full · 暂停与恢复说明（FISHERRF_FULL_PAUSED）

**暂停时间**：2026-11-11，用户决定为主任务让出 GPU。
暂停时的状态：v3 试跑的 15 条 episode（dunnottar / redeemer / barts × 5）连同官方覆盖率回放都已跑完，结果见 `PILOT_FULL.md`。此后没有启动任何新一轮运行，没有 Slurm 作业，gpu114 上不留本工作的进程。
全量 14×5、MP3D、F1c 变体都**没有运行**。

## 1. 环境

| 项 | 说明 |
|---|---|
| 规划端 | fisherrf env `/bsuscratch/jiahuizhang/envs/fisherrf/bin/python`（torch 2.2.2+cu118，官方扩展编译为 sm_89，只能在 L40S 上跑）；open3d 用桩模块 `scripts/stubs/open3d`，详见 `ENVIRONMENT.md` |
| 仿真 / 评测端 | magician env `/bsuscratch/jiahuizhang/envs/magician/bin/python`，统一用 `env -u LD_PRELOAD -u PYTHONPATH` 调用 |
| 必需的环境变量 | `SLURM_JOB_QOS=kd-high`（让官方 ClusterStateManager 失效；队列脚本已设置）；`CUDA_VISIBLE_DEVICES` 选 GPU，仿真端和规划端共用同一张 |
| 官方克隆（只读，不得修改） | `/bsuhome/jiahuizhang/scratch/projects/occupancy/FisherRF/FisherRF-active-mapping`（8b196cd） |

## 2. 脚本（`experiments/cvpr_results/fisherrf_reproduction/scripts/`）

| 文件 | sha256 前 16 位 | 作用 |
|---|---|---|
| `fisherrf_full_planner.py` | fbcb4774cd6b6715 | **FisherRF-full 规划子进程**，即 v3：官方 GaussianSLAM、`pose_eval`、官方规划循环，3D / 网格适配见 `PROTOCOL_ADAPTATION.md` F1b。`FRF_FRONTIER_MODE` 不设或为 `official` 时就是 v3；设为 `magician` 时是 F1c 变体，**只做过 20 步冒烟** |
| `fisherrf_planner.py` | 774af49bfdbc914f | D3a 规划器（只对一步邻居打分），已完成的消融结果，**冻结** |
| `fisherrf_sim_out.py` | a0437b45198ce6c8 | Macarons++ 仿真包装：冻结的 t24 → v2_sim → m_t6_sim 链，加 RGB 旁路文件。`--planner fisherrf_full_planner.py` 选 FisherRF-full；不带此选项时是 D3a |
| `fisherrf_out_cov.py` | 8500dd4751dc489f | 官方覆盖率回放，即 Table 1 的 t25 流程，必须与存储曲线逐位一致 |
| `frf_full_queue2.sh` | ec5d30d4d8babfca | **v3 队列 worker**：先跑闭环，再做回放。输出根为 `full/outdoor/closed_loop_v3/`，claims 为 `full/jobs/claims2/`；需要 `full/jobs/GO_v3` 存在（现在存在） |
| `frf_full_queue.sh` | aebc30febe15df44 | v2 队列，输出根 `full/outdoor/closed_loop/`，**只留作记录，不要再用**：它会配合不服从动作规则的 v2 行为 |
| `frf_full_queue3.sh` | 4510732d83e9802c | F1c 队列：`FRF_FRONTIER_MODE=magician`，输出根 `closed_loop_f1c/`，claims3，需要 `GO_f1c`。**未运行**，产物在 `_aborted_f1c/` |
| `frf_full_collect.py` | 5943a760dd17d7f7 | 汇总：逐条与按场景的 AUC / Final、回放检查、行为与耗时；对照论文、D3a 和 Table 1 的 MAG-B |
| `fisherrf_sim_in.py` | — | MP3D 仿真包装，目前只接 D3a 规划器 |

**冻结（不要改）**：
- 官方克隆；
- `fisherrf_planner.py`、`fisherrf_sim_out.py` 的 D3a 默认路径；
- 冻结的基准代码：`exp5_future_view/task2/scripts/t24_out_sim.py`、`t25_out_cov.py`、`task1/scripts/v2_sim.py`、`exp3b_macaronspp_transfer/m_t6_sim.py`、`m_common.py`；
- `table1/` 下的一切；
- 已完成的结果目录 `outdoor/closed_loop/`（D3a，70 条）和 `full/outdoor/closed_loop_v3/`（v3 试跑）。

## 3. 恢复命令

所有命令都在仓库根目录运行。先设：

```bash
R=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene
F=$R/work_dirs/cvpr_results/fisherrf_reproduction/full
S=$R/experiments/cvpr_results/fisherrf_reproduction/scripts
```

**(a) 重跑或扩展试跑（v3）**

队列会跳过已有 metrics 的条目；被别人占用的条目也会跳过，所以要先清掉未完成条目的 claim。

```bash
# 例：补跑 / 新增条目，每行 <scene>:<start>[:<moves>]
printf "barts:0:100\n" > $F/jobs/rerun.txt
for u in $(cut -d: -f1,2 --output-delimiter=_s $F/jobs/rerun.txt); do
  [ -f $F/outdoor/closed_loop_v3/$u/metrics/FisherRF/FisherRF_${u}_r0.json ] || rm -rf $F/jobs/claims2/$u
done
# 每张 GPU 最多 2 个 worker；先确认 nvidia-smi 上有空闲显存
for g in 1 2; do for w in 1 2; do setsid nohup bash $S/frf_full_queue2.sh $g $F/jobs/rerun.txt R$w > /dev/null 2>&1 < /dev/null & sleep 3; done; done
```

**(b) 全量 Macarons++ 14×5（v3）**

```bash
: > $F/jobs/full14.txt
for s in alhambra bannerman barts bridge colosseum dunnottar eiffel fushimi liberty neuschwanstein pantheon pisa redeemer sestino_museum; do
  for k in 0 1 2 3 4; do echo "$s:$k:100" >> $F/jobs/full14.txt; done
done
# 试跑过的 15 条已有 metrics，会被自动跳过
for g in 0 1 2 3; do for w in 1 2; do setsid nohup bash $S/frf_full_queue2.sh $g $F/jobs/full14.txt A$w > /dev/null 2>&1 < /dev/null & sleep 3; done; done
```

- 规模：每条 episode 在共享 GPU 上墙钟约 12–17 分钟，回放约 2.5 分钟。55 条新 episode 在 8 个 worker 上约 2–2.5 小时。
- 改用 Slurm：仿照 `full/jobs/pilot_full.sbatch`，但要把其中的 `frf_full_queue.sh` 换成 `frf_full_queue2.sh`。提交前必须先跑 `sbatch --test-only`，**输出中出现 `Preempts:` 就不能提交**。
- **D13 注意**：dunnottar s1 / s2 / s4 和 redeemer s1 的起点帧没有有效深度。规划器会按 D13 处理（跳过建图，没有地图时按种子随机），不会崩溃。

**(c) 回放与汇总**

```bash
# 回放（队列已经在每条 episode 之后自动做）；单独补做：
CUDA_VISIBLE_DEVICES=1 env -u LD_PRELOAD -u PYTHONPATH /bsuscratch/jiahuizhang/envs/magician/bin/python $S/fisherrf_out_cov.py \
  --root $F/outdoor/closed_loop_v3/<scene>_s<k> --out $F/outdoor/closed_loop_v3/<scene>_s<k>/metrics --scene <scene> --start <k>
# 汇总（只用 CPU）
/bsuscratch/jiahuizhang/envs/fisherrf/bin/python -B $S/frf_full_collect.py --root $F/outdoor/closed_loop_v3 \
  --scenes alhambra bannerman barts bridge colosseum dunnottar eiffel fushimi liberty neuschwanstein pantheon pisa redeemer sestino_museum \
  --tag full14_v3
# 输出到 experiments/cvpr_results/fisherrf_reproduction/results/<tag>_episodes.csv 和 <tag>_summary.json
```

**停止 worker**：按 PID 停，**不要用 `pkill -f`**，它会匹配到发命令的那个 shell 自身。

```bash
ps -eo pid,args | awk '$2=="bash" && $3 ~ /frf_full_queue2?\.sh$/'
# 再对列出的 worker 及其子进程（ps --ppid）执行 kill <PID>
```

## 4. 输出根与杂项目录（`work_dirs/cvpr_results/fisherrf_reproduction/full/`）

| 路径 | 内容 | 处理 |
|---|---|---|
| `outdoor/closed_loop_v3/<s>_s<k>/` | **v3 试跑**（结论用这个）：episodes、ckpt、svc（含 RGB 旁路文件）、planner_logs（逐步 JSON，含每次重规划的记录）、planner_work（官方建图 PNG）、metrics（回放结果） | 保留 |
| `outdoor/closed_loop/` | v2 记录，9 条（路径不服从动作规则） | 保留作记录，不进表 |
| `outdoor/smoke/` | 冒烟：barts s0（v1，10 步）、dunnottar s0 v2（15 步）、redeemer s1 v3（30 步） | 可删 |
| `_aborted_v1/` | v1 中止的 8 条（两处适配缺陷） | 可删 |
| `_aborted_f1c/` | F1c：20 步冒烟和被暂停中止的那条 | 可删；恢复 F1c 时参考 |
| `_failed/` | v2 的失败尝试（GPU 0 OOM 和手动中止） | 可删 |
| `_failed_v3/` | v3 的失败尝试：redeemer s1 前两次、barts s2 第一次的 OOM 半成品，以及 redeemer s1 被释放的 claim。目录名中的 tryN，是**移走时正要开始的那次尝试的编号**，不是半成品自己的尝试编号。redeemer s1 后来在 GPU 1 上重跑完成，用的是 `jobs/rerun_v3_oom.txt` | 可删 |
| `jobs/claims/`、`jobs/claims2/` | mkdir 锁：v2 和 v3。要重跑某条时，删掉它的 claim 子目录 | 见 §3(a) |
| `jobs/GO_v3` | queue2 的启动闸门 | 保留 |
| `jobs/pilot_joblist.txt` | 试跑条目 | — |
| `jobs/launch_v2b.sh` | "等旧 worker 退出再启动 v3 worker"的启动器（一次性） | 不需要再用 |
| `logs/` | `worker2_*.log`（v3 worker）、`episodes_v3/`（每条 episode 的仿真日志和回放日志）、`worker_*.log` 和 `episodes/`（v2） | — |

## 5. 未完成事项

1. **MP3D 室内版 FisherRF-full 还没有实现**。
   - `fisherrf_sim_in.py` 目前只接 D3a 规划器。
   - 室内的相机网格来自 scene card（`task0/.../grid_env.py`），没有 settings.json。所以 `PoseGrid` 需要改为从 card 读取，`ELEVS` / `N_AZIM` 用室内的值。
   - 服务记录名是 `svc_t{t}.npz`，规划端已经支持 `--svc-name`；但 FisherRF-full 的 `step()` 里写死了 outdoor 的 scene_key 和文件命名，要一并改。
2. **与论文的差距**：机制和候选方案见 `PILOT_FULL.md` §5、§6。其中 F1c（MAGICIAN 式全局位姿采样）已实现，只做过 20 步冒烟。
3. **D12 和 D13** 的处理已在 `SMOKE_TEST.md` 和 `PROTOCOL_ADAPTATION.md` 中定下，恢复时不需要再改。
4. 运行中曾遇到另一个实验的进程在 GPU 0 或 GPU 3 上涨到 35–44 GB。恢复前先看 `nvidia-smi`，**避开显存紧张的卡**。

## 6. 2026-11-12 恢复：v4 / f1c4（通过共享 GPU 队列运行）

- **新增**：
  - 可达性过滤 `FRF_REACH_FILTER=1`，对应官方 `gaussian.py:1244-1257`，说明见 `PROTOCOL_ADAPTATION.md` F1b 第 5 条；
  - 任务生成器 `scripts/frf_queue_tasks.py`：每条 episode 生成一个任务 = 闭环 + 官方回放；metrics 已存在、或同名任务在 pending / running / done 中的，一律跳过；任务内部用 mkdir 加锁。
- **只用共享队列** `work_dirs/cvpr_results/gpu_queue`：worker 脚本是 `experiments/cvpr_results/scripts/gpu_queue_worker.sh`，gpu114 和 gpu115 上各 4 个 worker，每张 GPU 同时只跑一个任务。**不直接起 GPU 进程，不用 Slurm。**
- **冒烟**（2026-11-12 19:28–19:32，gpu115 GPU 1，redeemer s1，20 步）：两个变体都是 rc = 0、回放逐位一致、过滤器开启。
  - f1c4：cov@20 0.4061；
  - v4：cov@20 0.2184；
  - 前 20 步中过滤器都没有丢掉任何候选（n_before = n_after）。
  - 输出在 `full/outdoor/smoke_{v4,f1c4}/`。
- **已入队**：试跑 2 × 15 条（dunnottar / redeemer / barts × 起点 0–4，100 步），任务名为 `frf_v4_*` 和 `frf_f1c4_*`，输出根分别为 `full/outdoor/closed_loop_v4/` 和 `full/outdoor/closed_loop_f1c4/`。
  - 每条 episode 的日志：`full/logs/queue_<variant>/`；任务日志：`gpu_queue/logs/<任务名>.log`。
  - 中途失败的 episode 移到 `full/_failed_<variant>/`；任务的 mkdir 锁在 `full/jobs/qclaims_<variant>/`。
  - 任务被硬杀时锁会残留，需要手动删。

```bash
S=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/experiments/cvpr_results/fisherrf_reproduction/scripts
F=/bsuscratch/jiahuizhang/projects/occupancy/seen2scene/work_dirs/cvpr_results/fisherrf_reproduction/full
PY=/bsuscratch/jiahuizhang/envs/fisherrf/bin/python
# 试跑入队（已执行；重复执行不会产生重复任务）
$PY -B $S/frf_queue_tasks.py --variant v4   --joblist $F/jobs/pilot_joblist.txt
$PY -B $S/frf_queue_tasks.py --variant f1c4 --joblist $F/jobs/pilot_joblist.txt
# 全量 14×5（等用户决定；试跑过的 15 条有 metrics，会被自动跳过，每个变体新增 55 条）
$PY -B $S/frf_queue_tasks.py --variant v4   --joblist $F/jobs/full14_joblist.txt
$PY -B $S/frf_queue_tasks.py --variant f1c4 --joblist $F/jobs/full14_joblist.txt
# 失败任务重新入队
$PY -B $S/frf_queue_tasks.py --variant v4 --joblist $F/jobs/pilot_joblist.txt --retry-failed
# 汇总（只用 CPU）
$PY -B $S/frf_full_collect.py --root $F/outdoor/closed_loop_v4   --tag pilot_v4
$PY -B $S/frf_full_collect.py --root $F/outdoor/closed_loop_f1c4 --tag pilot_f1c4
# 全量汇总时加：--scenes alhambra bannerman barts bridge colosseum dunnottar eiffel fushimi liberty neuschwanstein pantheon pisa redeemer sestino_museum --tag full14_v4（或 full14_f1c4）
```
