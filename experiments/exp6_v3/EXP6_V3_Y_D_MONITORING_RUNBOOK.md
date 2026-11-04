# EXP6 V3 Y / Experiment D — Night Monitoring & Recovery Runbook

配合 `EXP6_V3_Y_AND_ABLATION_D_8H_AUTONOMOUS_PLAN.md` 使用。**这是给有 HPC shell 权限的代码 AI 实施的操作说明；当前只是文档，不代表这些脚本已安装、任务已启动。**

## 1. 启动前（必须通过）

- `squeue -u "$USER"` 核对 8 GPU allocation 所属 job IDs、每个节点实际资源。
- 如果在 Allocation 内：`scontrol show job "$SLURM_JOB_ID"`；如果 Slurm 不是从当前 Shell 提供的，则使用 `squeue` 获取 job ID，不能猜。
- `nvidia-smi -L`、`torch.cuda.device_count()`、GPU UUID，防止 Y D 争抢相同 GPU。
- 核对训练 checkpoint 完整 SHA256 和严格模型权重加载；检查 optimizer/EMA 是否能恢复。
- `git status --short`、`git rev-parse HEAD`、CVPR Table1 registry 与 official evaluator 真实位置。
- 所有 logs/results/checkpoints 进入独立 `run_id` 命名空间，不写旧结果。

## 2. 状态查看命令

```bash
# Slurm 状态（公共节点上低频执行）
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.12l %.6D %R'

# 作业详情（真实 job ID）
scontrol show job <JOB_ID> | sed -n '1,110p'

# Accounting 功能允许时
sacct -j <JOB_ID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%50

# 在有真实 GPU 资源权限的计算节点上：
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total --format=csv

# 在脚本真正创建状态文件之后：
cat <RUN_DIR>/STATUS.json
tail -n 80 <RUN_DIR>/logs/controller.log
tail -n 30 <RUN_DIR>/EVENTS.jsonl
```

> 不在 Login Node 上运行无限 `watch` 或重型日志扫描；若要每分钟自动检查，在已授权计算节点使用 Slurm 管理的低资源 controller/worker，或作业脚本内部有限循环。`tmux` 不能替代 Slurm 的自动续跑。

## 3. 自动控制器应有的有限状态机

```text
PENDING
  -> RUNNING
  -> CHECKPOINTING
  -> VALIDATING
  -> EVALUATING
  -> REGISTERING
  -> COMPLETED

RUNNING/EVALUATING
  -> RETRYABLE_ERROR -> RECOVERING -> RUNNING
  -> BLOCKED_FATAL   -> save state and free GPU
  -> PARTIAL_BUDGET  -> save state, summarize, free GPU
```

- 任何异常要先记录事件再重试。
- 同一个 episode key，成功完成后不重复跑；本来运行中不能再次提交。
- 重试一般最多 2 次；超过则停止该子项，**不是退出另外一条线**。
- 保存 `last_good_checkpoint`，checksum 校验失败不可恢复使用。
- 为每个工作的 `run_id` 加运行锁；控制器重启要先做幂等恢复。
- 训练和评估不可混用同一个 CUDA_VISIBLE_DEVICES，除非明确受调度且通过资源验证。

## 4. 自动记录模板

`STATUS.json` 实例形状（下面是字段样例，不是已运行状态）：

```json
{
  "run_id": "PLACEHOLDER_Y_20261104",
  "track": "Y",
  "phase": "RUNNING",
  "job_id": null,
  "hostname": null,
  "gpu_uuids": [],
  "parent_checkpoint_sha256": "VERIFY_BEFORE_RUN",
  "code_commit": "VERIFY_BEFORE_RUN",
  "protocol_sha256": "VERIFY_BEFORE_RUN",
  "last_heartbeat_utc": null,
  "steps_completed": 0,
  "episodes_completed": 0,
  "episodes_target": 0,
  "last_good_checkpoint": null,
  "last_error": null,
  "retry_count": 0,
  "resume_command": "TO_BE_FILLED_BY_ACTUAL_RUNNER"
}
```

`EVENTS.jsonl` 每行需要 timestamp、track、run_id、phase、event_type、old/new state、step/episode ID、error details、action、job ID。**真实运行时不要复制本模板的假时间或 ID。**

## 5. 自动续跑触发规则

1. 训练 worker 终止：读 exit code + slurm reason；检查最新 ckpt hash 和 step，再从正确 step resume，不能重复更新已完成的 optimizer step。
2. 评估 Episode worker 终止：检查 `episode_complete=true, steps=100, official evaluator verified`，否则重新排队同 seed；部分曲线不可用于正式 Final。
3. 共享 NFS 变慢：先检查 inode / 目录 I/O / rank 日志，减少频繁 scan；不因为写 log 慢就无限重启 GPU。
4. GPU OOM：首先减少 episode worker 并发/修改 placement；不要无记录改变分辨率、模型结构、D100/S10、K4、Euler10。
5. Python exception 可预期且非致命：保留 traceback 和 seed，修复时记录 diff；如果修复会改变科学定义，必须新 protocol ID，之前的轨迹不应和新实现混合。
6. 预算临界：停止新 optional D4/Y3、优先 completed primary Dev & All14、生成晨间报告。

## 6. 早晨用户检查的最快路径

```bash
# 路径按代码 AI 输出的真实位置替换：
cat <REPORT_ROOT>/OVERNIGHT_8H_SUMMARY.md
cat <REPORT_ROOT>/CVPR_TABLE1_ALIGNMENT.md
cat <Y_RUN_DIR>/STATUS.md
cat <D_RUN_DIR>/STATUS.md

# 检查有没有未完成任务
squeue -u "$USER" -o '%.18i %.24j %.2t %.10M %.12l %.6D %R'
```

**必须看到**：Y 续训完成到哪一步、最佳可用 checkpoint、D1/D2/D3 哪些 100-step episodes 完整、Dev / Train / Test / All14 官方 AUC@20、AUC@100、Final 和 Colosseum/Pisa 诊断，以及每个未完成/失败项的原因与继续命令。

## 7. 与 CVPR Table1 的验收原则

- 仅通过原流水线 registry 注册新增模型/策略；旧 checkpoint/旧结果不可重写。
- 14 个 scene × 5 starts 全部完成才叫 All-14；`sofia_church` 缺失继续标 `All-15 pending`。
- 官方 Coverage 主口径与 secondary monotonic 清楚区分；所有 variants 统一真实 D100 输入 / S10 scoring、R10、K4、Euler10、100 steps。Y 的训练可以沿用原 V3-T1 mixed D100，不应改评估协议。
- **强制汇报**：Early AUC、Overall AUC、Final Coverage、runtime、失败场景、配对 CI（如果有完整 scenes）。
