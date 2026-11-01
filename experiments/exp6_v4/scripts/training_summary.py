"""Training summary of a finished Outdoor-only arm (CPU): facts from run_none.json / log.jsonl / eval.jsonl / train.log
-> report/OUTDOOR_TRAINING.md (V4) or exp6_v3.1/report/OUTDOOR_TRAINING.md (V3.1) + figures/training_<arm>.png.
Usage: python training_summary.py v4|v31"""
import json
import os
import sys
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
arm = sys.argv[1]
W = f"{R}/work_dirs/exp6_v4" if arm == "v4" else f"{R}/work_dirs/exp6_v3_1"
E = f"{R}/experiments/exp6_v4" if arm == "v4" else f"{R}/experiments/exp6_v3.1"
NAME = "V4（V2 3D U-Net + Ray-Aligned X-Attn + 原 2D Flow）" if arm == "v4" else "V3.1（V3-T0 架构：Simple 3D VAE + 原 Ray Encoder + 原 2D Flow）"
run = json.load(open(f"{W}/runs/main/run_none.json"))
log = [json.loads(x) for x in open(f"{W}/runs/main/log.jsonl")]
ev = [json.loads(x) for x in open(f"{W}/runs/main/eval.jsonl")]
tl = open(f"{W}/logs/train.log").read().splitlines()
start = next(x for x in tl if x.startswith("start ")); end = next(x for x in tl if x.startswith("EXIT "))
t0 = time.mktime(time.strptime(" ".join(start.split()[-2:]), "%Y-%m-%d %H:%M:%S")); t1 = time.mktime(time.strptime(" ".join(end.split()[-2:]), "%Y-%m-%d %H:%M:%S"))
hours = (t1 - t0) / 3600
rays_key = "rays" if arm == "v4" else "rays_outdoor"
sps = np.asarray([l["s_per_step"] for l in log[1:]])
rays = np.asarray([l[rays_key] for l in log])
fin = json.load(open(f"{W}/checkpoints/final.json")) if os.path.exists(f"{W}/checkpoints/final.json") else {}
L = [f"# {NAME} —— Outdoor-only 正式训练报告（自动生成 `exp6_v4/scripts/training_summary.py {arm}`）", "",
     f"> 状态：**Completed**（{start.split()[-2]} {start.split()[-1]} → {end.split()[-2]} {end.split()[-1]}，{end.split()[0]} {end.split()[1]}，{log[-1]['step']:,} / {run['train_cfg']['steps']:,} 步；"
     f"{run['host']} 4 × {run['gpu']}，作业 {run.get('slurm_job')}；墙钟 {hours:.2f} h ≈ **{4 * hours:.1f} GPU-h**）。checkpoint 规则：固定 EMA@{run['train_cfg']['steps']:,}"
     + (f" → `checkpoints/final.pt` sha256 `{fin.get('sha256', '')[:16]}…`" if fin else "（待发布）") + "。", "",
     "| 项 | 值 |", "|---|---|",
     f"| git | `{run['git']['sha'][:8]}`，dirty = {run['git']['dirty']} |",
     f"| 配置 sha256 | train `{run['config_sha256'][:12]}…`，model `{(run.get('model_config_sha256') or run.get('model_cfg_sha256'))[:12]}…`，data manifest `{run['data_manifest_sha256'][:12]}…` |",
     f"| 数据 | 域 {run['domains']}，MP3D 样本 {run['mp3d_training_samples']}；" + (f"训练 {run['train_trajs']} 条轨迹 / {run['train_views']:,} 视角 / {run['train_states']:,} 状态；S-c {run['held_trajs']} 条 / {run['held_views']:,} 视角" if arm == "v4" else f"训练 {run['data_counts']['out_train']} 条轨迹（rank 0 持有 {run['views']['out_train']:,} 视角）；S-c {run['data_counts']['out_sc']} 条 / {run['views']['out_sc']:,} 视角") + " |",
     f"| 初始化 | 随机（scratch），不加载任何权重" + (f"；公共模块初始 sha `{run['common_init_sha256'][:12]}…`（= V3-T0）" if arm != "v4" else "") + " |",
     f"| 参数 | {run['params']} |",
     f"| 每步 | 全局 {run['train_cfg']['batch_views']} 视角 / {run['train_cfg']['states_per_batch']} 状态；每 rank {run['per_rank_batch']} |",
     f"| 优化 | lr {run['train_cfg']['lr']}，预热 {run['train_cfg']['warmup']}，余弦；EMA {run['train_cfg']['ema']}；grad clip {run['train_cfg']['grad_clip']}；seed {run['train_cfg']['seed']} |",
     f"| 吞吐 | 均值 **{sps.mean():.3f} s/步**（p90 {np.quantile(sps, 0.9):.3f}），峰值显存 {max(l['mem_gb'] for l in log):.1f} GB/卡 |",
     f"| 有效监督射线 | 每步均值 {rays.mean():,.0f}（min {rays.min():,.0f} / max {rays.max():,.0f}）；40k 步累计约 {rays.mean() * len(log) * 50 / 1e9 if len(log) > 1 else 0:.2f} B（按每 50 步记录外推） |",
     f"| 损失 | 第 1 步 {log[0]['loss']:.4f} → 最后 50 步均值 {run['final_loss_mean50']:.4f} |", ""]
if arm == "v4":
    L += ["| 步 | FM loss S-c | FM loss train_sub | ES S-c (vox) | Brier | false wall | hit ratio |", "|---|---|---|---|---|---|---|"]
    for e in ev:
        s = e.get("samp_sc_sub")
        L.append(f"| {e['step']:,} | {e['fm_sc_sub']:.4f} | {e['fm_train_sub']:.4f} | " + (f"{s['fair_es_arclength_vox']:.3f} | {s['brier']:.3f} | {s['false_wall']:.3f} | {s['hit_ratio']:.2f} |" if s else "— | — | — | — |"))
else:
    L += ["| 步 | FM loss S-c | FM loss out_train | ES S-c (vox) | Brier | false wall | hit ratio | recon acc (E/U/B) | sdf MAE | KL/unit | active units |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in ev:
        s, rc = e.get("samp_out_sc"), e.get("recon_out_sc", {})
        kl = [l for l in log if l["step"] <= max(e["step"], 1) and "posterior" in l]
        po = kl[-1]["posterior"] if kl else {}
        L.append(f"| {e['step']:,} | {e['fm_out_sc']:.4f} | {e['fm_out_train']:.4f} | {s['fair_es_arclength_vox']:.3f} | {s['brier']:.3f} | {s['false_wall']:.3f} | {s['hit_ratio']:.2f} | "
                 f"{rc.get('acc_empty', float('nan')):.3f} / {rc.get('acc_unknown', float('nan')):.3f} / {rc.get('acc_band', float('nan')):.3f} | {rc.get('sdf_mae', float('nan')):.3f} | {kl[-1]['kl'] if kl else float('nan'):.3f} | {po.get('active_units', '—')} |")
L += ["", "S-c = Train-6 六个场景的保留轨迹，只作监控；没有任何评估场景（Dev-3 / Test-3 / MP3D）参与训练或选择。", ""]
os.makedirs(f"{E}/report", exist_ok=True)
open(f"{E}/report/OUTDOOR_TRAINING.md", "w").write("\n".join(L))
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot([l["step"] for l in log], [l["loss"] for l in log], lw=1.2, color="#1c5cab"); axs[0].set_yscale("log"); axs[0].set_title("train loss (mean of 50 steps)", loc="left"); axs[0].set_xlabel("step")
    k = "fm_sc_sub" if arm == "v4" else "fm_out_sc"; k2 = "fm_train_sub" if arm == "v4" else "fm_out_train"
    axs[1].plot([e["step"] for e in ev], [e[k] for e in ev], "o-", color="#d6336c", label="S-c"); axs[1].plot([e["step"] for e in ev], [e[k2] for e in ev], "s--", color="#999", label="train subset"); axs[1].set_title("dev FM loss (EMA)", loc="left"); axs[1].legend(); axs[1].set_xlabel("step")
    sk = "samp_sc_sub" if arm == "v4" else "samp_out_sc"
    ee = [e for e in ev if e.get(sk)]
    axs[2].plot([e["step"] for e in ee], [e[sk]["fair_es_arclength_vox"] for e in ee], "o-", color="#0b7285"); axs[2].set_title("S-c fair ES (vox, EMA)", loc="left"); axs[2].set_xlabel("step")
    fig.suptitle(f"{arm.upper()} Outdoor-only training ({run['host']}, {hours:.1f} h wall)", x=0.01, ha="left")
    fig.tight_layout(); os.makedirs(f"{E}/report/figures", exist_ok=True); fig.savefig(f"{E}/report/figures/training_{arm}.png", dpi=110)
except Exception as ex:                                       # noqa: BLE001
    print("figure skipped:", ex)
print("written", f"{E}/report/OUTDOOR_TRAINING.md", f"{4 * hours:.1f} GPU-h", f"{sps.mean():.3f} s/step")
