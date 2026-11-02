"""EXP6 V5 -- training report of both arms (CPU): run_none.json / log.jsonl / eval.jsonl / train_<arm>.log /
<arm>_final.json -> experiments/exp6_v5/reports/TRAINING.md + figures/training_v5.png. Usage: python training_summary_v5.py"""
import json
import os
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
W, E = f"{R}/work_dirs/exp6_v5", f"{R}/experiments/exp6_v5"
NAME = {"v5": "V5-Warm12k-S10Score", "v4c": "V4-Continue12k-S10Score"}


def arm_facts(arm):
    run_p = f"{W}/runs/{arm}/run_none.json"
    if not os.path.exists(run_p):
        return None
    run = json.load(open(run_p))
    log = [json.loads(x) for x in open(f"{W}/runs/{arm}/log.jsonl")]
    ev = [json.loads(x) for x in open(f"{W}/runs/{arm}/eval.jsonl")]
    tl = open(f"{W}/logs/train_{arm}.log").read().splitlines()
    st = [x for x in tl if x.startswith("start ")]
    en = [x for x in tl if x.startswith("EXIT ")]
    ts = lambda x: time.mktime(time.strptime(" ".join(x.split()[-2:]), "%Y-%m-%d %H:%M:%S"))  # noqa: E731
    hours = (ts(en[-1]) - ts(st[0])) / 3600
    fin = json.load(open(f"{W}/checkpoints/{arm}_final.json")) if os.path.exists(f"{W}/checkpoints/{arm}_final.json") else {}
    sps = np.asarray([l["s_per_step"] for l in log[1:]])
    return {"run": run, "log": log, "ev": ev, "start": st[0], "end": en[-1], "hours": hours, "gpu_h": 4 * hours, "final": fin, "sps": sps}


def main():
    F = {a: arm_facts(a) for a in ("v5", "v4c")}
    L = ["# EXP6 V5 正式训练报告（自动生成 `scripts/training_summary_v5.py`）", ""]
    L += ["| 项 | " + " | ".join(NAME[a] for a in F if F[a]) + " |", "|---|" + "---|" * sum(1 for a in F if F[a])]
    rows = [("状态", lambda f: f"{f['end'].split()[0]} {f['end'].split()[1]}（{f['start'].split()[-2]} {f['start'].split()[-1]} → {f['end'].split()[-2]} {f['end'].split()[-1]}）"),
            ("节点 / 作业", lambda f: f"{f['run']['host']} 4 × {f['run']['gpu']}，作业 {f['run'].get('slurm_job') or '3443137（交互分配内 setsid）'}"),
            ("git", lambda f: f"`{f['run']['git']['sha'][:8]}` dirty = {f['run']['git']['dirty']}"),
            ("配置 sha256", lambda f: f"train `{f['run']['config_sha256'][:12]}…`，model `{f['run']['model_cfg_sha256'][:12]}…`"),
            ("初始化", lambda f: f"{f['run']['init']['v4_ckpt'].split('seen2scene/')[-1]} sha `{f['run']['init']['v4_sha256'][:12]}…`；" + (f"复制 {f['run']['init']['copied']} 张量，新模块 {f['run']['init']['new_modules_numel']:,} 参数" if 'new_modules_numel' in f['run']['init'] else "全部参数")),
            ("数据", lambda f: f"训练 {f['run']['train_trajs']} 轨迹 / {f['run']['train_views']:,} 视角；MP3D 样本 {f['run']['mp3d_training_samples']}" + ("；多表面标签 138 + 42 条（Train-6）" if f['run']['label_meta'] else "")),
            ("步数", lambda f: f"{f['log'][-1]['step']:,} / {f['run']['train_cfg']['steps']:,}"),
            ("墙钟 / GPU-h", lambda f: f"{f['hours']:.2f} h ≈ **{f['gpu_h']:.1f} GPU-h**"),
            ("吞吐 / 显存", lambda f: f"{f['sps'].mean():.3f} s/步（p90 {np.quantile(f['sps'], 0.9):.3f}），峰值 {max(l['mem_gb'] for l in f['log']):.1f} GB/卡"),
            ("损失", lambda f: f"第 1 步 {f['log'][0]['loss']:.4f} → 最后 50 步 {f['run'].get('final_loss_mean50', float('nan')):.4f}"),
            ("最终权重", lambda f: (f"固定 EMA@{f['final']['chosen_step']} → `checkpoints/{f['final']['arm']}_final.pt` sha `{f['final']['sha256'][:16]}…`" if f['final'] else "待发布"))]
    for nm, fn in rows:
        L.append(f"| {nm} | " + " | ".join(fn(F[a]) for a in F if F[a]) + " |")
    L += ["", "## S-c 监控（EMA；只监控，不选权重）", ""]
    for a in F:
        f = F[a]
        if not f:
            continue
        L += [f"### {NAME[a]}", "", "| 步 | FM S-c | FM train_sub | ES S-c (vox) | Brier | false wall | hit ratio" + (" | belief 首表面召回 | 第二+ 召回 | 负例正确率 | 首表面后负例正确率 | L_surf" if a == "v5" else "") + " |",
              "|---|---|---|---|---|---|---" + ("|---|---|---|---|---" if a == "v5" else "") + "|"]
        for e in f["ev"]:
            s = e.get("samp_sc_sub") or {}
            row = f"| {e['step']:,} | {e['fm_sc_sub']:.4f} | {e['fm_train_sub']:.4f} | {s.get('fair_es_arclength_vox', float('nan')):.3f} | {s.get('brier', float('nan')):.3f} | {s.get('false_wall', float('nan')):.3f} | {s.get('hit_ratio', float('nan')):.2f}"
            if a == "v5":
                b = e.get("surf_sc") or {}
                row += f" | {b.get('recall_first', float('nan')):.3f} | {b.get('recall_second_plus', float('nan')):.3f} | {b.get('neg_acc', float('nan')):.3f} | {b.get('neg_acc_behind_first', float('nan')):.3f} | {b.get('loss', float('nan')):.3f}"
            L.append(row + " |")
        L.append("")
    if F["v5"]:
        lg = F["v5"]["log"]
        pick = [l for l in lg if l["step"] in (1, 500, 1000, 2000, 2050, 3000, 4000, 6000, 8000, 10000, 12000)]
        L += ["### V5 训练过程（Bridge → Joint；H = 64 维射线条件，Teacher = 冻结 V4）", "", "| 步 | 阶段 | L_FM | L_surf | L_bridge | λ_bridge | H_new / H_old 余弦 | H_new std | H_old std | 梯度 dec / encoder / unet |", "|---|---|---|---|---|---|---|---|---|---|"]
        for l in pick:
            h, g = l.get("H") or {}, l.get("grad_parts") or {}
            L.append(f"| {l['step']:,} | {l['phase']} | {l.get('fm', float('nan')):.4f} | {l.get('surf', float('nan')):.3f} | {l.get('bridge', float('nan')):.3f} | {l['lambda_bridge']:.3f} | "
                     f"{h.get('H_cos', float('nan')):.3f} | {h.get('H_new_std', float('nan')):.4f} | {h.get('H_old_std', float('nan')):.4f} | {g.get('dec', float('nan')):.3f} / {g.get('encoder', float('nan')):.3f} / {g.get('unet', float('nan')):.3f} |")
        L.append("")
    L += ["S-c = Train-6 六个场景的保留轨迹；没有任何评估场景（Dev-3 / Test-3 / MP3D）参与训练、监控或选择。", ""]
    open(f"{E}/reports/TRAINING.md", "w").write("\n".join(L))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.2))
        for a, c in (("v4c", "#495057"), ("v5", "#7048e8")):
            f = F[a]
            if not f:
                continue
            axs[0].plot([l["step"] for l in f["log"]], [l.get("fm", np.nan) for l in f["log"]], lw=1.0, color=c, label=NAME[a])
            axs[1].plot([e["step"] for e in f["ev"]], [e["fm_sc_sub"] for e in f["ev"]], "o-", color=c, label=NAME[a])
            axs[2].plot([e["step"] for e in f["ev"] if e.get("samp_sc_sub")], [e["samp_sc_sub"]["fair_es_arclength_vox"] for e in f["ev"] if e.get("samp_sc_sub")], "o-", color=c, label=NAME[a])
        if F["v5"]:
            ev = [e for e in F["v5"]["ev"] if e.get("surf_sc")]
            for k, c in (("recall_first", "#1971c2"), ("recall_second_plus", "#e8590c"), ("neg_acc_behind_first", "#2f9e44")):
                axs[3].plot([e["step"] for e in ev], [e["surf_sc"][k] for e in ev], "o-", color=c, label=k)
        axs[0].set_yscale("log"); axs[0].set_title("train L_FM (per 50 steps)", loc="left")
        axs[1].set_title("S-c FM loss (EMA)", loc="left"); axs[2].set_title("S-c fair ES (vox, EMA)", loc="left"); axs[3].set_title("V5 S-c surface belief (EMA)", loc="left")
        for ax in axs:
            ax.set_xlabel("step"); ax.legend(fontsize=7); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{E}/figures/training_v5.png", dpi=110)
    except Exception as ex:                                   # noqa: BLE001
        print("figure skipped:", ex)
    print("written", f"{E}/reports/TRAINING.md", {a: round(F[a]["gpu_h"], 1) for a in F if F[a]})


if __name__ == "__main__":
    main()
