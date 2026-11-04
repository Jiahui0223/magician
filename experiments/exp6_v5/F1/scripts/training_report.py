"""EXP6 V5 / F1 -- reports/TRAINING.md + figures/training_curves.{png,pdf} from the run logs (CPU; generated).
Per arm: run json (params, warm start, data, config hash, git), log.jsonl (FM loss of the step batch, lr per group, grad
norms per module, s/step, peak memory), eval.jsonl (EMA FM loss on train_sub / S-c and sample metrics at every 1000 steps),
digest.jsonl (per-step digest of batch rows + tau + eps slice) -> stream equality of the two arms; EMA files and sha256."""
import glob
import hashlib
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
F1 = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(os.path.dirname(F1)))
W = os.path.join(R, "work_dirs", "exp6_v5", "F1", "runs")


def jl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def main():
    D = {}
    for arm in ("A", "B"):
        d = os.path.join(W, arm)
        runs = sorted(glob.glob(os.path.join(d, "run_none_*.json")))
        D[arm] = {"run": json.load(open(runs[0])) if runs else None, "final": json.load(open(os.path.join(d, "run_none.json"))) if os.path.exists(os.path.join(d, "run_none.json")) else None,
                  "log": jl(os.path.join(d, "log.jsonl")), "eval": jl(os.path.join(d, "eval.jsonl")), "dg": {r["step"]: r["digest"] for r in jl(os.path.join(d, "digest.jsonl"))},
                  "ema": {f: hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest() for f in sorted(os.listdir(d)) if f.startswith("ema_")}}
    common = sorted(set(D["A"]["dg"]) & set(D["B"]["dg"]))
    same = sum(D["A"]["dg"][s] == D["B"]["dg"][s] for s in common)
    first_diff = next((s for s in common if D["A"]["dg"][s] != D["B"]["dg"][s]), None)
    f3 = lambda v: "—" if v is None else f"{v:.4f}"  # noqa: E731
    L = ["# TRAINING — F1-A / F1-B 训练记录（自动生成，`scripts/training_report.py`）", ""]
    for arm in ("A", "B"):
        r = D[arm]["run"] or {}
        lg = D[arm]["log"]
        L += [f"## F1-{arm}（{'Continue' if arm == 'A' else 'Geometry Query'}）", ""]
        if r:
            L += [f"- **运行**：主机 {r.get('host')}；GPU {r.get('gpu')} × {r.get('world_size')}；Slurm 作业 {r.get('slurm_job') or '已有交互分配'}；"
                  f"git {r.get('git', {}).get('sha', '')[:7]}（有本地改动：{r.get('git', {}).get('dirty')}）；开始于 {r.get('started')}",
                  f"- **配置**：配置哈希 `{r.get('config_hash', '')[:12]}…`；模型配置哈希 `{r.get('model_cfg_sha256', '')[:12]}…`；父 sha `{r.get('parent_sha256', '')[:12]}…`",
                  f"- **数据**：{r.get('train_trajs')} 条训练轨迹 / {r.get('train_views')} 个视角，S10 分片，共 {len(r.get('data_files_sha256', {}))} 个文件，均记录了 sha；MP3D 样本数 0",
                  f"- **参数**：{r.get('params')}",
                  f"- **初始化**：{ {k: v for k, v in (r.get('warm_start') or {}).items() if k != 'not_copied_parent_keys'} }"]
        fin = D[arm]["final"]
        if fin:
            L.append(f"- **结束**：{fin.get('finished')}；最后 50 步损失均值 {f3(fin.get('final_loss_mean50'))}")
        if lg:
            ph = {}
            for x in lg:
                ph.setdefault(x["phase"], []).append(x)
            L += ["", "| 阶段 | 步数范围 | s/step（中位数） | 峰值显存 GB / rank | 可训练参数 | grad 范数中位数（裁剪前，按模块） |", "|---|---|---|---|---|---|"]
            for p, xs in ph.items():
                gp = {}
                for x in xs:
                    for k, v in (x.get("grad_parts") or {}).items():
                        gp.setdefault(k, []).append(v)
                L.append(f"| {p} | {xs[0]['step']}–{xs[-1]['step']} | {np.median([x['s_per_step'] for x in xs[1:]] or [0]):.3f} | {max(x['mem_gb'] for x in xs):.1f} | "
                         f"{xs[-1].get('n_trainable')} | " + ", ".join(f"{k} {np.median(v):.3f}" for k, v in gp.items()) + " |")
        ev = D[arm]["eval"]
        if ev:
            L += ["", "| EMA step | FM train_sub | FM S-c | S-c samples：Brier / 假墙比例 / 预测命中率 / 标签命中率 |", "|---|---|---|---|"]
            for e in ev:
                s = e.get("samp_sc_sub", {})
                L.append(f"| {e['step']} | {f3(e['fm_train_sub'])} | {f3(e['fm_sc_sub'])} | {f3(s.get('brier'))} / {f3(s.get('false_wall'))} / {f3(s.get('pred_hit_rate'))} / {f3(s.get('label_hit_rate'))} |")
        L += ["", "| EMA 文件 | sha256 |", "|---|---|"] + [f"| {f} | `{h}` |" for f, h in D[arm]["ema"].items()] + [""]
    L += ["## 两个分支的样本流", "",
          f"- **比对结果**：逐步摘要（batch 行号、tau、eps 片段的 sha256）在共同的 {len(common)} 步中有 **{same}** 步相同；第一个不同的步：{first_diff}。",
          "- **续训**：如果某个分支中途续训，按 DDP 规则会在续训步重新设定种子，之后的摘要会不同；两个分支都没有续训时应全部相同。", "",
          "**说明**：",
          "- 日志里的 `loss` 是 rank 0 那一份 batch 的损失；",
          "- `fm` 是全局损失（全局有标签射线归一化）；",
          "- 监控只看 S-c（Train-6 场景中留出的轨迹），不用于选择 checkpoint。"]
    open(os.path.join(F1, "reports", "TRAINING.md"), "w").write("\n".join(L) + "\n")
    # curves
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2))
    for arm, col in (("A", "#1f5fbf"), ("B", "#d9480f")):
        lg = D[arm]["log"]
        if lg:
            st = np.asarray([x["step"] for x in lg])
            fm = np.asarray([x["fm"] for x in lg])
            k = 10
            sm = np.convolve(fm, np.ones(k) / k, mode="valid") if len(fm) > k else fm
            axs[0].plot(st[len(st) - len(sm):], sm, color=col, label=f"F1-{arm} (step batch FM, 10-pt mean)")
        ev = D[arm]["eval"]
        if ev:
            axs[1].plot([e["step"] for e in ev], [e["fm_sc_sub"] for e in ev], marker="o", color=col, label=f"F1-{arm} EMA, S-c")
            axs[1].plot([e["step"] for e in ev], [e["fm_train_sub"] for e in ev], marker="s", ls=":", color=col, label=f"F1-{arm} EMA, train_sub")
    for ax in axs:
        ax.axvline(2000, color="#999", lw=0.8, ls="--")
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.25)
    axs[0].set_ylabel("FM loss")
    axs[0].set_title("training FM loss (global batch)", loc="left", fontsize=9)
    axs[1].set_title("EMA FM loss (monitoring only)", loc="left", fontsize=9)
    axs[1].legend(fontsize=7)
    axs[0].legend(fontsize=7)
    os.makedirs(os.path.join(F1, "figures"), exist_ok=True)
    fig.savefig(os.path.join(F1, "figures", "training_curves.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(F1, "figures", "training_curves.pdf"), bbox_inches="tight")
    print("TRAINING.md written; digest equal", same, "/", len(common), "first diff", first_diff)


if __name__ == "__main__":
    main()
