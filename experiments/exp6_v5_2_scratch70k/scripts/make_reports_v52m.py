"""EXP6 V5.2-M -- report builder (seen2scene env, CPU). Called by overnight_driver.py after every stage; idempotent; every
section is built only from files that exist (missing stages are written as pending / skipped, nothing is invented).
Writes TRAINING_LOG_SUMMARY.md, G1_FIRSTHIT.md, G2_FIXED_STATE.md, G3_SMOKE.md, G3_FULL.md, FINAL_REPORT.md (automatic
part), STATUS.md and figures/. Usage: python make_reports_v52m.py"""
import glob
import json
import os
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
W = os.path.join(R, "work_dirs", "exp6_v5_2_scratch70k")
EV, CK = os.path.join(W, "eval"), os.path.join(W, "checkpoints")
FIG = os.path.join(E, "figures")
STEPS = 70000


def jload(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def jl(p):
    out = []
    if os.path.exists(p):
        for x in open(p):
            try:
                out.append(json.loads(x))
            except Exception:
                pass
    return out


def f3(x, n=3):
    return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def training_summary(st):
    L = jl(os.path.join(W, "log.jsonl"))
    H = jl(os.path.join(W, "holdout.jsonl"))
    lm = jload(os.path.join(CK, "last_manifest.json"), {})
    best = jload(os.path.join(CK, "best_train_holdout.json"), {})
    out = ["# V5.2-M Scratch70k —— 训练日志摘要（自动生成）", "", f"生成时间 {time.strftime('%F %T')}。已完成的同步 optimizer step：**{lm.get('step', 0):,} / {STEPS:,}**"
           f"（seen samples = {lm.get('step', 0) * 64:,} views；全局 batch 64 views / 8 states）。", ""]
    if st.get("rounds"):
        out += ["## 训练轮次（弹性 DDP）", "", "| 轮次 | 节点 | 开始 | 结束 | step | exit |", "|---|---|---|---|---|---|"]
        for r in st["rounds"]:
            out.append(f"| {r['round']} | {' + '.join(r.get('hosts', []))}（{4 * r['nnodes']} rank） | {r['start']} | {r.get('end', '运行中')} | {r['step_before']:,} → {r.get('step_after', '…')} | {r.get('exit', '…')} |")
        out.append("")
    if L:
        by_world = {}
        for x in L:
            if x["step"] > 20:
                by_world.setdefault(x.get("world"), []).append(x["s_per_step"])
        out += ["## 吞吐", "", "| world size | 记录数 | s/step（中位数） | views/s | 估算 70k 所需（h） |", "|---|---|---|---|---|"]
        for w, v in sorted(by_world.items()):
            m = float(np.median(v))
            out.append(f"| {w} | {len(v)} | {m:.3f} | {64 / m:.1f} | {STEPS * m / 3600:.1f} |")
        out += ["", f"峰值显存 {max(x.get('mem_gb', 0) for x in L):.1f} GB / rank；数据等待 {np.median([x.get('data_s_per_step', 0) for x in L]):.4f} s/step（中位数）。", ""]
        rows = [x for x in L if x["step"] % 5000 == 0 or x["step"] in (1, 500, 1000, 2000)]
        out += ["## 训练曲线（训练 batch，含补采样，不是校准）", "", "| step | loss | L_surface | L_firsthit | NLL UNKNOWN | 首表面召回 | 第二+召回 | lr | grad norm |", "|---|---|---|---|---|---|---|---|---|"]
        for x in rows:
            out.append(f"| {x['step']:,} | {x['loss']:.3f} | {x['surf']:.3f} | {x['firsthit']:.3f} | {f3(x.get('nll_unknown'))} | {f3(x.get('surf_recall_first'))} | {f3(x.get('surf_recall_second_plus'))} | {x['lr']:.2e} | {x['grad_norm']:.2f} |")
        sh = [x for x in L if "share_all_unknown_hit" in x]
        if sh:
            keys = ("known_free", "known_band", "unknown_hit", "unknown_nohit", "near_hit", "far_hit")
            out += ["", "### 采样分层份额（训练首中射线；nat = 自然比例的均匀射线）", "", "| 组 | " + " | ".join(keys) + " | 第二+表面标签射线 |", "|---|" + "---|" * (len(keys) + 1)]
            for g in ("all", "nat"):
                out.append(f"| {g} | " + " | ".join(f3(np.mean([x[f'share_{g}_{k}'] for x in sh])) for k in keys) + f" | {f3(np.mean([x['share_label_second_plus'] for x in sh]))} |")
        out.append("")
    if H:
        out += ["## G1 Train-6 S-c holdout（EMA；每 5k；选择分数预注册）", "",
                "| step | NLL UNK | NLL nat | Top UNK | p_nohit>.95 真命中率 UNK | 已知穿透 | no-hit ECE(nat) | 预测/实际 no-hit(nat) | 首表面召回 | 第二+召回 | 深遮挡召回 | 众数误差 UNK vox | select |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in H:
            out.append(f"| {x['step']:,} | {f3(x.get('nll_unknown'))} | {f3(x.get('nll_nat'))} | {f3(x.get('top_unknown'))} | {f3(x.get('hc_false_nohit_unknown'), 4)} | {f3(x.get('known_passthrough'))} | "
                       f"{f3(x.get('nohit_ece_nat'))} | {f3(x.get('pred_nohit_nat'))} / {f3(x.get('true_nohit_nat'))} | {f3(x.get('surf_recall_first'))} | {f3(x.get('surf_recall_second_plus'))} | "
                       f"{f3(x.get('surf_recall_far128'))} | {f3(x.get('mode_err_vox_unknown'), 1)} | {f3(x.get('select_score'))} |")
        out += ["", f"当前 best_train_holdout：step {best.get('step')}，select {f3(best.get('select_score'))}。", ""]
    open(os.path.join(E, "TRAINING_LOG_SUMMARY.md"), "w").write("\n".join(out) + "\n")
    if L and H:
        try:
            figure(L, H)
        except Exception:
            traceback.print_exc()


def figure(L, H):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    s = np.array([x["step"] for x in L])
    for k, c in (("surf", "#1971c2"), ("firsthit", "#e8590c")):
        v = np.array([x[k] for x in L])
        ax[0].plot(s, v, color=c, lw=1, label=k)
    ax[0].set_title("training loss (batch)"); ax[0].legend(frameon=False); ax[0].set_xlabel("step")
    hs = [x["step"] for x in H]
    for k, c in (("nll_unknown", "#2b8a3e"), ("nll_nat", "#868e96")):
        ax[1].plot(hs, [x.get(k) for x in H], "o-", color=c, ms=4, lw=2, label=k)
    ax[1].set_title("Train-6 S-c holdout first-hit NLL"); ax[1].legend(frameon=False); ax[1].set_xlabel("step")
    for k, c in (("hc_false_nohit_unknown", "#c92a2a"), ("known_passthrough", "#5f3dc4"), ("surf_recall_second_plus", "#1971c2")):
        ax[2].plot(hs, [x.get(k) for x in H], "o-", color=c, ms=4, lw=2, label=k)
    ax[2].set_title("holdout physics / recall"); ax[2].legend(frameon=False); ax[2].set_xlabel("step")
    for a in ax:
        a.grid(alpha=0.25); a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training_dashboard.png"), dpi=120); plt.close(fig)


def g2_report(st):
    out = ["# V5.2-M G2 固定状态（只读诊断；自动生成）", "",
           "每个 checkpoint 用自己在 Train-6 S-c 上校准的 floor；λ ∈ {0, 0.1}、η 0.5 固定；与 V5 / V5.2 同一状态、同一候选。增益 = 被选相机真实一步覆盖增益（pp）。", ""]
    seq = [("milestone_20k", "20k 里程碑"), ("milestone_40k", "40k 里程碑"), ("final_last70k", "last70k"), ("final_best", "best_train_holdout")]
    traj = []
    for name, lab in seq:
        g = jload(os.path.join(EV, name, "g12.json"))
        if not g:
            out += [f"## {lab}", "", "未运行 / 运行中。", ""]
            continue
        S = g["summary"]
        row = {"name": lab}
        for d in ("dev3", "valsel"):
            for m in ("V52ML0", "V52ML01", "V52L0", "V5"):
                row[f"{d}_{m}_top1"] = S.get(d, {}).get("all", {}).get("means", {}).get(f"{m}_top1", {}).get("mean")
                row[f"{d}_{m}_gain"] = S.get(d, {}).get("all", {}).get("means", {}).get(f"{m}_gain", {}).get("mean")
        traj.append(row)
        md = open(os.path.join(EV, name, "g12.md")).read().split("\n", 2)[-1]
        out += [f"## {lab}", "", md, ""]
    if traj:
        out[4:4] = ["## 里程碑走势（全部状态均值）", "", "| checkpoint | Dev-3 Top-1 λ0 / λ0.1 | Dev-3 增益 λ0 | MP3D Top-1 λ0 / λ0.1 | MP3D 增益 λ0 | 旧 V5.2 λ0 Top-1 Dev-3 / MP3D | V5 Top-1 Dev-3 / MP3D |", "|---|---|---|---|---|---|---|"] + [
            f"| {r['name']} | {f3(r['dev3_V52ML0_top1'])} / {f3(r['dev3_V52ML01_top1'])} | {f3(r['dev3_V52ML0_gain'])} | {f3(r['valsel_V52ML0_top1'])} / {f3(r['valsel_V52ML01_top1'])} | {f3(r['valsel_V52ML0_gain'])} | "
            f"{f3(r['dev3_V52L0_top1'])} / {f3(r['valsel_V52L0_top1'])} | {f3(r['dev3_V5_top1'])} / {f3(r['valsel_V5_top1'])} |" for r in traj] + [""]
    open(os.path.join(E, "G2_FIXED_STATE.md"), "w").write("\n".join(out) + "\n")
    return traj


BELIEF_SETS = (("valsel/B0", "MP3D VALSEL"), ("dev3/B0", "Dev-3 B0"), ("dev3/B2", "Dev-3 B2"))
OLD_BELIEF = os.path.join(R, "experiments", "exp6_v5_2", "reports", "belief_metrics_eval.json")


def launch_belief(st):
    """Start the evaluation-side multi-surface belief metrics of every evaluated checkpoint once (detached, GPU 3 of the
    head node; ~10 min each), so this report builder itself stays fast."""
    for name in ("milestone_20k", "milestone_40k", "final_last70k", "final_best"):
        od = os.path.join(EV, name)
        mf = sorted(glob.glob(os.path.join(od, "fixed", "*", "*", "models_0.json")))
        mj = (jload(mf[0]) or {}) if mf else {}
        if not os.path.exists(os.path.join(od, "DONE")) or os.path.exists(os.path.join(od, "belief.started")) or not mj.get("ckpt"):
            continue
        open(os.path.join(od, "belief.started"), "w").write(time.strftime("%F %T"))
        subprocess.Popen(["bash", os.path.join(HERE, "belief_eval_v52m.sh"), od, mj["ckpt"], "3"], cwd=R, stdout=open(os.path.join(od, "belief.log"), "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)


def belief_rows(bm, key_of):
    out = []
    for sub, lab in BELIEF_SETS:
        k = key_of(sub)
        if not bm or k not in bm.get("metrics", {}):
            continue
        for grp in ("all", "unknown_rays"):
            x = bm["metrics"][k].get(grp)
            if not x:
                continue
            gap = x.get("recall_second_plus_by_gap", {}).get("gap>=128vox", {})
            out.append((lab, grp, x))
    return out


def g1_report(st):
    H = jl(os.path.join(W, "holdout.jsonl"))
    out = ["# V5.2-M G1 First-Hit / Geometry（自动生成）", "", "两部分：(1) Train-6 S-c holdout（训练中每 5k；选择依据）；(2) 评估固定状态上所有候选视角的首中（相对 oracle t*），与旧 V5.2 同口径。", ""]
    if H:
        x = H[-1]
        out += [f"## (1) Train-6 S-c holdout 最新（step {x['step']:,}）", "", "完整走势见 `TRAINING_LOG_SUMMARY.md` 与 `figures/training_dashboard.png`。", ""]
    for name, lab in (("final_best", "best_train_holdout"), ("final_last70k", "last70k"), ("milestone_40k", "40k"), ("milestone_20k", "20k")):
        p = os.path.join(EV, name, "g12.md")
        if os.path.exists(p):
            txt = open(p).read()
            out += [f"## (2) 评估固定状态首中 —— {lab}", ""]
            for blk in txt.split("## ")[1:]:
                head = blk.split("\n", 1)[0]
                tab = [ln for ln in blk.splitlines() if ln.startswith("| First-Hit") or ln.startswith("|---|---|---|---|---|---|---|---|---|---|") or ln.startswith("| V5.2-M") or ln.startswith("| 旧 V5.2")]
                out += [f"### {head}", ""] + tab + [""]
            break
    out += ["## (3) 评估场景多表面 belief（网格深度剥离 GT，仅评估端；每候选视角 256 条射线，UNKNOWN 优先；与旧 V5.2 同口径）", "",
            "| 模型 | 数据 | 射线组 | 射线 | 首表面召回 | 第二+召回 | 深遮挡（间隔 ≥128 vox）召回 | 网格 precision / recall | 网格 ECE | 首表面后虚假表面率 |", "|---|---|---|---|---|---|---|---|---|---|"]
    have = False
    for name, lab in (("final_best", "V5.2-M best"), ("final_last70k", "V5.2-M last70k"), ("milestone_40k", "V5.2-M 40k"), ("milestone_20k", "V5.2-M 20k")):
        bm = jload(os.path.join(EV, name, "belief_metrics.json"))
        if not bm:
            continue
        have = True
        for dl, grp, x in belief_rows(bm, lambda sub, name=name: os.path.join(EV, name, "fixed", sub)):
            g = x.get("recall_second_plus_by_gap", {}).get("gap>=128vox", {})
            out.append(f"| {lab} | {dl} | {grp} | {x['rays']:,} | {f3(x.get('recall_first'))} | {f3(x.get('recall_second_plus'))} | {f3(g.get('recall'))}（{g.get('n', 0):,}） | "
                       f"{f3(x.get('grid_precision'))} / {f3(x.get('grid_recall'))} | {f3(x.get('grid_ece'))} | {f3(x.get('false_surface_rate_behind_first'))} |")
    ob = jload(OLD_BELIEF)
    mp = {"valsel/B0": "work_dirs/exp6_v5_2/eval_fixed/valsel_s_final/B0", "dev3/B0": "work_dirs/exp6_v5_2/eval_fixed/dev3_s_final/B0", "dev3/B2": "work_dirs/exp6_v5_2/eval_fixed/dev3_s_final/B2"}
    for dl, grp, x in belief_rows(ob, lambda sub: mp[sub]):
        g = x.get("recall_second_plus_by_gap", {}).get("gap>=128vox", {})
        out.append(f"| 旧 V5.2（V5 belief） | {dl} | {grp} | {x['rays']:,} | {f3(x.get('recall_first'))} | {f3(x.get('recall_second_plus'))} | {f3(g.get('recall'))}（{g.get('n', 0):,}） | "
                   f"{f3(x.get('grid_precision'))} / {f3(x.get('grid_recall'))} | {f3(x.get('grid_ece'))} | {f3(x.get('false_surface_rate_behind_first'))} |")
    if not have:
        out.append("")
        out.append("V5.2-M 的多表面 belief 评估尚未完成（每个评估 checkpoint 的 G2 结束后自动启动）。")
    out.append("")
    open(os.path.join(E, "G1_FIRSTHIT.md"), "w").write("\n".join(out) + "\n")


def g3_reports():
    res = {}
    for kind in ("smoke", "full"):
        scr = os.path.join(HERE, f"eval_g3_{kind}_v52m.py")
        if os.path.exists(scr) and glob.glob(os.path.join(EV, "g3", "closed_loop", "*", "V52ML0*")):
            r = subprocess.run([sys.executable, scr], cwd=R, capture_output=True, text=True, timeout=1800)
            res[kind] = r.returncode
            if r.returncode != 0:
                print(r.stderr[-3000:])
    return res


def status_md(st, traj):
    dec = jload(os.path.join(E, "AUTO_DECISION.json"), {})
    lm = jload(os.path.join(CK, "last_manifest.json"), {})
    mk = {"done": "✅", "running": "🔄", "pending": "⏳", "failed": "❌", "skipped": "⬜"}
    rows = [("P1", "Scratch70k 训练", f"{lm.get('step', 0):,} / 70,000 step；状态 {st.get('state')}；阶段 {st.get('phase')}")]
    for k, v in sorted(st.get("milestones", {}).items(), key=lambda kv: int(kv[0])):
        rows.append((f"P2.{int(k) // 20000}", f"里程碑 G2 {int(k) // 1000}k", f"{mk.get(v.get('state'), v.get('state'))} {v.get('start', '')} → {v.get('end', '')}"))
    fin = st.get("final", {})
    if fin:
        rows.append(("P3", "最终 G1/G2（last70k + best）", f"best step {fin.get('best_step')}；主权重 {fin.get('primary', {}).get('name')}"))
    if dec:
        rows.append(("P4", "门槛 / 决策", f"{dec.get('g2_gate')} → {dec.get('decision')}"))
    for k, v in st.get("g3", {}).items():
        rows.append((f"P5.{k}", f"G3 {k}", f"{mk.get(v.get('state'), v.get('state'))} {v.get('start', '')} → {v.get('end', '')} {v.get('reason', '')}"))
    out = ["# EXP6 V5.2-M Scratch70k —— 状态板（驱动脚本自动更新）", "", f"**最近更新：{time.strftime('%F %T')}**　标记：✅ 完成 · 🔄 正在跑 · ⏳ 等待 · ❌ 失败 · ⬜ 跳过", "",
           "## GPU / 作业", "", "| 作业 | 节点 | 到期 | 用途 |", "|---|---|---|---|",
           "| 3443137 `vscode-dev`（交互分配，头节点） | gpu115 4 × L40S | 11-04 15:11 | 驱动、训练 rank 0–3、里程碑 G2、最终评估 |",
           "| 3449325 `v52m-nodeB`（sbatch） | gpu112 4 × L40S | 11-04 03:00 | 训练 rank 4–7、最终评估 / G3（结束后释放） |", "",
           "## 进度", "", "| 编号 | 内容 | 状态 |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in rows] + ["", "## 事件（最近 15 条）", ""] + [f"- {e}" for e in st.get("events", [])[-15:]] + [""]
    open(os.path.join(E, "STATUS.md"), "w").write("\n".join(out) + "\n")


def main():
    st = jload(os.path.join(E, "RUN_STATUS.json"), {}) or {}
    traj = []
    for fn in (training_summary, g1_report):
        try:
            fn(st)
        except Exception:
            traceback.print_exc()
    try:
        launch_belief(st)
    except Exception:
        traceback.print_exc()
    try:
        traj = g2_report(st)
    except Exception:
        traceback.print_exc()
    try:
        g3_reports()
    except Exception:
        traceback.print_exc()
    try:
        import final_report_v52m as FR
        FR.write(st, traj)
    except ImportError:
        pass
    except Exception:
        traceback.print_exc()
    status_md(st, traj)


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    main()
