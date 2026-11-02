"""EXP6 V5.2-M -- automatic part of FINAL_REPORT.md (imported by make_reports_v52m.py). Every number is read from the
stage outputs (RUN_STATUS.json, checkpoints/*.json, eval/*/g12.json, reports/g3_*.json); a missing stage is written as
missing. The three questions of the task (MP3D UNKNOWN over-confident no-hit, Outdoor G2 Top-1, G3) are answered
mechanically from these numbers. A hand-written interpretation, if any, lives in FINAL_REPORT_NOTES.md and is appended."""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
E = os.path.dirname(HERE)
R = os.path.dirname(os.path.dirname(E))
W = os.path.join(R, "work_dirs", "exp6_v5_2_scratch70k")
EV, CK = os.path.join(W, "eval"), os.path.join(W, "checkpoints")
BENCH = {"V5": {"outdoor": (0.711, 0.891), "mp3d": (0.333, 0.501)}, "old V5.2 λ0": {"outdoor": (0.608, 0.781), "mp3d": (0.238, 0.310)},
         "B2": {"outdoor": (0.730, 0.906)}, "E2-D100": {"mp3d": (0.364, 0.553)}}


def jload(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(p):
    return "—" if not p else f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]（{p.get('scenes')} 场景）"


def mean(S, d, key):
    return S.get(d, {}).get("all", {}).get("means", {}).get(key, {}).get("mean")


def write(st, traj):
    dec = jload(os.path.join(E, "AUTO_DECISION.json"), {})
    fin = st.get("final", {})
    prim = (fin.get("primary") or {}).get("name")
    g = jload(os.path.join(EV, prim, "g12.json")) if prim else None
    lm = jload(os.path.join(CK, "last_manifest.json"), {})
    best = jload(os.path.join(CK, "best_train_holdout.json"), {})
    H = [json.loads(x) for x in open(os.path.join(W, "holdout.jsonl"))] if os.path.exists(os.path.join(W, "holdout.jsonl")) else []
    rounds = st.get("rounds", [])
    gpu_h = 0.0
    for r in rounds:
        try:
            t0 = time.mktime(time.strptime(r["start"], "%Y-%m-%d %H:%M:%S"))
            t1 = time.mktime(time.strptime(r["end"], "%Y-%m-%d %H:%M:%S")) if r.get("end") else time.time()
            gpu_h += (t1 - t0) / 3600 * 4 * r["nnodes"]
        except Exception:
            pass
    L = ["# EXP6 V5.2-M Scratch70k —— 最终报告", "", f"自动生成于 {time.strftime('%F %T')}（`scripts/final_report_v52m.py`，数字全部读自阶段输出；评估场景非盲）。人工解读见文末（若有）。", "",
         "## 0. 运行事实", "",
         f"- 状态：**{st.get('state')}**（阶段 {st.get('phase')}）；完成的同步 optimizer step **{lm.get('step', 0):,} / 70,000**；seen samples {lm.get('step', 0) * 64:,} views（全局 batch 64 views / 8 states）。",
         f"- 训练轮次 {len(rounds)}：" + "；".join(f"{r['round']} {4 * r['nnodes']} rank {r['step_before']:,}→{r.get('step_after', '…')} exit {r.get('exit', '…')}" for r in rounds) + f"；训练 GPU-h ≈ {gpu_h:.1f}。",
         f"- Slurm：头节点交互分配 {st.get('head', {}).get('job')}（{st.get('head', {}).get('host')}），节点 B sbatch " + ", ".join(sorted({j for r in rounds for j in r.get('jobs', [])[1:]})) + "。",
         f"- Git：启动时 {st.get('git_sha_launch', '—')[:12]}，报告提交 {(st.get('git_sha_reports') or '—')[:12]}。",
         f"- 权重 sha256：" + ("；".join(f"`{k}` {v[:16]}…" for k, v in fin.get("sha256", {}).items()) or "—") + f"；best_train_holdout = step {best.get('step')}（select {f3(best.get('select_score'))}）；主权重 {prim or '—'}。", ""]
    if st.get("blocker"):
        L += [f"- **阻断**：{st['blocker']}", ""]
    if H:
        h0, h1 = H[0], H[-1]
        L += ["## 1. Train-6 S-c holdout（选择依据；自然比例校准）", "",
              "| step | NLL UNK | Top UNK | p_nohit>.95 真命中率 UNK | 已知穿透 | no-hit ECE(nat) | 第二+召回 | 深遮挡召回 |", "|---|---|---|---|---|---|---|---|"]
        for x in (h0, *(x for x in H if x["step"] in (10000, 20000, 40000)), h1):
            L.append(f"| {x['step']:,} | {f3(x.get('nll_unknown'))} | {f3(x.get('top_unknown'))} | {f3(x.get('hc_false_nohit_unknown'))} | {f3(x.get('known_passthrough'))} | {f3(x.get('nohit_ece_nat'))} | "
                     f"{f3(x.get('surf_recall_second_plus'))} | {f3(x.get('surf_recall_far128'))} |")
        L.append("")
    L += ["## 2. 三个问题（机械判读）", ""]
    if g:
        S = g["summary"]
        new_in, old_in = g["g1_new"].get("valsel", {}).get("unknown", {}), g["g1_old_v52"].get("valsel", {}).get("unknown", {})
        new_out, old_out = g["g1_new"].get("dev3", {}).get("unknown", {}), g["g1_old_v52"].get("dev3", {}).get("unknown", {})

        def fn(x):
            return None if not x.get("bin_09_1_rays") else 1 - x["bin_09_1_actual_nohit"]
        q1 = (fn(old_in) is not None) and (fn(new_in) is None or fn(new_in) < fn(old_in))
        L += [f"**Q1 MP3D UNKNOWN 过度自信 no-hit 是否减少：{'是' if q1 else '否'}。** VALSEL UNKNOWN 射线中 p_nohit ∈ [0.9, 1] 的射线："
              f"旧 V5.2 {old_in.get('bin_09_1_rays', 0):,} 条，其中实际命中 {f3(fn(old_in))}；V5.2-M {new_in.get('bin_09_1_rays', 0):,} 条，实际命中 {f3(fn(new_in))}。"
              f"V5.2-M p_nohit > 0.95 占真实命中射线的比例 {f3(new_in.get('hc_false_nohit_rate'))}；UNKNOWN 首中 NLL 旧 {f3(old_in.get('nll'))} → 新 {f3(new_in.get('nll'))}，最高区间准确率 {f3(old_in.get('top_acc'))} → {f3(new_in.get('top_acc'))}。"
              f"（Dev-3 同口径：旧 {f3(fn(old_out))}（{old_out.get('bin_09_1_rays', 0):,} 条）→ 新 {f3(fn(new_out))}（{new_out.get('bin_09_1_rays', 0):,} 条）；NLL {f3(old_out.get('nll'))} → {f3(new_out.get('nll'))}。）", ""]
        t_new, t_new1, t_old, t_v5 = (mean(S, "dev3", f"{m}_top1") for m in ("V52ML0", "V52ML01", "V52L0", "V5"))
        p = S.get("dev3", {}).get("all", {}).get("paired", {})
        L += [f"**Q2 Outdoor G2 Top-1 是否提升：{'是' if (t_new or 0) > (t_old or 1) else '否'}（相对旧 V5.2）。** Dev-3 275 态 Top-1：V5.2-M λ0 {f3(t_new)} / λ0.1 {f3(t_new1)}；旧 V5.2 λ0 {f3(t_old)}；V5 {f3(t_v5)}；"
              f"配对差 V5.2-M λ0 − 旧 V5.2 λ0 = {ci(p.get('V52ML0-V52L0|top1'))}，− V5 = {ci(p.get('V52ML0-V5|top1'))}；增益（pp）配对差 − 旧 V5.2 {ci(p.get('V52ML0-V52L0|gain'))}，− V5 {ci(p.get('V52ML0-V5|gain'))}。"
              f"MP3D VALSEL Top-1：V5.2-M λ0 {f3(mean(S, 'valsel', 'V52ML0_top1'))}，旧 V5.2 {f3(mean(S, 'valsel', 'V52L0_top1'))}，V5 {f3(mean(S, 'valsel', 'V5_top1'))}。", ""]
    else:
        L += ["G2 最终评估尚无结果，Q1 / Q2 暂不能回答。", ""]
    sm = jload(os.path.join(E, "reports", "g3_smoke_v52m.json"))
    full = jload(os.path.join(E, "reports", "g3_v52m_full.json"))
    if sm:
        m = sm.get("mean", {})
        L += [f"**Q3 G3 是否改善（smoke，4 Outdoor + 2 MP3D episode，探索性）。** smoke 有效：{'是' if sm.get('valid') else '否'}。", "",
              "| 域 | V5.2-M λ0 | V5.2-M λ0.1 | 旧 V5.2 λ0 | 旧 V5.2 λ0.1 | V5 | V4C | B2 | E2-D100 |", "|---|---|---|---|---|---|---|---|---|"]
        for d, nm in (("outdoor", "Outdoor Test-3（AUC / Final / Cov20）"), ("indoor", "MP3D VALSEL")):
            L.append(f"| {nm} | " + " | ".join((f"{m[d][k]['auc']:.3f} / {m[d][k]['final']:.3f} / {m[d][k]['cov20']:.3f}" if k in m.get(d, {}) else "—")
                                               for k in ("V52ML0", "V52ML01", "V52L0", "V52L01", "V5", "V4C", "B2", "E2D")) + " |")
        L.append("")
    else:
        L += ["**Q3 G3**：smoke 尚无结果。", ""]
    if full:
        L += ["**G3 Full**（Outdoor 30 × 100、MP3D 10 × 150；分场景 / 楼层 bootstrap）：详见 `reports/g3_v52m_full.md`。", ""]
        try:
            txt = open(os.path.join(E, "reports", "g3_v52m_full.md")).read()
            L += [txt.split("\n", 1)[1], ""]
        except Exception:
            pass
    elif dec:
        L += [f"**G3 Full**：{(st.get('g3', {}).get('full') or {}).get('state', '未运行')} —— {dec.get('decision')}。", ""]
    L += ["## 3. 门槛与自动决策（FROZEN_PROTOCOL §5）", "", "```json", json.dumps(dec, ensure_ascii=False, indent=1, default=str) if dec else "{}", "```", "",
          "## 4. G3 用户基准（以往实验，AUC / Final）", "", "| 模型 | Outdoor | MP3D |", "|---|---|---|"]
    for k, v in BENCH.items():
        L.append(f"| {k} | {' / '.join(f'{x:.3f}' for x in v['outdoor']) if 'outdoor' in v else '—'} | {' / '.join(f'{x:.3f}' for x in v['mp3d']) if 'mp3d' in v else '—'} |")
    L += ["", "## 5. 明细文件", "", "- 训练：`TRAINING_LOG_SUMMARY.md`、`figures/training_dashboard.png`", "- G1：`G1_FIRSTHIT.md`；G2：`G2_FIXED_STATE.md`（20k / 40k / 70k / best）",
          "- G3：`G3_SMOKE.md`、`G3_FULL.md`；决策：`AUTO_DECISION.json`；状态：`RUN_STATUS.json`、`STATUS.md`",
          "- 复现：`scripts/submit.sh`（续跑 / 节点 B）、`scripts/overnight_driver.py`、`scripts/slurm/*.sbatch`；冻结协议 `FROZEN_PROTOCOL.md`", "",
          "**归因限制**：容量、从零训练、2D 上下文、训练分布与几何约束同时改变；结论只针对这套联合干预。", ""]
    notes = os.path.join(E, "FINAL_REPORT_NOTES.md")
    if os.path.exists(notes):
        L += ["## 6. 人工解读", "", open(notes).read()]
    open(os.path.join(E, "FINAL_REPORT.md"), "w").write("\n".join(L) + "\n")
    if not (st.get("g3", {}).get("full") or {}).get("state") == "running" and not full:
        fs = (st.get("g3", {}).get("full") or {})
        open(os.path.join(E, "G3_FULL.md"), "w").write(
            "# V5.2-M G3 Full\n\n" + (f"**跳过**：{fs.get('reason')}（FROZEN_PROTOCOL §5 冻结门槛；见 `AUTO_DECISION.json`）。\n" if fs.get("state") == "skipped" else "尚未运行。\n"))
    elif full:
        txt = open(os.path.join(E, "reports", "g3_v52m_full.md")).read()
        open(os.path.join(E, "G3_FULL.md"), "w").write("# V5.2-M G3 Full（自动生成；评估场景非盲）\n\n" + txt.split("\n", 1)[1])
