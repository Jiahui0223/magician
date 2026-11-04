"""EXP6 V3 overnight Y + D -- generated tables / figures for the morning reports (CPU; read-only inputs).
Data: CVPR Table 1 canonical records (work_dirs/cvpr_results/table1/episodes) for H0 (V3), V3-H3, the D rows, the Y rows, B0 / B2 /
MAG-B; planner step logs (runtime / memory); Y run dir (log.jsonl, eval.jsonl, checkpoint_manifest.json, STATUS.json) and the
parent 40K eval (task1_xattn/runs/main/eval.jsonl). Official coverage; AUC@k = trapz(C[0..k]) / k; Final = C[100]; scene = mean
of its 5 starts; split = scene-equal mean over COMPLETE scenes; a split with missing scenes is labelled partial (n/N scenes)
and never called complete; paired differences on the scenes complete for both models, scene-level bootstrap 95% CI (10k, seed 0).
Outputs results/overnight_summary.json, Y/reports/{G3_DEV_AND_ALL14,TRAINING_TIMELINE,VALIDATION_G1_G2}.md,
D/reports/G3_DEV_AND_ALL14.md, experiments/exp6_v3/overnight/{OVERNIGHT_TABLES.md, figures/*}."""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
YX = os.path.dirname(HERE)
X3 = os.path.dirname(YX)
R = os.path.dirname(os.path.dirname(X3))
DX = os.path.join(X3, "ablation_study", "experiment_D_h3_lookahead")
OV = os.path.join(X3, "overnight")
E = os.path.join(R, "work_dirs", "cvpr_results", "table1", "episodes")
YW = os.path.join(R, "work_dirs", "exp6_v3", "Y")
SPLIT = {"train": ["bannerman", "dunnottar", "fushimi", "liberty", "pisa", "redeemer"], "dev": ["barts", "colosseum", "eiffel", "pantheon", "sestino_museum"],
         "test": ["alhambra", "bridge", "neuschwanstein"]}
SPLIT["all14"] = sorted(SPLIT["train"] + SPLIT["dev"] + SPLIT["test"])
MODELS = {"H0 (V3-T1 40K)": "V3", "H3 (historical)": "V3-H3", "D1 H3-NoRisk": "V3-D1-H3-NORISK", "D2 Depth3-B3": "V3-D2-DEPTH3-B3",
          "D3 Depth3-B5": "V3-D3-DEPTH3-B5", "D4 Depth4-B3": "V3-D4-DEPTH4-B3", "Y45K + H0": "V3-Y45K-H0", "Y50K + H0": "V3-Y50K-H0",
          "YBEST + H3": "V3-YBEST-H3", "B0": "B0", "B2": "B2", "MAGICIAN Beam": "MAG-B"}
PAIRS = [("V3-D1-H3-NORISK", "V3-H3"), ("V3-D2-DEPTH3-B3", "V3-D1-H3-NORISK"), ("V3-D3-DEPTH3-B5", "V3-D2-DEPTH3-B3"), ("V3-D4-DEPTH4-B3", "V3-D2-DEPTH3-B3"),
         ("V3-D2-DEPTH3-B3", "V3-H3"), ("V3-D3-DEPTH3-B5", "V3-H3"), ("V3-D1-H3-NORISK", "V3"), ("V3-D2-DEPTH3-B3", "V3"),
         ("V3-Y45K-H0", "V3"), ("V3-Y50K-H0", "V3"), ("V3-Y50K-H0", "V3-Y45K-H0"), ("V3-YBEST-H3", "V3-H3"), ("V3-YBEST-H3", "V3")]
KEYS = ("auc10", "auc20", "auc50", "auc100", "cov20", "cov50", "final")


def met(c):
    c = np.asarray(c, float)
    d = {f"auc{k}": float(np.trapz(c[:k + 1]) / k) for k in (10, 20, 40, 50, 60, 80, 100)}
    d.update({f"cov{k}": float(c[k]) for k in (10, 20, 40, 50, 60, 80, 100)})
    d["final"] = float(c[100])
    return d


def boot(v, n=10000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    b = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def load():
    rec = {}
    for mid in set(MODELS.values()):
        for f in glob.glob(os.path.join(E, mid, "*.json")):
            r = json.load(open(f))
            rec[(mid, r["scene"], r["start"])] = r
    return rec


def scene_mean(M, mid, s):
    if all((mid, s, k) in M for k in range(5)):
        return {q: float(np.mean([M[(mid, s, k)][q] for k in range(5)])) for q in KEYS}
    return None


def runtime(rec, mid):
    dec, mem = [], []
    for (m, s, k), r in rec.items():
        if m != mid:
            continue
        ep = os.path.join(R, r["episode_file"])
        for f in glob.glob(os.path.join(os.path.dirname(os.path.dirname(ep)), "planner_logs", f"pl_V2M_{s}_s{k}_r0.steps.jsonl")):
            for l in open(f):
                x = json.loads(l)
                dec.append(x["decision_s"]); mem.append(x["peak_mem_gb"])
    return (float(np.median(dec)), float(np.quantile(dec, 0.9)), float(max(mem))) if dec else (None, None, None)


def main():
    rec = load()
    M = {k: met(r["coverage_official"]) for k, r in rec.items()}
    out = {"models": {}, "pairs": {}}
    for nm, mid in MODELS.items():
        row = {"id": mid, "splits": {}}
        for g, sc in SPLIT.items():
            sm = {s: scene_mean(M, mid, s) for s in sc}
            done = [s for s in sc if sm[s]]
            n_ep = sum((mid, s, k) in M for s in sc for k in range(5))
            row["splits"][g] = {"episodes": n_ep, "episodes_total": 5 * len(sc), "scenes_complete": len(done), "scenes_total": len(sc),
                                "complete": len(done) == len(sc), **{q: (float(np.mean([sm[s][q] for s in done])) if done else None) for q in KEYS},
                                "missing": [f"{s}_s{k}" for s in sc for k in range(5) if (mid, s, k) not in M]}
        row["runtime"] = runtime(rec, mid) if mid not in ("B0", "B2", "MAG-B") else (None, None, None)
        out["models"][nm] = row
    for a, b in PAIRS:
        for g, sc in SPLIT.items():
            both = [s for s in sc if scene_mean(M, a, s) and scene_mean(M, b, s)]
            if not both:
                continue
            q = {"n_scenes": len(both), "of": len(sc)}
            for key in ("auc20", "auc100", "final"):
                d = [scene_mean(M, a, s)[key] - scene_mean(M, b, s)[key] for s in both]
                q[key] = {"d": float(np.mean(d)), "ci": boot(d), "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d))}
            out["pairs"][f"{a} − {b} | {g}"] = q
    os.makedirs(os.path.join(YX, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(YX, "results", "overnight_summary.json"), "w"), indent=1)
    f3 = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731

    def cell(sp):
        tag = "" if sp["complete"] else f" (partial {sp['scenes_complete']}/{sp['scenes_total']} sc)"
        return f"{sp['episodes']}/{sp['episodes_total']}", (f"{f3(sp['auc20'])} / {f3(sp['auc100'])} / {f3(sp['final'])}{tag}" if sp["scenes_complete"] else "—")
    T = ["| Run | model ID | Train ep | Dev ep | Test ep | All14 ep | Train AUC20/AUC100/Final | Dev AUC20/AUC100/Final | Test AUC20/AUC100/Final | All14 AUC20/AUC100/Final | s/step P50 / P90 | peak VRAM GB |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for nm, row in out["models"].items():
        sp = row["splits"]
        if sp["all14"]["episodes"] == 0:
            continue
        c = {g: cell(sp[g]) for g in ("train", "dev", "test", "all14")}
        rt = row["runtime"]
        T.append(f"| {nm} | {row['id']} | {c['train'][0]} | {c['dev'][0]} | {c['test'][0]} | {c['all14'][0]} | {c['train'][1]} | {c['dev'][1]} | {c['test'][1]} | {c['all14'][1]} | "
                 f"{'—' if rt[0] is None else f'{rt[0]:.1f} / {rt[1]:.1f}'} | {'—' if rt[2] is None else f'{rt[2]:.1f}'} |")
    P = ["| A − B | 组 | 场景 | ΔAUC@20 | ΔAUC@100 | ΔFinal |", "|---|---|---|---|---|---|"]

    def fd(z):
        st = " ✱" if z["ci"][0] is not None and (z["ci"][0] > 0 or z["ci"][1] < 0) else ""
        ci = f"[{z['ci'][0]:+.3f}, {z['ci'][1]:+.3f}]" if z["ci"][0] is not None else "[—]"
        return f"{z['d']:+.3f} {ci} {z['wins']}/{z['losses']}{st}"
    for k, q in out["pairs"].items():
        a_b, g = k.split(" | ")
        P.append(f"| {a_b} | {g} | {q['n_scenes']}/{q['of']}{'' if q['n_scenes'] == q['of'] else ' partial'} | {fd(q['auc20'])} | {fd(q['auc100'])} | {fd(q['final'])} |")
    os.makedirs(os.path.join(OV, "figures"), exist_ok=True)
    hdr = ["*自动生成（`experiments/exp6_v3/Y/scripts/yd_report.py`）。数据来自 Table 1 规范记录，官方 S10 覆盖。"
           "partial 表示该组有场景缺起点，均值只基于已完整的场景，不是完整结果。✱ 表示场景级 bootstrap CI 不含 0。*", ""]
    open(os.path.join(OV, "OVERNIGHT_TABLES.md"), "w").write("\n".join(["# OVERNIGHT_TABLES", ""] + hdr + ["## 主表", ""] + T + ["", "## 配对差", ""] + P) + "\n")
    for p, sel in ((os.path.join(DX, "reports", "G3_DEV_AND_ALL14.md"), lambda nm: nm.startswith(("H0", "H3", "D", "B0", "B2", "MAG"))),
                   (os.path.join(YX, "reports", "G3_DEV_AND_ALL14.md"), lambda nm: nm.startswith(("H0", "H3", "Y", "B0", "B2", "MAG")))):
        rows = [T[0], T[1]] + [t for t, (nm, row) in zip(T[2:], [(nm, r) for nm, r in out["models"].items() if r["splits"]["all14"]["episodes"]]) if sel(nm)]
        open(p, "w").write("\n".join([f"# G3_DEV_AND_ALL14（{os.path.basename(os.path.dirname(os.path.dirname(p)))}）", ""] + hdr + rows + ["", "## 配对差", ""] + P) + "\n")
    # figures: Dev-5 and All-14 coverage curves (complete scenes only), colosseum 20-50, compute vs AUC
    COL = {"V3": "#333333", "V3-H3": "#d6336c", "V3-D1-H3-NORISK": "#2a78d6", "V3-D2-DEPTH3-B3": "#1baf7a", "V3-D3-DEPTH3-B5": "#c9a227", "V3-D4-DEPTH4-B3": "#7b4fd6",
           "V3-Y45K-H0": "#8a8a8a", "V3-Y50K-H0": "#e67e22", "V3-YBEST-H3": "#a61e4d", "B0": "#9ec5fe", "B2": "#b2f2bb", "MAG-B": "#adb5bd"}
    curves = {}
    for (mid, s, k), r in rec.items():
        curves.setdefault(mid, {}).setdefault(s, {})[k] = np.asarray(r["coverage_official"])
    for g, tmax, nm in (("dev", 20, "coverage_0_20"), ("dev", 100, "coverage_0_100"), ("all14", 100, "coverage_all14_0_100")):
        fig, ax = plt.subplots(figsize=(7.4, 3.6))
        for name, mid in MODELS.items():
            d = curves.get(mid, {})
            sc = [s for s in SPLIT[g] if s in d and len(d[s]) == 5]
            if len(sc) != len(SPLIT[g]):
                continue
            mat = np.asarray([np.mean([d[s][k] for k in range(5)], 0) for s in sc])
            ax.plot(np.arange(tmax + 1), mat.mean(0)[:tmax + 1], color=COL[mid], lw=1.8 if mid in ("V3", "V3-H3") else 1.4, label=name)
        ax.set_xlim(0, tmax); ax.set_ylim(0, 1); ax.grid(alpha=0.25)
        ax.set_xlabel("exploration step"); ax.set_ylabel("coverage (MAGICIAN official)")
        ax.set_title(f"{g} (complete models only; scene-equal mean)", loc="left", fontsize=9)
        ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.savefig(os.path.join(OV, "figures", nm + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(os.path.join(OV, "figures", nm + ".pdf"), bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for name, mid in MODELS.items():
        d = curves.get(mid, {}).get("colosseum", {})
        if len(d) == 5:
            ax.plot(np.arange(20, 51), np.mean([d[k] for k in range(5)], 0)[20:51], color=COL[mid], lw=1.6, label=name)
    ax.set_xlim(20, 50); ax.grid(alpha=0.25); ax.set_xlabel("exploration step"); ax.set_ylabel("coverage (official)")
    ax.set_title("colosseum, steps 20-50 (5 starts)", loc="left", fontsize=9); ax.legend(fontsize=7)
    fig.savefig(os.path.join(OV, "figures", "colosseum_20_50.png"), dpi=200, bbox_inches="tight"); fig.savefig(os.path.join(OV, "figures", "colosseum_20_50.pdf"), bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, key in zip(axs, ("auc20", "auc100")):
        for name, row in out["models"].items():
            sp, rt = row["splits"]["dev"], row["runtime"]
            if sp["complete"] and rt[0] is not None:
                ax.scatter(rt[0], sp[key], color=COL[row["id"]], s=40)
                ax.annotate(name.split(" ")[0], (rt[0], sp[key]), fontsize=7.5, xytext=(4, 2), textcoords="offset points")
        ax.set_xlabel("planner decision s/step, median (shared GPUs)"); ax.set_ylabel(f"Dev-5 {key}"); ax.grid(alpha=0.25)
    fig.savefig(os.path.join(OV, "figures", "compute_vs_auc.png"), dpi=200, bbox_inches="tight"); fig.savefig(os.path.join(OV, "figures", "compute_vs_auc.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Y training timeline + validation
    run = os.path.join(YW, "runs", "main")
    if os.path.exists(os.path.join(run, "log.jsonl")):
        L = [json.loads(l) for l in open(os.path.join(run, "log.jsonl"))]
        ev = [json.loads(l) for l in open(os.path.join(run, "eval.jsonl"))] if os.path.exists(os.path.join(run, "eval.jsonl")) else []
        par = [json.loads(l) for l in open(os.path.join(R, "work_dirs", "exp6_v3", "task1_xattn", "runs", "main", "eval.jsonl"))]
        p40 = next((e for e in par if e["step"] == 40000), None)
        man = json.load(open(os.path.join(run, "checkpoint_manifest.json"))) if os.path.exists(os.path.join(run, "checkpoint_manifest.json")) else {}
        T2 = ["# TRAINING_TIMELINE — Y（自动生成）", "", "| step | loss（50 步均值） | fm | rec | kl | lr | grad norm | s/step | 显存 GB |", "|---|---|---|---|---|---|---|---|---|"]
        for x in L:
            if x["step"] % 1000 == 0 or x["step"] in (40001, L[-1]["step"]):
                T2.append(f"| {x['step']} | {x['loss']:.4f} | {x['fm']:.4f} | {x['rec']:.4f} | {x['kl']:.3f} | {x['lr']:.2e} | {x['grad_norm']:.3f} | {x['s_per_step']:.2f} | {x['mem_gb']:.1f} |")
        T2 += ["", "**checkpoint 清单**（`runs/main/checkpoint_manifest.json`）：", "", "```", json.dumps({"milestones": man.get("milestones"), "rolling": {k: v.get("sha256", "")[:16] for k, v in man.get("steps", {}).items()}}, indent=1), "```"]
        open(os.path.join(YX, "reports", "TRAINING_TIMELINE.md"), "w").write("\n".join(T2) + "\n")
        V = ["# VALIDATION_G1_G2 — Y 的固定验证集（VALSEL / S-c；自动生成）", "",
             "**指标**：FM 损失越低越好；fair ES 是样本质量（越低越好，同时也是原 V3 的 checkpoint 选择指标）；Brier / false wall 针对命中与无命中。数据来自训练器内置评估（EMA），不是 Test。", "",
             "| step | FM in_val | FM out_sc | fair ES in_val | fair ES out_sc | Brier out_sc | false wall out_sc | 选择分数（相对 5K） |", "|---|---|---|---|---|---|---|---|"]
        e5 = next((e for e in par if e["step"] == 5000), None)
        for e in ([p40] if p40 else []) + ev:
            es_i, es_o = e["samp_in_val"]["fair_es_arclength_vox"], e["samp_out_sc"]["fair_es_arclength_vox"]
            sc = es_i / e5["samp_in_val"]["fair_es_arclength_vox"] + es_o / e5["samp_out_sc"]["fair_es_arclength_vox"] if e5 else None
            V.append(f"| {e['step']}{' (父)' if e is p40 else ''} | {e['fm_in_val']:.4f} | {e['fm_out_sc']:.4f} | {es_i:.3f} | {es_o:.3f} | {e['samp_out_sc'].get('brier', float('nan')):.4f} | "
                     f"{e['samp_out_sc'].get('false_wall', float('nan')):.4f} | {f3(sc)} |")
        V += ["", "*G2 固定状态诊断本轮没有运行（预算优先给 G3）；Y 对决策的影响以 H0 下的 G3 闭环为准。*"]
        open(os.path.join(YX, "reports", "VALIDATION_G1_G2.md"), "w").write("\n".join(V) + "\n")
    print("\n".join(T))


if __name__ == "__main__":
    main()
