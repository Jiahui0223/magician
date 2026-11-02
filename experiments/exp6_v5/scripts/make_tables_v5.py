"""EXP6 V5 -- unified comparison tables (CPU) from the measured analyses: reports/g3_v5.json (eval_g3.py),
reports/g12_v5_s_final.json (analyze_g12_v5.py), reports/belief_metrics_eval.json (eval_belief_metrics.py), the S-c belief
monitor (runs/v5/eval.jsonl), the training manifests and G3 per-episode timing -> reports/V5_COMPARISON_TABLES.md and
reports/V5_RESOURCES.md. Usage: python make_tables_v5.py"""
import json
import os
import time

import numpy as np

R = "/bsuscratch/jiahuizhang/projects/occupancy/seen2scene"
E, W = f"{R}/experiments/exp6_v5", f"{R}/work_dirs/exp6_v5"
G3 = json.load(open(f"{E}/reports/g3_v5.json"))
G12 = json.load(open(f"{E}/reports/g12_v5_s_final.json")) if os.path.exists(f"{E}/reports/g12_v5_s_final.json") else None
BEL = json.load(open(f"{E}/reports/belief_metrics_eval.json")) if os.path.exists(f"{E}/reports/belief_metrics_eval.json") else None
NAME = {"MAG-G": "MAG-G", "MAG-B": "MAG-B", "V2O": "V2 Original", "B0": "B0", "B2": "B2", "E2D": "E2-D100", "T0": "V3-T0", "T1": "V3-T1", "V31": "V3.1",
        "V4": "V4@40k (A0)", "A1": "V4-A1 (D100 score, diag.)", "V4C": "**V4-Continue12k**", "V5": "**V5-Warm12k**"}
ORDER = ("MAG-G", "MAG-B", "V2O", "B2", "E2D", "T1", "V31", "V4", "A1", "V4C", "V5")


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def ci(p):
    return f"{p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]"


def main():
    L = ["# EXP6 V5 对比表（自动生成 `scripts/make_tables_v5.py`；全部为实测；评估场景已多次使用，**非盲测**）", ""]
    L += ["## Table A — G3 双域闭环（均值 = 场景 / 楼层均值的平均；[CI] 见 `g3_v5.md`）", "",
          "| 组 | MP3D zero-shot AUC@150 | Final | Cov@20 | 停滞 ep | Macarons++ AUC@100 | Final | Cov@20 |", "|---|---|---|---|---|---|---|---|"]
    mi, mo, st = G3["means"]["indoor"], G3["means"]["outdoor"], G3["stall_mp3d"]
    for g in ORDER:
        if g not in G3["groups"]:
            continue
        L.append(f"| {NAME[g]} | {f3(mi['auc'][g]['mean'])} | {f3(mi['final'][g]['mean'])} | {f3(mi['cov20'][g]['mean'])} | {st[g]['episodes_with_stall']} | "
                 f"{f3(mo['auc'][g]['mean'])} | {f3(mo['final'][g]['mean'])} | {f3(mo['cov20'][g]['mean'])} |")
    L += ["", "## Table B — 配对差（场景 / 楼层级 bootstrap 95 % CI；胜 / 负按场景）", "",
          "| 配对 | 数据集 | ΔAUC [CI] | 胜/负 | ΔFinal [CI] | 胜/负 | ΔCov@20 [CI] | 判定 |", "|---|---|---|---|---|---|---|---|"]
    for k, v in G3["pairs"].items():
        for d in ("indoor", "outdoor"):
            p = v[d]["paired"]
            L.append(f"| {k} | {'MP3D' if d == 'indoor' else 'Macarons++'} | {ci(p['auc'])} | {p['auc']['wins']}/{p['auc']['losses']} | {ci(p['final'])} | "
                     f"{p['final']['wins']}/{p['final']['losses']} | {ci(p['cov20'])} | {G3['decisions'][k]['verdict']} |")
    if G12:
        S = G12["summary"]
        ms = [m for m in ("B2", "E2D", "V4", "V4C", "V5") if m in G12["models"]]
        L += ["", "## Table C — G2 固定状态：被选候选的真实一步覆盖增益（pp）与 Top-1（365 态）", "",
              "| 集 | 阶段 | 状态数 | oracle | " + " | ".join(f"{m} 增益" for m in ms) + " | " + " | ".join(f"{m} Top-1" for m in ms) + " |", "|---|---|---|---|" + "---|" * (2 * len(ms))]
        for d, nm in (("valsel", "MP3D VALSEL"), ("dev3", "Macarons++ Dev-3")):
            for sub in ("early", "middle", "late", "all"):
                if not S[d].get(sub):
                    continue
                M = S[d][sub]["means"]
                L.append(f"| {nm} | {sub} | {S[d][sub]['n_states']} | {f3(M[f'{ms[0]}_oracle']['mean'])} | " + " | ".join(f3(M[f'{m}_gain']['mean']) for m in ms) + " | "
                         + " | ".join(f3(M[f'{m}_top1']['mean']) for m in ms) + " |")
        L += ["", "| 集 | " + " | ".join(f"{m} 与 V4 选择不同" for m in ms if m != "V4") + " |", "|---|" + "---|" * (len(ms) - 1)]
        for d, nm in (("valsel", "MP3D VALSEL"), ("dev3", "Macarons++ Dev-3")):
            L.append(f"| {nm} | " + " | ".join(f3(G12['pick_vs_V4'][d][m]['pick_differs_from_V4']) for m in ms if m != "V4") + " |")
        L += ["", "## Table D — G1 首中预测质量（同状态同噪声；MAE 为未知弧长体素，越小越好）", "",
              "| 集 | 指标 | " + " | ".join(ms) + " |", "|---|---|" + "---|" * len(ms)]
        for d, nm in (("valsel", "MP3D VALSEL"), ("dev3", "Macarons++ Dev-3")):
            M = S[d]["all"]["means"]
            for met in ("mae_unknown", "mae_new", "mae_known", "hit_agree", "rel_std"):
                if f"{ms[0]}_{met}" in M:
                    L.append(f"| {nm} | {met} | " + " | ".join(f3(M.get(f'{m}_{met}', {}).get('mean')) for m in ms) + " |")
    if BEL:
        L += ["", "## Table E — V5 多表面 Belief（评估端网格 GT，只用于指标；阈值 0.5；Δ = 8 VS）", "",
              "| 集 | 射线 | 射线 ≥ 2 表面 | 首表面召回 | **第二+ 召回** | 第二+（间隔 < 8 VS） | 8–32 | 32–128 | ≥ 128 | 网格 precision | recall | Brier | ECE | 首表面后虚假表面率 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, v in BEL["metrics"].items():
            for sub in ("all", "unknown_rays"):
                x = v[sub]
                b = x["recall_second_plus_by_gap"]
                L.append(f"| {k.split('eval_fixed/')[-1]} ({sub}) | {x['rays']:,} | {f3(x['rays_ge2_surfaces'])} | {f3(x['recall_first'])} | {f3(x['recall_second_plus'])} | "
                         f"{f3(b['gap<8vox']['recall'])} | {f3(b['gap8-32vox']['recall'])} | {f3(b['gap32-128vox']['recall'])} | {f3(b['gap>=128vox']['recall'])} | "
                         f"{f3(x['grid_precision'])} | {f3(x['grid_recall'])} | {f3(x['grid_brier'])} | {f3(x['grid_ece'])} | {f3(x['false_surface_rate_behind_first'])} |")
        L += ["", "首中模型（V4 / V4C / B2 / E2-D100：K = 4 个首中深度）不表达第二表面，第二+ 召回按构造为 0。"]
    ev = [json.loads(x) for x in open(f"{W}/runs/v5/eval.jsonl")] if os.path.exists(f"{W}/runs/v5/eval.jsonl") else []
    if ev:
        L += ["", "S-c（Train-6 留出轨迹，训练监控口径：存储表面 / 随机 / 首表面后查询）：", "", "| 步 | 首表面召回 | 第二+ 召回 | 负例正确率 | 首表面后负例正确率 |", "|---|---|---|---|---|"]
        for e in ev:
            b = e.get("surf_sc") or {}
            L.append(f"| {e['step']:,} | {f3(b.get('recall_first'))} | {f3(b.get('recall_second_plus'))} | {f3(b.get('neg_acc'))} | {f3(b.get('neg_acc_behind_first'))} |")
    open(f"{E}/reports/V5_COMPARISON_TABLES.md", "w").write("\n".join(L) + "\n")
    # ---- resources
    Rr = ["# EXP6 V5 资源（自动生成；L40S）", "", "| 项 | 节点 / 作业 | 墙钟 | GPU-h | 备注 |", "|---|---|---|---|---|"]
    for arm in ("v5", "v4c"):
        tl = open(f"{W}/logs/train_{arm}.log").read().splitlines() if os.path.exists(f"{W}/logs/train_{arm}.log") else []
        st = [x for x in tl if x.startswith("start ")]
        en = [x for x in tl if x.startswith("EXIT ")]
        if st and en:
            ts = lambda x: time.mktime(time.strptime(" ".join(x.split()[-2:]), "%Y-%m-%d %H:%M:%S"))  # noqa: E731
            h = (ts(en[-1]) - ts(st[0])) / 3600
            log = [json.loads(x) for x in open(f"{W}/runs/{arm}/log.jsonl")]
            sps = np.mean([l["s_per_step"] for l in log[1:]])
            Rr.append(f"| 训练 {arm}（12k 步，4 卡） | {st[0].split()[2]} | {h:.2f} h | {4 * h:.1f} | {sps:.3f} s/步，峰值 {max(l['mem_gb'] for l in log):.1f} GB/卡 |")
    for d, nm in (("indoor", "MP3D"), ("outdoor", "Macarons++")):
        M = G3["means"][d]
        Rr.append(f"| G3 推理 {nm}（每步时间 / 峰值显存，共享负载记录） | — | — | — | " + "；".join(f"{g} {M['time_per_move'][g]['mean']:.2f} s / {M['peak_gb'][g]['mean']:.1f} GB" for g in ("V4", "V4C", "V5") if g in M["time_per_move"]) + " |")
    Rr += ["", "标签生成：12 进程，进程时间合计约 4.2 h（墙钟 40 min）；G0 冒烟约 1.5 GPU-h；G3 / G1-G2 的墙钟见 `STATUS.md`。"]
    open(f"{E}/reports/V5_RESOURCES.md", "w").write("\n".join(Rr) + "\n")
    print("written V5_COMPARISON_TABLES.md, V5_RESOURCES.md")


if __name__ == "__main__":
    main()
