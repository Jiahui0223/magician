"""CVPR Table 1 -- reports/BASELINE_REPRODUCTION.md (CPU; generated, do not edit by hand).
Per scene: MAGICIAN paper numbers (SCENE_REGISTRY magician_paper, transcribed from Magician.pdf p.12 tables 6/7), the local
upstream-loop LMDB (reference only: our run of upstream test_magician_planning.py, unseeded; AUC = trapz/100) and the local
frozen-protocol MAG-B (this table; official and monotone curves), plus the regression replays (replay_check.log) and the
historical reported numbers re-computed from the imported records (monotone AUC / Final of Dev-3 + Test-3)."""
import json
import os

import numpy as np
import yaml

import aggregate as AG

T1 = AG.T1
G3 = os.path.join(AG.REPO, "work_dirs", "cvpr_results", "table1", "g3")


def main():
    sr = yaml.safe_load(open(os.path.join(T1, "SCENE_REGISTRY.yaml")))["scenes"]
    lm = json.load(open(os.path.join(T1, "lmdb_summary.json")))["rows"]
    rows = []
    for s in sr:
        sid = s["local_id"]
        paper = s.get("magician_paper") or {}
        pa, pf = (paper.get("auc") or {}).get("magician"), (paper.get("final") or {}).get("magician")
        L = [v for k, v in lm.items() if k.split("/")[0] == sid]
        la = float(np.mean([v["auc_exp1"] for v in L])) if L else None
        lf = float(np.mean([v["cov_100"] for v in L])) if L else None
        ao, fo, am, fm, n = [], [], [], [], 0
        for k in range(5):
            p = os.path.join(AG.EPI, "MAG-B", f"{sid}_s{k}.json")
            if os.path.exists(p):
                r = json.load(open(p))
                mo, mm = AG.metrics(r["coverage_official"]), AG.metrics(r["coverage_mono"])
                ao.append(mo["auc100"]), fo.append(mo["final"]), am.append(mm["auc100"]), fm.append(mm["final"])
                n += 1
        rows.append({"scene": sid, "split": s["split"], "paper": (pa, pf), "lmdb": (la, lf), "n": n,
                     "official": (np.mean(ao) if ao else None, np.mean(fo) if fo else None), "mono": (np.mean(am) if am else None, np.mean(fm) if fm else None)})
    f = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    out = ["# BASELINE_REPRODUCTION — MAGICIAN 基线：论文值 / 本地上游 LMDB / 本地冻结协议（自动生成，`scripts/baseline_report.py`）", "",
           "**三类数值不可混用**：",
           "- 论文值：Magician.pdf 第 12 页表 6 / 7，AUC 的归一化方式论文没有说明；",
           "- LMDB：我们本地运行上游 `test_magician_planning.py` 的轨迹（未设种子，只作参照），AUC = trapz / 100；",
           "- 本表 MAG-B：冻结 M 协议，带种子，与我们的模型同一个模拟器和评价器。", "",
           "| 场景 | 划分 | 论文 AUC / Final | LMDB AUC / Final | 本地 MAG-B 官方 AUC / Final | 本地 MAG-B 单调 AUC / Final | MAG-B 起点数 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['scene']} | {r['split']} | {f(r['paper'][0])} / {f(r['paper'][1])} | {f(r['lmdb'][0])} / {f(r['lmdb'][1])} | "
                   f"{f(r['official'][0])} / {f(r['official'][1])} | {f(r['mono'][0])} / {f(r['mono'][1])} | {r['n']} |")
    comp = [r for r in rows if r["n"] == 5 and r["lmdb"][0] is not None and r["paper"][0] is not None]
    if comp:
        mean = lambda i, j: float(np.mean([r[i][j] for r in comp]))  # noqa: E731
        out += ["", f"**{len(comp)} 个场景均值**（同时有论文值、LMDB 与 5 个本地 MAG-B 起点的场景）：",
                f"- 论文：AUC {mean('paper', 0):.3f} / Final {mean('paper', 1):.3f}",
                f"- LMDB：AUC {mean('lmdb', 0):.3f} / Final {mean('lmdb', 1):.3f}",
                f"- 本地 MAG-B（官方口径）：AUC {mean('official', 0):.3f} / Final {mean('official', 1):.3f}",
                f"- 本地 MAG-B（单调口径）：AUC {mean('mono', 0):.3f} / Final {mean('mono', 1):.3f}"]
    rc = os.path.join(G3, "logs", "replay_check.log")
    if os.path.exists(rc):
        out += ["", "## 回归重放（当前代码重跑 barts_s0，与历史 episode 比较）", "", "```"] + [l.rstrip() for l in open(rc) if "equal" in l or "DONE" in l] + ["```"]
    out += ["", "## 历史报告值（单调口径，Dev-3 + Test-3）由导入记录重算", "", "| 模型 | 单调 AUC@100 | 单调 Final | 官方 AUC@100 | 官方 Final |", "|---|---|---|---|---|"]
    for m in ("MAG-B", "MAG-G", "V2O", "B0", "B2", "B1", "E2S", "E2D", "V3", "V3.1", "V4", "V4C", "V5"):
        vals = {"am": [], "fm": [], "ao": [], "fo": []}
        for sc in ("barts", "pantheon", "colosseum", "alhambra", "bridge", "neuschwanstein"):
            ep = [json.load(open(os.path.join(AG.EPI, m, f"{sc}_s{k}.json"))) for k in range(5) if os.path.exists(os.path.join(AG.EPI, m, f"{sc}_s{k}.json"))]
            if len(ep) != 5:
                continue
            vals["am"].append(np.mean([AG.metrics(e["coverage_mono"])["auc100"] for e in ep]))
            vals["fm"].append(np.mean([AG.metrics(e["coverage_mono"])["final"] for e in ep]))
            vals["ao"].append(np.mean([AG.metrics(e["coverage_official"])["auc100"] for e in ep]))
            vals["fo"].append(np.mean([AG.metrics(e["coverage_official"])["final"] for e in ep]))
        if len(vals["am"]) == 6:
            out.append(f"| {m} | {np.mean(vals['am']):.3f} | {np.mean(vals['fm']):.3f} | {np.mean(vals['ao']):.3f} | {np.mean(vals['fo']):.3f} |")
    os.makedirs(os.path.join(T1, "reports"), exist_ok=True)
    open(os.path.join(T1, "reports", "BASELINE_REPRODUCTION.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out[:30]))


if __name__ == "__main__":
    main()
